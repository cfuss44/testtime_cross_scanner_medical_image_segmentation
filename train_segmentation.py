# ==================================================================
# import 
# ==================================================================
import logging
import os.path
import time
import shutil
import tensorflow as tf
import numpy as np
import utils
import model as model
import config.system as sys_config
from random import randint
import data_hcp

import pdb

# ==================================================================
# Set the config file of the experiment you want to run here:
# ==================================================================
from experiments import branched_network_config as exp_config

# ==================================================================
# setup logging
# ==================================================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
log_dir = os.path.join(sys_config.log_root, exp_config.experiment_name)
logging.info('Logging directory: %s' %log_dir)

# ==================================================================
# main function for training
# ==================================================================
def run_training(continue_run):

    # ============================
    # log experiment details
    # ============================
    logging.info('============================================================')
    logging.info('EXPERIMENT NAME: %s' % exp_config.experiment_name)

    # ============================
    # Initialize step number - this is number of mini-batch runs
    # ============================
    init_step = 0

    # ============================
    # if continue_run is set to True, load the model parameters saved earlier
    # else start training from scratch
    # ============================
    if continue_run:
        logging.info('============================================================')
        logging.info('Continuing previous run')
        try:
            init_checkpoint_path = utils.get_latest_model_checkpoint_path(log_dir, 'models/model.ckpt')
            logging.info('Checkpoint path: %s' % init_checkpoint_path)
            init_step = int(init_checkpoint_path.split('/')[-1].split('-')[-1]) + 1  # plus 1 as otherwise starts with eval
            logging.info('Latest step was: %d' % init_step)
        except:
            logging.warning('Did not find init checkpoint. Maybe first run failed. Disabling continue mode...')
            continue_run = False
            init_step = 0
        logging.info('============================================================')

    # ============================
    # Load data
    # ============================   
    logging.info('============================================================')
    logging.info('Loading data...')
    logging.info('Reading HCP - 3T - T1 images...')    
    logging.info('Data root directory: ' + sys_config.orig_data_root_hcp)
    imtr, gttr = data_hcp.load_data(sys_config.orig_data_root_hcp, sys_config.preproc_folder_hcp, 'T1', 1, exp_config.train_size+1)
    imvl, gtvl = data_hcp.load_data(sys_config.orig_data_root_hcp, sys_config.preproc_folder_hcp, 'T1', exp_config.train_size+1, exp_config.train_size+exp_config.val_size+1)
    logging.info('Training Images: %s' %str(imtr.shape)) # expected: [num_slices, img_size_x, img_size_y]
    logging.info('Training Labels: %s' %str(gttr.shape)) # expected: [num_slices, img_size_x, img_size_y]
    logging.info('Validation Images: %s' %str(imvl.shape))
    logging.info('Validation Labels: %s' %str(gtvl.shape))
    logging.info('============================================================')
    
    # ================================================================
    # Define the segmentation network here
    # ================================================================
    mask = ~(imtr == 0).all(axis=(1,2))
    imtr = imtr[mask]
    gttr = gttr[mask]
    
    mask = ~(imvl == 0).all(axis=(1,2))
    imvl = imvl[mask]
    gtvl = gtvl[mask]
            
    # ================================================================
    # build the TF graph
    # ================================================================
    with tf.Graph().as_default():

        # ================================================================
        # create placeholders
        # ================================================================
        logging.info('Creating placeholders...')
        # Placeholders for the images and labels
        image_tensor_shape = [exp_config.batch_size] + list(exp_config.image_size) + [1]
        images_pl = tf.compat.v1.placeholder(tf.float32, shape=image_tensor_shape, name = 'images')
        labels_tensor_shape_segmentation = [exp_config.batch_size] + list(exp_config.image_size)
        labels_tensor_shape_self_supervised = [exp_config.batch_size] + [exp_config.num_rotation_values]
        labels_segmentation_pl = tf.compat.v1.placeholder(tf.uint8, shape=labels_tensor_shape_segmentation, name = 'labels_segmentation')
        labels_self_supervised_pl = tf.compat.v1.placeholder(tf.float16, shape=labels_tensor_shape_self_supervised, name = 'labels_self_supervised')
        # Placeholders for the learning rate and to indicate whether the model is being trained or tested
        learning_rate_pl = tf.compat.v1.placeholder(tf.float32, shape=[], name = 'learning_rate')
        training_pl = tf.compat.v1.placeholder(tf.bool, shape=[], name = 'training_or_testing')

        # ================================================================
        # Define the segmentation network here
        # ================================================================
        logits_segmentation = exp_config.model_handle_segmentation(images_pl, image_tensor_shape, training_pl, exp_config.nlabels)
        
        # ================================================================
        # determine trainable variables
        # ================================================================
        segmentation_vars = []
        self_supervised_vars = []
        test_time_opt_vars = []
        
        for v in tf.compat.v1.trainable_variables():
            var_name = v.name
            if 'rotation' in var_name:
                self_supervised_vars.append(v)
            elif 'segmentation' in var_name:
                segmentation_vars.append(v)
            elif 'normalization' in var_name:
                test_time_opt_vars.append(v)

        if exp_config.debug is True:
            logging.info('================================')
            logging.info('List of trainable variables in the graph:')
            for v in tf.compat.v1.trainable_variables(): logging.info(v.name)
            logging.info('================================')
            logging.info('List of all segmentation variables:')
            for v in segmentation_vars: logging.info(v.name)
            logging.info('================================')
            logging.info('List of all self-supervised variables:')
            for v in self_supervised_vars: logging.info(v.name)
            logging.info('================================')
            logging.info('List of all test time variables:')
            for v in test_time_opt_vars: logging.info(v.name)

        # ================================================================
        # Add ops for calculation of the training loss
        # ================================================================
        loss_segmentation = model.loss(logits_segmentation,
                          labels_segmentation_pl,
                          nlabels=exp_config.nlabels,
                          loss_type=exp_config.loss_type)
        tf.compat.v1.summary.scalar('loss_segmentation', loss_segmentation)
        
        # ================================================================
        # Add optimization ops.
        # ================================================================
        train_op_train_time = model.training_step(loss_segmentation, segmentation_vars, exp_config.optimizer_handle_training, learning_rate_pl)
        
        # ================================================================
        # Add ops for model evaluation
        # ================================================================
        eval_loss_segmentation = model.evaluation(logits_segmentation,
                                     labels_segmentation_pl,
                                     images_pl,
                                     nlabels = exp_config.nlabels,
                                     loss_type = exp_config.loss_type)
        
        # ================================================================
        # Build the summary Tensor based on the TF collection of Summaries.
        # ================================================================
        summary = tf.compat.v1.summary.merge_all()

        # ================================================================
        # Add init ops
        # ================================================================
        init_g = tf.compat.v1.global_variables_initializer()
        init_l = tf.compat.v1.local_variables_initializer()
        
        # ================================================================
        # Find if any vars are uninitialized
        # ================================================================
        logging.info('Adding the op to get a list of initialized variables...')
        uninit_vars = tf.compat.v1.report_uninitialized_variables()

        # ================================================================
        # create savers for each domain
        # ================================================================
        max_to_keep = 15
        saver = tf.compat.v1.train.Saver(max_to_keep=max_to_keep)
        saver_best_da = tf.compat.v1.train.Saver()

        # ================================================================
        # Create session
        # ================================================================
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        sess = tf.compat.v1.Session(config = config)

        # ================================================================
        # create a summary writer
        # ================================================================
        summary_writer = tf.compat.v1.summary.FileWriter(log_dir, sess.graph)

        # ================================================================
        # summaries of the training errors
        # ================================================================        
        tr_error_seg = tf.compat.v1.placeholder(tf.float32, shape=[], name='tr_error_seg')
        tr_error_summary_seg = tf.compat.v1.summary.scalar('training/loss_seg', tr_error_seg)
        tr_dice_seg = tf.compat.v1.placeholder(tf.float32, shape=[], name='tr_dice_seg')
        tr_dice_summary_seg = tf.compat.v1.summary.scalar('training/dice_seg', tr_dice_seg)
        tr_summary_seg = tf.compat.v1.summary.merge([tr_error_summary_seg, tr_dice_summary_seg])
        
        # ================================================================
        # summaries of the validation errors
        # ================================================================
        vl_error_seg = tf.compat.v1.placeholder(tf.float32, shape=[], name='vl_error_seg')
        vl_error_summary_seg = tf.compat.v1.summary.scalar('validation/loss_seg', vl_error_seg)
        vl_dice_seg = tf.compat.v1.placeholder(tf.float32, shape=[], name='vl_dice_seg')
        vl_dice_summary_seg = tf.compat.v1.summary.scalar('validation/dice_seg', vl_dice_seg)
        vl_summary_seg = tf.compat.v1.summary.merge([vl_error_summary_seg, vl_dice_summary_seg])
       
        # ================================================================
        # freeze the graph before execution
        # ================================================================
        logging.info('Freezing the graph now!')
        tf.compat.v1.get_default_graph().finalize()

        # ================================================================
        # Run the Op to initialize the variables.
        # ================================================================
        logging.info('============================================================')
        logging.info('initializing all variables...')
        sess.run(init_g)
        sess.run(init_l)

        # ================================================================
        # print names of uninitialized variables
        # ================================================================
        logging.info('============================================================')
        logging.info('This is the list of uninitialized variables:' )
        uninit_variables = sess.run(uninit_vars)
        for v in uninit_variables: logging.info(v)

        # ================================================================
        # continue run from a saved checkpoint
        # ================================================================
        if continue_run:
            # Restore session
            logging.info('============================================================')
            logging.info('Restroring session from: %s' %init_checkpoint_path)
            saver.restore(sess, init_checkpoint_path)

        # ================================================================
        # ================================================================        
        step = init_step
        curr_lr = exp_config.learning_rate
        best_da = 0

        # ================================================================
        # run training epochs
        # ================================================================
        for epoch in range(exp_config.max_epochs):

            logging.info('============================================================')
            logging.info('EPOCH %d' % epoch)

            
            for batch in iterate_minibatches(imtr, gttr, batch_size = exp_config.batch_size):
                
                curr_lr = exp_config.learning_rate
                start_time = time.time()
                x, y_seg, y_ss = batch

                if step == 0:
                    x_s = x
                    y_seg_s = y_seg
                    y_ss_s = y_ss

                # ===========================
                # avoid incomplete batches
                # ===========================
                if y_seg.shape[0] < exp_config.batch_size:
                    step += 1
                    continue

                feed_dict = {images_pl: x,
                            labels_segmentation_pl: y_seg,
                            labels_self_supervised_pl: y_ss,
                            learning_rate_pl: curr_lr,
                            training_pl: True}
                
                # ===========================
                # update vars
                # ===========================
                _, loss_value = sess.run([train_op_train_time, loss_segmentation], feed_dict=feed_dict)

                # ===========================
                # compute the time for this mini-batch computation
                # ===========================
                duration = time.time() - start_time

                # ===========================
                # write the summaries and print an overview fairly often
                # ===========================
                if (step+1) % exp_config.summary_writing_frequency == 0:                    
                    logging.info('Step %d: loss = %.2f (%.3f sec for the last step)' % (step+1, loss_value, duration))
                    
                    # ===========================
                    # print values of a parameter (to debug)
                    # ===========================
                    if exp_config.debug is True:
                        var_value = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES)[0].eval(session = sess) # can add name if you only want parameters with a specific name
                        logging.info('value of one of the parameters %f' % var_value[0,0,0,0])
                                            
                    # ===========================
                    # Update the events file
                    # ===========================
                    summary_str = sess.run(summary, feed_dict=feed_dict)
                    summary_writer.add_summary(summary_str, step)
                    summary_writer.flush()

                # ===========================
                # compute the loss on the entire training set
                # ===========================
                if step % exp_config.train_eval_frequency == 0:

                    logging.info('Training Data Eval:')
                    [train_loss_seg, train_dice_seg] = do_eval(sess,
                                                            eval_loss_segmentation,
                                                            images_pl,
                                                            labels_segmentation_pl,
                                                            training_pl,
                                                            imtr,
                                                            gttr,
                                                            exp_config.batch_size)                    

                    tr_summary_msg = sess.run(tr_summary_seg, feed_dict={tr_error_seg: train_loss_seg, tr_dice_seg: train_dice_seg})
                    summary_writer.add_summary(tr_summary_msg, step)
                                        
                # ===========================
                # Save a checkpoint periodically
                # ===========================
                if step % exp_config.save_frequency == 0:

                    checkpoint_file = os.path.join(log_dir, 'models/model.ckpt')
                    saver.save(sess, checkpoint_file, global_step=step)

                # ===========================
                # Evaluate the model periodically
                # ===========================
                if step % exp_config.val_eval_frequency == 0:
                    
                    # ===========================
                    # Evaluate against the validation set of each domain
                    # ===========================
                    logging.info('Validation Data Eval:')
                    [val_loss_seg, val_dice_seg] = do_eval(sess,
                                                        eval_loss_segmentation,
                                                        images_pl,
                                                        labels_segmentation_pl,
                                                        training_pl,
                                                        imvl,
                                                        gtvl,
                                                        exp_config.batch_size)                    

                    vl_summary_msg = sess.run(vl_summary_seg, feed_dict={vl_error_seg: val_loss_seg, vl_dice_seg: val_dice_seg})
                    summary_writer.add_summary(vl_summary_msg, step)
                                    
                    # ===========================
                    # save model if the val dice/accuracy is the best yet
                    # ===========================
                    if val_dice_seg > best_da:
                        best_da = val_dice_seg
                        best_file = os.path.join(log_dir, 'models/best_dice.ckpt')
                        saver_best_da.save(sess, best_file, global_step=step)
                        logging.info('Found new average best dice on validation sets! - %f -  Saving model.' % val_dice_seg)

                step += 1

        sess.close()

def do_eval_single(sess,
            eval_loss,
            images_placeholder,
            labels_placeholder,
            training_time_placeholder,
            x,
            y):
    
    '''
    Function for running the evaluations every X iterations on the training and validation sets. 
    :param sess: The current tf session 
    :param eval_loss: The placeholder containing the eval loss
    :param images_placeholder: Placeholder for the images
    :param labels_placeholder: Placeholder for the masks
    :param training_time_placeholder: Placeholder toggling the training/testing mode. 
    :param x: A numpy array or h5py dataset containing the images
    :param y: A numpy array or h45py dataset containing the corresponding labels
    :return: The loss and dice/accuracy over the image(s)
    '''

    feed_dict = {images_placeholder: x,
                labels_placeholder: y,
                training_time_placeholder: False}

    closs, cda = sess.run(eval_loss, feed_dict=feed_dict)

    return closs, cda

# ==================================================================
# ==================================================================
def do_eval(sess,
            eval_loss_seg,
            images_placeholder,
            labels_seg_placeholder,
            training_time_placeholder,
            images,
            labels,
            batch_size):

    '''
    Function for running the evaluations every X iterations on the training and validation sets. 
    :param sess: The current tf session 
    :param eval_loss: The placeholder containing the eval loss
    :param images_placeholder: Placeholder for the images
    :param labels_placeholder: Placeholder for the masks/labels
    :param training_time_placeholder: Placeholder toggling the training/testing mode. 
    :param images: A numpy array or h5py dataset containing the images
    :param labels: A numpy array or h45py dataset containing the corresponding labels 
    :param batch_size: The batch size to use. 
    :return: The average loss (as defined in the experiment), and the average dice over all `images`. 
    '''

    loss_ii = 0
    dice_ii = 0
    num_batches = 0

    for batch in iterate_minibatches(images, labels, batch_size=batch_size):

        x, y_seg, y_ss = batch

        closs_seg, cdice = do_eval_single(sess,
                                    eval_loss_seg,
                                    images_placeholder,
                                    labels_seg_placeholder,
                                    training_time_placeholder,
                                    x,
                                    y_seg)

        
        loss_ii += closs_seg
        dice_ii += cdice

        num_batches += 1

    avg_loss_seg = loss_ii / num_batches
    avg_dice = dice_ii / num_batches

    logging.info('  Average loss segmentation: %0.04f, average dice segmentation: %0.04f' % (avg_loss_seg, avg_dice))

    return avg_loss_seg, avg_dice

# ==================================================================
# ==================================================================
def iterate_minibatches(images, labels, batch_size):
    '''
    Function to create mini batches containing rotated images from the dataset of a certain batch size 
    :param images: hdf5 dataset
    :param labels: hdf5 dataset
    :param batch_size: batch size
    :return: mini batches
    '''
    # ===========================
    # generate indices to randomly select slices in each minibatch
    # ===========================
    n_images = images.shape[0]
    random_indices = np.arange(n_images)
    np.random.shuffle(random_indices)

    # ===========================
    # generate batches with rotated images
    # ===========================
    for b_i in range(0, n_images, batch_size):

        if b_i + batch_size > n_images:
            continue
        batch_indices = random_indices[b_i:b_i+batch_size]

        batch_images = []
        batch_labels_seg = []
        batch_labels_ss = []

        for n in range(batch_size):
            batch_index = batch_indices[n]
            image = images[batch_index]
            label_seg = labels[batch_index]
            r = randint(0,3)
            batch_images.append(np.rot90(image, r))
            batch_labels_seg.append(np.rot90(label_seg, r))
            l = np.zeros((4,1))
            l[r] = 1
            batch_labels_ss.append(l)

        X = np.concatenate([arr[np.newaxis] for arr in batch_images])
        X = np.expand_dims(X, axis=-1)
        y_seg = np.concatenate([arr[np.newaxis] for arr in batch_labels_seg])
        y_ss = np.concatenate([arr[np.newaxis] for arr in batch_labels_ss])
        y_ss = np.squeeze(y_ss)
        
        yield X, y_seg, y_ss

# ==================================================================
# ==================================================================
def main():
    
    # ===========================
    # Create dir if it does not exist
    # ===========================
    continue_run = exp_config.continue_run
    if not tf.io.gfile.exists(log_dir):
        tf.io.gfile.makedirs(log_dir)
        tf.io.gfile.makedirs(log_dir + '/models')
        continue_run = False

    # ===========================
    # Copy experiment config file
    # ===========================
    shutil.copy(exp_config.__file__, log_dir)

    run_training(continue_run)

# ==================================================================
# ==================================================================
if __name__ == '__main__':
    main()




