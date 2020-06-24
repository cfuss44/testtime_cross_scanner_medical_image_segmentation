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
import data_hcp
import data_abide
from random import randint

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
def run_training():

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
    # Determine the data set
    # ============================
    target_data = True

    # ============================
    # Load data
    # ============================   

    if target_data:
        hcp = True
        if hcp:
            logging.info('============================================================')
            logging.info('Loading data...')
            logging.info('Reading HCP - 3T - T2 images...')    
            logging.info('Data root directory: ' + sys_config.orig_data_root_hcp)
            imts, gtts = data_hcp.load_data(sys_config.orig_data_root_hcp, sys_config.preproc_folder_hcp, 'T2', 1, exp_config.test_size+1)
            logging.info('Test Images: %s' %str(imts.shape)) # expected: [num_slices, img_size_x, img_size_y]
            logging.info('Test Labels: %s' %str(gtts.shape)) # expected: [num_slices, img_size_x, img_size_y]
            logging.info('============================================================')
        else:
            logging.info('============================================================')
            logging.info('Loading data...')
            logging.info('Reading ABIDE caltech...')    
            logging.info('Data root directory: ' + sys_config.orig_data_root_abide)
            imts, gtts = data_abide.load_data(sys_config.orig_data_root_abide, sys_config.preproc_folder_abide, 1, exp_config.test_size+1)
            logging.info('Test Images: %s' %str(imts.shape)) # expected: [num_slices, img_size_x, img_size_y]
            logging.info('Test Labels: %s' %str(gtts.shape)) # expected: [num_slices, img_size_x, img_size_y]
            logging.info('============================================================')        
    else:
        logging.info('============================================================')
        logging.info('Loading data...')
        logging.info('Reading HCP - 3T - T1 images...')    
        logging.info('Data root directory: ' + sys_config.orig_data_root_hcp)
        imts, gtts = data_hcp.load_data(sys_config.orig_data_root_hcp, sys_config.preproc_folder_hcp, 'T1', exp_config.train_size+exp_config.val_size+1, exp_config.train_size+exp_config.val_size+exp_config.test_size+1)
        logging.info('Test Images: %s' %str(imts.shape)) # expected: [num_slices, img_size_x, img_size_y]
        logging.info('Test Labels: %s' %str(gtts.shape)) # expected: [num_slices, img_size_x, img_size_y]
        logging.info('============================================================')

    # ============================
    # Remove exclusively black images
    # ============================
    mask = ~(imts == 0).all(axis=(1,2))
    imts = imts[mask]
    gtts = gtts[mask]
    
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
        # Define the image normalization function - these parameters will be updated for each test image
        # ================================================================
        image_normalized = exp_config.model_handle_normalizer(images_pl, image_tensor_shape)
        
        # ================================================================
        # Define the segmentation network here
        # ================================================================
        logits_segmentation = exp_config.model_handle_segmentation(image_normalized, image_tensor_shape, training_pl, exp_config.nlabels)
        
        # ================================================================
        # Define the self-supervised network here
        # ================================================================
        logits_self_supervised = exp_config.model_handle_rotation(image_normalized, image_tensor_shape, training_pl)

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
        loss_self_supervised = tf.reduce_mean(exp_config.loss_handle_self_supervised(labels_self_supervised_pl, logits_self_supervised))
        tf.compat.v1.summary.scalar('loss_self_supervised', loss_self_supervised)

        # ================================================================
        # Add optimization ops.
        # ================================================================
        train_op_test_time = model.training_step(loss_self_supervised, test_time_opt_vars, exp_config.optimizer_handle_test_time, learning_rate_pl)
        
        # ================================================================
        # Add ops for model evaluation
        # ================================================================
        eval_loss_segmentation = model.evaluation(logits_segmentation,
                                     labels_segmentation_pl,
                                     images_pl,
                                     nlabels = exp_config.nlabels,
                                     loss_type = exp_config.loss_type)
        eval_loss_self_supervised = model.evaluation_rotate(logits_self_supervised, labels_self_supervised_pl)

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
        saver_seg = tf.compat.v1.train.Saver(var_list=segmentation_vars)
        saver_ss = tf.compat.v1.train.Saver(var_list=self_supervised_vars)

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
        # summaries of the test errors
        # ================================================================        
        ts_error_seg = tf.compat.v1.placeholder(tf.float32, shape=[], name='ts_error_seg')
        ts_error_summary_seg = tf.compat.v1.summary.scalar('test/loss_seg', ts_error_seg)
        ts_dice_seg = tf.compat.v1.placeholder(tf.float32, shape=[], name='ts_dice_seg')
        ts_dice_summary_seg = tf.compat.v1.summary.scalar('test/dice_seg', ts_dice_seg)
        ts_summary_seg = tf.compat.v1.summary.merge([ts_error_summary_seg, ts_dice_summary_seg])

        ts_error_ss = tf.compat.v1.placeholder(tf.float32, shape=[], name='ts_error_ss')      
        ts_error_summary_ss = tf.compat.v1.summary.scalar('test/loss_ss', ts_error_ss)
        ts_acc_ss = tf.compat.v1.placeholder(tf.float32, shape=[], name='ts_acc_ss')
        ts_acc_summary_ss = tf.compat.v1.summary.scalar('test/acc_ss', ts_acc_ss)
        ts_summary_ss = tf.compat.v1.summary.merge([ts_error_summary_ss, ts_acc_summary_ss])
        
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
        # restore shared weights
        # ================================================================
        logging.info('============================================================')
        logging.info('Restore segmentation...')
        model_path = os.path.join(sys_config.log_root, 'Initial_training_segmentation')
        checkpoint_path = utils.get_latest_model_checkpoint_path(model_path, 'models/best_dice.ckpt')
        logging.info('Restroring session from: %s' %checkpoint_path)
        saver_seg.restore(sess, checkpoint_path)

        logging.info('============================================================')
        logging.info('Restore self-supervised...')
        model_path = os.path.join(sys_config.log_root, 'Initial_training_rotation')
        checkpoint_path = utils.get_latest_model_checkpoint_path(model_path, 'models/best_dice.ckpt')
        logging.info('Restroring session from: %s' %checkpoint_path)
        saver_ss.restore(sess, checkpoint_path)

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

            for batch in iterate_minibatches(imts, gtts, batch_size=exp_config.batch_size):
                curr_lr = exp_config.learning_rate
                start_time = time.time()
                x, y_seg, y_ss = batch

                # ===========================
                # avoid incomplete batches
                # ===========================
                if y_seg.shape[0] < exp_config.batch_size:
                    step += 1
                    continue

                feed_dict = {images_pl: x,
                            labels_self_supervised_pl: y_ss,
                            learning_rate_pl: curr_lr,
                            training_pl: True}

                # ===========================
                # update vars
                # ===========================
                _, loss_value = sess.run([train_op_test_time, loss_self_supervised], feed_dict=feed_dict)

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
                # compute the loss on the entire test set
                # ===========================
                if step % exp_config.train_eval_frequency == 0:

                    logging.info('Test Data Eval:')
                    [test_loss_seg, test_dice_seg, test_loss_ss, test_acc_ss] = do_eval(sess,
                                                            eval_loss_segmentation,
                                                            eval_loss_self_supervised,
                                                            images_pl,
                                                            labels_segmentation_pl,
                                                            labels_self_supervised_pl,
                                                            training_pl,
                                                            imts,
                                                            gtts,
                                                            exp_config.batch_size)                    

                    ts_summary_msg = sess.run(ts_summary_seg, feed_dict={ts_error_seg: test_loss_seg, ts_dice_seg: test_dice_seg})
                    summary_writer.add_summary(ts_summary_msg, step)
                    ts_summary_msg = sess.run(ts_summary_ss, feed_dict={ts_error_ss: test_loss_ss, ts_acc_ss: test_acc_ss})
                    summary_writer.add_summary(ts_summary_msg, step)
                    
                # ===========================
                # Save a checkpoint periodically
                # ===========================
                if step % exp_config.save_frequency == 0:

                    checkpoint_file = os.path.join(log_dir, 'models/model.ckpt')
                    saver.save(sess, checkpoint_file, global_step=step)

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
            eval_loss_ss,
            images_placeholder,
            labels_seg_placeholder,
            labels_ss_placeholder,
            training_time_placeholder,
            images,
            labels,
            batch_size):

    '''
    Function for running the evaluations every X iterations on the training and validation sets. 
    :param sess: The current tf session 
    :param eval_loss_seg: The placeholder containing the eval loss for the segmentation task
    :param eval_loss_ss: The placeholder containing the eval loss for the self-supervised task
    :param images_placeholder: Placeholder for the images
    :param labels_seg_placeholder: Placeholder for the masks/labels for the segmentation task
    :param labels_ss_placeholder: Placeholder for the masks/labels for the self-supervised task
    :param training_time_placeholder: Placeholder toggling the training/testing mode. 
    :param images: A numpy array or h5py dataset containing the images
    :param labels: A numpy array or h45py dataset containing the corresponding labels 
    :param batch_size: The batch size to use. 
    :return: The average losses for both tasks (as defined in the experiment), the average dice over all `images`,
    and the accuracy of the self-supervised task. 
    '''

    loss_seg_ii = 0
    loss_ss_ii = 0
    dice_ii = 0
    acc_ii = 0
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

        closs_ss, cacc = do_eval_single(sess,
                                    eval_loss_ss,
                                    images_placeholder,
                                    labels_ss_placeholder,
                                    training_time_placeholder,
                                    x,
                                    y_ss)
        

        loss_seg_ii += closs_seg
        loss_ss_ii += closs_ss
        dice_ii += cdice
        acc, acc_op = cacc
        acc_ii += acc_op
        num_batches += 1

    avg_loss_seg = loss_seg_ii / num_batches
    avg_loss_ss = loss_ss_ii / num_batches
    avg_dice = dice_ii / num_batches
    avg_acc = acc_ii / num_batches

    logging.info('  Average loss segmentation: %0.04f, average dice segmentation: %0.04f, average loss self supervised: %0.04f, average accuracy self supervised: %0.04f' % (avg_loss_seg, avg_dice, avg_loss_ss, avg_acc))

    return avg_loss_seg, avg_dice, avg_loss_ss, avg_acc

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
    # generate batches with rotated images
    # ===========================
    n_images = images.shape[0]
    random_indices = np.arange(n_images)
    np.random.shuffle(random_indices)

    # ===========================
    # using only a fraction of the batches in each epoch
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
        y_ss = np.squeeze(y_ss, axis=2)

        yield X, y_seg, y_ss

# ==================================================================
# ==================================================================
def main():
    
    # ===========================
    # Create dir if it does not exist
    # ===========================
    if not tf.io.gfile.exists(log_dir):
        tf.io.gfile.makedirs(log_dir)
        tf.io.gfile.makedirs(log_dir + '/models')

    # ===========================
    # Copy experiment config file
    # ===========================
    shutil.copy(exp_config.__file__, log_dir)

    run_training()

# ==================================================================
# ==================================================================
if __name__ == '__main__':
    main()




