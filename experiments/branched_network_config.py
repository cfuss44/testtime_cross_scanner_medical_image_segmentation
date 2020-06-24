import model_zoo
import tensorflow as tf

# ======================================================================
# Model settings
# ======================================================================
model_handle_segmentation = model_zoo.unet2D_segmentation
model_handle_rotation = model_zoo.net2D_rotation
model_handle_normalizer = model_zoo.net2D_normalization
model_handle_autoencode = model_zoo.net2D_autoencoder

# ======================================================================
# data settings
# ======================================================================
data_mode = '2D'
image_size = (256, 256) # applies to HCP and ABIDE images
nlabels = 3

train_size = 30     # size of training dataset
val_size = 5        # size of validation dataset
test_size = 10      # size of test dataset

# ======================================================================
# IMPORTANT: Before running 'train' files set testtime to False
# IMPORTANT: Before running 'test' files set testtime to True
# ======================================================================
testtime = False

# ======================================================================
# training and testing settings
# ======================================================================

if testtime:
    max_epochs = 10
    experiment_name = 'Testing'
    batch_size = 32
else:
    max_epochs = 30
    experiment_name = 'Training'
    batch_size = 32

learning_rate = 1e-4
optimizer_handle_training = tf.compat.v1.train.AdamOptimizer

loss_type = 'dice'  # crossentropy/dice
summary_writing_frequency = 20
train_eval_frequency = 100
val_eval_frequency = 100
save_frequency = 500
continue_run = False
debug = True

# self-supervised rotation network
num_rotation_values = 4
loss_handle_self_supervised = tf.nn.softmax_cross_entropy_with_logits
eval_handle_self_supervised = tf.compat.v1.metrics.accuracy

# self-supervised autoencoder network
loss_handle_autoencode = tf.compat.v1.losses.mean_squared_error # needs: labels, predictions
eval_handle_autoencode = tf.compat.v1.losses.mean_squared_error # needs: labels, predictions

# normalization network
optimizer_handle_test_time = tf.compat.v1.train.AdamOptimizer

# biasing the segmentation and self-supervised losses
lambda_seg = 0.5
lambda_ss = 0.5
