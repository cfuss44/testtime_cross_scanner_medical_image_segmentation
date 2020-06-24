# Test-Time Training for Cross-Scanner Robustness in Deep Learning Based Medical Image Segmentation

This is an implementation for the experiments done in my semester project about "Test-Time Training for Cross-Scanner Robustness in Deep Learning Based Medical Image Segmentation".

# Requirements
The code has been tested with tensorflow 2.0.0 and python 3.7.6.

# Running the experiments
Set the paths for your code and data files in 'config/system.py'. Training and testing hyperparameters can be adpated in 'expermients/branched_network_config.py'.

For initial training, the files 'train_[...].py' can be used. Run 'train_branched_network_rot_joint.py' to train the joint segmentation and rotation network, 'train_segmentation.py' to train the separate segmentation network, 'train_rotation.py' to train the separate rotation network, and 'train_autoencoder.py' to train the autoencoder separately.

For test-time training, the files 'test_[...].py' can be used. In 'test_branched_network_rot_joint.py', the jointly trained segmentation and rotation network is loaded and the shared parameters are fine-tuned on the loaded images. In 'test_branched_network_rot_sep.py', the separately trained segmentation and rotation networks are loaded and the normalization network is initialized at random. In 'test_branched_network_ae.py', the separately trained segmentation network and autoencoder are loaded and the normalization network is initialized at random.
The type of the images that should be loaded as well as the paths to the pretrained models need to be adapted.

# Acknowledgements
This code is based on the code on https://github.com/neerakara/Adaptive_batch_norm_for_segmentation, the data readers are taken from https://github.com/neerakara/data_readers_for_segmentation_datasets.
