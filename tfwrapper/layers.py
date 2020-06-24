import tensorflow as tf
import numpy as np
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

## ======================================================================
## ======================================================================
def crop_and_concat_layer(inputs, axis=-1):

    '''
    Layer for cropping and stacking feature maps of different size along a different axis. 
    Currently, the first feature map in the inputs list defines the output size. 
    The feature maps can have different numbers of channels. 
    :param inputs: A list of input tensors of the same dimensionality but can have different sizes
    :param axis: Axis along which to concatentate the inputs
    :return: The concatentated feature map tensor
    '''

    output_size = inputs[0].get_shape().as_list()
    concat_inputs = [inputs[0]]

    for ii in range(1,len(inputs)):

        larger_size = inputs[ii].get_shape().as_list()
        start_crop = np.subtract(larger_size, output_size) // 2

        if len(output_size) == 5:  # 3D images
            cropped_tensor = tf.slice(inputs[ii],
                                     (0, start_crop[1], start_crop[2], start_crop[3], 0),
                                     (-1, output_size[1], output_size[2], output_size[3], -1))
        elif len(output_size) == 4:  # 2D images
            cropped_tensor = tf.slice(inputs[ii],
                                     (0, start_crop[1], start_crop[2], 0),
                                     (-1, output_size[1], output_size[2], -1))
        else:
            raise ValueError('Unexpected number of dimensions on tensor: %d' % len(output_size))

        concat_inputs.append(cropped_tensor)

    return tf.concat(concat_inputs, axis=axis)

## ======================================================================
## ======================================================================    
def pad_to_size(bottom, output_size):

    ''' 
    A layer used to pad the tensor bottom to output_size by padding zeros around it
    TODO: implement for 3D data
    '''

    input_size = bottom.get_shape().as_list()
    size_diff = np.subtract(output_size, input_size)

    pad_size = size_diff // 2
    odd_bit = np.mod(size_diff, 2)

    if len(input_size) == 4:

        padded =  tf.pad(bottom, paddings=[[0,0],
                                        [pad_size[1], pad_size[1] + odd_bit[1]],
                                        [pad_size[2], pad_size[2] + odd_bit[2]],
                                        [0,0]])

        return padded

    elif len(input_size) == 5:
        raise NotImplementedError('This layer has not yet been extended to 3D')
    else:
        raise ValueError('Unexpected input size: %d' % input_size)
       
## ======================================================================
# reshape
## ======================================================================
def reshape_like(target, size, name):
    
    '''
    target: tensor to be reshaped
    size: shape to which the target tensor should be reshaped to
    '''
    
    target_reshaped = tf.image.resize(target, size, method=tf.image.ResizeMethod.BILINEAR, name=name)    
    return target_reshaped

## ======================================================================
## ======================================================================
def _add_summaries(op, weights, biases):

    # Tensorboard variables
    tf.summary.histogram(weights.name[:-2], weights)
    if biases: tf.summary.histogram(biases.name[:-2], biases)
    tf.summary.histogram(op.op.name + '/activations', op)
