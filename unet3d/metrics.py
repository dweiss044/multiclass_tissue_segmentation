from functools import partial

from itertools import product

from keras import backend as K

import tensorflow as tf


def dice_coefficient(y_true, y_pred, smooth=1.):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coefficient_loss(y_true, y_pred):
    return -dice_coefficient(y_true, y_pred)


def weighted_dice_coefficient(y_true, y_pred, axis=(-3, -2, -1), smooth=0.00001):
    """
    Weighted dice coefficient. Default axis assumes a "channels first" data structure
    :param smooth:
    :param y_true:
    :param y_pred:
    :param axis:
    :return:
    """
    return K.mean(2. * (K.sum(y_true * y_pred,
                              axis=axis) + smooth/2)/(K.sum(y_true,
                                                            axis=axis) + K.sum(y_pred,
                                                                               axis=axis) + smooth))


def weighted_dice_coefficient_loss(y_true, y_pred):
    return -weighted_dice_coefficient(y_true, y_pred)


def label_wise_dice_coefficient(y_true, y_pred, label_index):
    return dice_coefficient(y_true[:, label_index], y_pred[:, label_index])


def get_label_dice_coefficient_function(label_index):
    f = partial(label_wise_dice_coefficient, label_index=label_index)
    f.__setattr__('__name__', 'label_{0}_dice_coef'.format(label_index))
    return f

def weighted_bce(alpha=0.9):
    def _loss(y_true, y_pred):
        # weight positives stronger than negatives --> 9:1, alpha = 0.9
        weights = (y_true * alpha/(1.-alpha)) + 1.
        bce = K.binary_crossentropy(y_true, y_pred)
        weighted_bce = K.mean(bce * weights)
        return weighted_bce
    return _loss

def categorical_crossentropy_loss(y_true, y_pred):
    return K.categorical_crossentropy(y_true, y_pred, axis=1)

def w_categorical_crossentropy(target, output, weights, axis=1):
    """Categorical crossentropy between an output tensor and a target tensor.

    # Arguments
        target: A tensor of the same shape as `output`.
        output: A tensor resulting from a softmax
            
    # Returns
        Loss tensor

    """
    # scale preds so that the class probas of each sample sum to 1
    output /= tf.reduce_sum(output, axis, True)
    # manual computation of crossentropy
    _epsilon = tf.convert_to_tensor(1e-7, output.dtype.base_dtype)
    output = tf.clip_by_value(output, _epsilon, 1. - _epsilon)
    target_channels_last = tf.transpose(target, [0,2,3,4,1])
    w = target_channels_last*tf.constant(weights, dtype = target.dtype.base_dtype)
    w = tf.reduce_sum(w, axis = -1)
    w = tf.expand_dims(w, 1)
    return - tf.reduce_sum(target * tf.log(output) * w, axis)

def w_categorical_crossentropy_loss(weights):
    def _loss(y_true, y_pred):
        return w_categorical_crossentropy(y_true, y_pred, weights)
    return _loss



def weighted_cce(weights):
    # weights must broadcast to [B,C,H,W,D]
    weights = K.reshape(K.variable(weights),(1,len(weights),1,1,1))
    def _loss(y_true, y_pred):
        return K.mean(K.categorical_crossentropy(y_true, y_pred, axis=1) * weights)
    return _loss

def w_categorical_crossentropy_old(weights):
    def _loss(y_true,y_pred):
        nb_cl = len(weights)
        final_mask = K.zeros_like(y_pred[:,0])
        y_pred_max = K.max(y_pred, axis=1, keepdims=True)
        y_pred_max_mat = K.cast(K.equal(y_pred, y_pred_max),'float32')
        for c_p, c_t in product(range(nb_cl),range(nb_cl)):
            final_mask += (weights[c_t, c_p] * y_pred_max_mat[:, c_p, :, :, :] * y_true[:, c_t, :, :, :])
        return K.categorical_crossentropy(y_true, y_pred, axis=1) * final_mask
    return _loss

'''
def categorical_crossentropy(target, output, from_logits=False, axis=-1):
    """Categorical crossentropy between an output tensor and a target tensor.

    # Arguments
        target: A tensor of the same shape as `output`.
        output: A tensor resulting from a softmax
            (unless `from_logits` is True, in which
            case `output` is expected to be the logits).
        from_logits: Boolean, whether `output` is the
            result of a softmax, or is a tensor of logits.
        axis: Int specifying the channels axis. `axis=-1`
            corresponds to data format `channels_last`,
            and `axis=1` corresponds to data format
            `channels_first`.

    # Returns
        Output tensor.

    # Raises
        ValueError: if `axis` is neither -1 nor one of
            the axes of `output`.
    """
    output_dimensions = list(range(len(output.get_shape())))
    if axis != -1 and axis not in output_dimensions:
        raise ValueError(
            '{}{}{}'.format(
                'Unexpected channels axis {}. '.format(axis),
                'Expected to be -1 or one of the axes of `output`, ',
                'which has {} dimensions.'.format(len(output.get_shape()))))
    # Note: tf.nn.softmax_cross_entropy_with_logits
    # expects logits, Keras expects probabilities.
    if not from_logits:
        # scale preds so that the class probas of each sample sum to 1
        output /= tf.reduce_sum(output, axis, True)
        # manual computation of crossentropy
        _epsilon = _to_tensor(epsilon(), output.dtype.base_dtype)
        output = tf.clip_by_value(output, _epsilon, 1. - _epsilon)
        return - tf.reduce_sum(target * tf.log(output), axis)
    else:
        return tf.nn.softmax_cross_entropy_with_logits(labels=target,
                                                       logits=output)
'''
dice_coef = dice_coefficient
dice_coef_loss = dice_coefficient_loss
