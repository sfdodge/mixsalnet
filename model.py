"""
Mixnet model definition
"""
from __future__ import division
from keras.models import Model
from keras.layers.core import Dropout, Activation, Dense, Flatten
from keras.layers import Input, merge
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.regularizers import l2
import keras.backend as K
import h5py
from eltwise_product import EltWiseProduct
import math
from baseline_config import *

MLNET_TRAINABLE = False
TEMPERATURE = 10.
N_CAT = 20


def get_weights_vgg16(id):
    if not hasattr(get_weights_vgg16, "f"):
        get_weights_vgg16.f = h5py.File("models/vgg16_weights.h5")

    g = get_weights_vgg16.f['layer_{}'.format(id)]
    return [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]


def VGGConvLayer(n_filt, size=3, vgg_layer=None):
    """
    Wrapper to make the code prettier
    """
    if vgg_layer is not None:
        weights = get_weights_vgg16(vgg_layer)
    else:
        weights = None

    return Convolution2D(n_filt, size, size,
                         weights=weights,
                         activation='relu',
                         border_mode='same',
                         name='vgg_' + str(vgg_layer),
                         trainable=MLNET_TRAINABLE)


def build_model(img_rows=480, img_cols=640,
                downsampling_factor_net=8, downsampling_factor_product=10):
    input_ml_net = Input(shape=(3, img_rows, img_cols))
    conv1_1 = VGGConvLayer(64, vgg_layer=1)(input_ml_net)
    conv1_2 = VGGConvLayer(64, vgg_layer=3)(conv1_1)
    conv1_pool = MaxPooling2D((2, 2), strides=(2, 2), border_mode='same')(conv1_2)

    conv2_1 = VGGConvLayer(128, vgg_layer=6)(conv1_pool)
    conv2_2 = VGGConvLayer(128, vgg_layer=8)(conv2_1)
    conv2_pool = MaxPooling2D((2, 2), strides=(2, 2), border_mode='same')(conv2_2)

    conv3_1 = VGGConvLayer(256, vgg_layer=11)(conv2_pool)
    conv3_2 = VGGConvLayer(256, vgg_layer=13)(conv3_1)
    conv3_3 = VGGConvLayer(256, vgg_layer=15)(conv3_2)
    conv3_pool = MaxPooling2D((2, 2), strides=(2, 2), border_mode='same')(conv3_3)

    conv4_1 = VGGConvLayer(512, vgg_layer=18)(conv3_pool)
    conv4_2 = VGGConvLayer(512, vgg_layer=20)(conv4_1)
    conv4_3 = VGGConvLayer(512, vgg_layer=22)(conv4_2)
    conv4_pool = MaxPooling2D((2, 2), strides=(1, 1), border_mode='same')(conv4_3)

    conv5_1 = VGGConvLayer(512, vgg_layer=25)(conv4_pool)
    conv5_2 = VGGConvLayer(512, vgg_layer=27)(conv5_1)
    conv5_3 = VGGConvLayer(512, vgg_layer=29)(conv5_2)

    concatenated = merge([conv3_pool, conv4_pool, conv5_3],
                         mode='concat', concat_axis=1)
    dropout = Dropout(0.5)(concatenated)

    # class prediction
    init_mode = 'glorot_normal'
    down = MaxPooling2D((1, 1), strides=(4,4))(input_ml_net)
    class_conv1 = Convolution2D(32, 3, 3,
                                activation='relu', init=init_mode,
                                name='class_conv1')(down)
    class_conv1_pool = MaxPooling2D((2, 2), strides=(2, 2), border_mode='same')(class_conv1)
    class_conv2 = Convolution2D(64, 3, 3,
                                activation='relu', init=init_mode,
                                name='class_conv2')(class_conv1_pool)
    class_conv2_pool = MaxPooling2D((2, 2), strides=(2, 2), border_mode='same')(class_conv2)
    class_conv3 = Convolution2D(128, 3, 3, init=init_mode,
                                activation='relu', name='class_conv3')(class_conv2_pool)
    class_conv3_pool = MaxPooling2D((2, 2), strides=(2, 2), border_mode='same')(class_conv3)
    class_conv4 = Convolution2D(128, 3, 3, init=init_mode,
                                activation='relu', name='class_conv4')(class_conv3_pool)
    class_conv4_pool = MaxPooling2D((2, 2), strides=(2, 2), border_mode='same')(class_conv4)
    flatten = Flatten(name='flatten')(class_conv4_pool)
    cat_full = Dense(128, name='cat_full', init=init_mode)(flatten)
    cat_drop = Dropout(0.5)(cat_full)
    cat_relu = Activation('relu')(cat_drop)
    cat_out = Dense(N_CAT, name='cat_out', init=init_mode)(cat_relu)
    cat_soft = Activation('softmax', name='cat_soft')(cat_out)
    cat_soft_temp = Activation(soft_temp, name='cat_soft_temp')(cat_out)

    # for each category have a new layer
    pre_last = []
    last = []
    cb = []
    for i in range(N_CAT):
        pre_last.append(Convolution2D(64, 3, 3,
                                      init='glorot_normal',
                                      activation='relu',
                                      border_mode='same')(dropout))
        last.append(Convolution2D(1, 1, 1,
                                  init='glorot_normal',
                                  activation='relu')(pre_last[-1]))
        cb.append(center_bias(last[-1], img_rows, img_cols,
                              downsampling_factor_net, downsampling_factor_product))

    # then take softmax output and
    # do weighted combination of several 1x1 layers
    output_ml_net = cat_merge(cat_soft_temp, cb, name='output_ml_net')

    model = Model(input=[input_ml_net, ], output=[output_ml_net, cat_soft])

    return model


def cat_merge(cat, experts, name=None):
    merge_layer = merge(experts, mode='concat', concat_axis=1)

    # layer to multiply
    def f(x):
        outs = x[0]
        multiplier = K.expand_dims(x[1], dim=-1)
        multiplier = K.expand_dims(multiplier, dim=-1)
        multiplier = K.repeat_elements(multiplier, outs._keras_shape[2],
                                    axis=2)
        multiplier = K.repeat_elements(multiplier, outs._keras_shape[3],
                                    axis=3)
        return K.sum(multiplier * outs, axis=1, keepdims=True)

    mult_layer = merge([merge_layer, cat],
                       mode=f,
                       output_shape=(1, merge_layer._keras_shape[2],
                                     merge_layer._keras_shape[3]),
                       name=name)

    return mult_layer


def center_bias(x, img_rows, img_cols, downsampling_factor_net, downsampling_factor_product, out_name=None, elt_name=None):
    rows_elt = math.ceil(img_rows / downsampling_factor_net) // downsampling_factor_product
    cols_elt = math.ceil(img_cols / downsampling_factor_net) // downsampling_factor_product
    eltprod = EltWiseProduct(init='zero',
                             trainable=True,
                             W_regularizer=l2(1/(rows_elt*cols_elt*N_CAT)),
                             name=elt_name)(x)
    output_ml_net = Activation('relu', name=out_name)(eltprod)

    return output_ml_net


def soft_temp(x):
    # higher temperatures will make the softmax more flat
    e = K.exp((x - K.max(x, axis=-1, keepdims=True)) / TEMPERATURE)
    s = K.sum(e, axis=-1, keepdims=True)
    return e / s


def sal_loss(y_true, y_pred):
    max_y = K.repeat_elements(
        K.expand_dims(
            K.repeat_elements(
                K.expand_dims(K.max(K.max(y_pred, axis=2), axis=2)),
                shape_r_gt, axis=-1)
        ), shape_c_gt, axis=-1)
    max_y = max_y + 1e-3 # added to insure than Nans don't happen

    return K.mean(K.square((y_pred / max_y) - y_true) / (1 - y_true + 0.1))
