"""
Modified from mlnet code

https://github.com/marcellacornia/mlnet
"""
from __future__ import division
from keras.models import Model
from keras.layers.core import Dropout, Activation
from keras.layers import Input, merge
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.regularizers import l2
import keras.backend as K
import h5py
from eltwise_product import EltWiseProduct
from baseline_config import *


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
                         trainable=True)


def build_model(img_rows=480, img_cols=640,
                downsampling_factor_net=8, downsampling_factor_product=10,
                use_pretrained=None,
                intermediate=False):
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

    int_conv = Convolution2D(64, 3, 3,
                             init='glorot_normal',
                             activation='relu',
                             border_mode='same', trainable=True,
                             name='int_conv')(dropout)

    pre_final_conv = Convolution2D(1, 1, 1,
                                   init='glorot_normal',
                                   activation='relu',
                                   name='pre_final_conv')(int_conv)


    rows_elt = math.ceil(img_rows / downsampling_factor_net) // downsampling_factor_product
    cols_elt = math.ceil(img_cols / downsampling_factor_net) // downsampling_factor_product
    eltprod = EltWiseProduct(init='zero',
                             trainable=True,
                             W_regularizer=l2(1/(rows_elt*cols_elt)),
                             name='eltprod')(pre_final_conv)
    output_ml_net = Activation('relu')(eltprod)

    model = Model(input=[input_ml_net], output=[output_ml_net])

    # load pretrained weights
    if use_pretrained == 'mlnet':
        model.load_weights('models/mlnet_salicon_weights.pkl')
    elif use_pretrained == 'my':
        model.load_weights('models/generalist_finetune.h5')
    elif use_pretrained is None:
        return model
    else:
        raise StandardError('INVALID use_pretrained')

    if intermediate:
        # still has correct weights
        model = Model(input=[input_ml_net], output=[concatenated])

    return model


def loss(y_true, y_pred):
    max_y = K.repeat_elements(
        K.expand_dims(
            K.repeat_elements(
                K.expand_dims(K.max(K.max(y_pred, axis=2), axis=2)),
                shape_r_gt, axis=-1)
        ), shape_c_gt, axis=-1)
    max_y = max_y + 1e-3 # added to insure than Nans don't happen

    return K.mean(K.square((y_pred / max_y) - y_true) / (1 - y_true + 0.1))
