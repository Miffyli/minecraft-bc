#!/usr/bin/env python3
#
# models.py
#
# Network models for Keras
#

import numpy as np
import keras
import keras.backend as K
from keras import layers
from keras import initializers
from keras import regularizers


def residual_block(y, nb_channels, _strides=(1, 1), _project_shortcut=False):
    """
    Implementation of residual block copied from here:
        https://gist.github.com/mjdietzx/5319e42637ed7ef095d430cb5c5e8c64

    NOTE: Not the same as IMPALA residual block
    """
    shortcut = y

    # down-sampling is performed with a stride of 2
    y = layers.Conv2D(nb_channels, kernel_size=(3, 3), strides=_strides, padding='same')(y)
    y = layers.BatchNormalization()(y)
    y = layers.ReLU()(y)

    y = layers.Conv2D(nb_channels, kernel_size=(3, 3), strides=(1, 1), padding='same')(y)
    y = layers.BatchNormalization()(y)

    # identity shortcuts used directly when the input and output are of the same dimensions
    if _project_shortcut or _strides != (1, 1):
        # when the dimensions increase projection shortcut is used to match dimensions (done by 1×1 convolutions)
        # when the shortcuts go across feature maps of two sizes, they are performed with a stride of 2
        shortcut = layers.Conv2D(nb_channels, kernel_size=(1, 1), strides=_strides, padding='same')(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    y = layers.add([shortcut, y])
    y = layers.ReLU()(y)

    return y


def resnet_head(input_shape):
    """
    Return keras Input and partial model with resnet head

    This is close-ish to IMPALA network but with batchnorm
    """
    input_layer = layers.Input(shape=input_shape)

    model = layers.Conv2D(16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=None)(input_layer)
    model = layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2))(model)
    model = layers.ReLU()(model)
    model = residual_block(model, 16)
    model = residual_block(model, 16)

    model = layers.Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=None)(model)
    model = layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2))(model)
    model = layers.ReLU()(model)
    model = residual_block(model, 32)
    model = residual_block(model, 32)

    model = layers.Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=None)(model)
    model = layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2))(model)
    model = layers.ReLU()(model)
    model = residual_block(model, 32)
    model = residual_block(model, 32)

    model = layers.Flatten()(model)

    return input_layer, model


def IMPALA_resnet_head(input_shape, l2_weight=0.0):
    """
    Resnet similar to one used by IMPALA (e.g. no batchnorm, but we have FixUp init)

    Reference:
        https://github.com/deepmind/scalable_agent/blob/master/experiment.py#L143

    Also has some bits of the FixUp initialization ("Rules 1 and 2"):
        https://arxiv.org/pdf/1901.09321.pdf
    With tips from:
        https://github.com/Zelgunn/CustomKerasLayers
        (handy to see VarianceScaling almost does the trick for us)
    """
    # Total number of layers in this resnet. Used to approximiately get the
    # FixUp initialization right
    TOTAL_RESIDUAL_BLOCKS = 6

    model = layers.Input(shape=input_shape)
    input_layer = model

    for i, (num_channels, num_blocks) in enumerate([[16, 2], [32, 2], [32, 2]]):
        model = layers.Conv2D(
            num_channels, kernel_size=(3, 3), strides=(1, 1), padding="same", activation=None,
            kernel_regularizer=regularizers.l2(l2_weight)
        )(model)
        model = layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2))(model)

        for j in range(num_blocks):
            block_input = model
            model = layers.ReLU()(model)
            model = layers.Conv2D(
                num_channels, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=None,
                kernel_regularizer=regularizers.l2(l2_weight),
                kernel_initializer=keras.initializers.VarianceScaling(
                    # Scaling is L^(-1/(2m - 2)) . In our case m = 2 (two layers in branch),
                    # so our rescaling is L^(-1/2) = 1 / sqrt(L)
                    scale=1 / np.sqrt(TOTAL_RESIDUAL_BLOCKS)
                )
            )(model)
            model = layers.ReLU()(model)
            model = layers.Conv2D(
                num_channels, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=None,
                kernel_initializer="zero", bias_initializer="zero",
                kernel_regularizer=regularizers.l2(l2_weight)
            )(model)
            model = layers.add([model, block_input])

    model = layers.ReLU()(model)
    model = layers.Flatten()(model)
    model = layers.Dense(256, activation="relu")(model)

    return input_layer, model


def nature_dqn_head(input_shape, l2_weight=0.0):
    """Return CNN head akin to Nature DQN CNN"""
    input_layer = layers.Input(shape=input_shape)

    model = layers.Conv2D(32, kernel_size=8, strides=4, activation="relu", kernel_regularizer=regularizers.l2(l2_weight))(input_layer)
    model = layers.Conv2D(64, kernel_size=4, strides=2, activation="relu", kernel_regularizer=regularizers.l2(l2_weight))(model)
    model = layers.Conv2D(64, kernel_size=3, strides=1, activation="relu", kernel_regularizer=regularizers.l2(l2_weight))(model)
    model = layers.Flatten()(model)
    model = layers.Dense(512, activation="relu")(model)

    return input_layer, model


def network_body(image_shape, num_channels, num_direct, head_func=nature_dqn_head, body_size="small", l2_weight=0.0):
    """
    Main network body, combining cnn features and direct features into one vector

    head_func specifies function which creates the cnn head.
    body_size ("small" or "large") specifies how large dense layers should be.
         "small" is closer to network used in IMPALA/R2D3
    """
    image_input, model = head_func(image_shape + (num_channels, ), l2_weight=l2_weight)

    # Part for direct features
    direct_input = layers.Input(shape=(num_direct,))

    if body_size == "small":
        model = layers.Concatenate()([model, direct_input])
        model = layers.Dense(256, activation="relu", kernel_regularizer=regularizers.l2(l2_weight))(model)
    elif body_size == "large":
        # Process direct input and combine with cnn output
        direct_model = layers.Dense(256, activation="relu", kernel_regularizer=regularizers.l2(l2_weight))(direct_input)
        model = layers.Concatenate()([model, direct_model])
        model = layers.Dense(512, activation="relu", kernel_regularizer=regularizers.l2(l2_weight))(model)
    else:
        raise ValueError("Unknown network body_size {}".format(body_size))

    # Small vs large network.
    # Some MineRL related tweaking:
    # Since some of our actions could be result of direct_inputs
    # (e.g. crafting planks just needs logs in inventory), feed
    # direct_input directly to actions
    model = layers.Concatenate()([model, direct_input])

    return image_input, direct_input, model

def policy_net(image_shape, num_channels, num_direct, action_nvec, l2_weight=0.0, **kwargs):
    """
    Small residual net with multidiscrete action space output

    Returns non-compiled model with appropiately softmaxed outputs of size sum(action_nvec).
    Also returns list of individual outputs
    """
    image_input, direct_input, model = network_body(
        image_shape, num_channels, num_direct, l2_weight=l2_weight, **kwargs
    )

    # Create one head per output
    outputs = []
    for i, num_actions in enumerate(action_nvec):
        outputs.append(
            layers.Dense(
                num_actions, activation="softmax", name="action_{}".format(i),
                kernel_regularizer=regularizers.l2(l2_weight)
            )(model)
        )

    model = None
    if len(outputs) == 1:
        model = outputs[0]
    else:
        model = layers.Concatenate()(outputs)

    return model, outputs, (image_input, direct_input)
