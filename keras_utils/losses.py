#!/usr/bin/env python3
#
# losses.py
#
# Losses for training Keras models
#

import numpy as np
import keras.backend as K
import tensorflow as tf


def create_multicategorical_loss(action_nvec, weights, target_value=0.995):
    """ Returns loss appropiate for multicategorical training.

    y_target is a list of integers (not one-hot), and y_pred is
    list of one-hot/raveled actions.

    weights is a list of lists according to action_nvec, each being
    a weight for that one value.

    target_value is the target for selected actions. Normally you try
    to predict {0, 1}, but problem there is that logits need to be
    {-inf, inf} to reach that. Instead aim for something more relaxed,
    e.g. {0.005, 0.995}. This is similar to Keras's/Tensorflow's label
    smoothing.
    """
    def multicategorical_loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.int32)
        losses = []
        current_index = 0
        for i, action_size in enumerate(action_nvec):
            preds = y_pred[:, current_index: current_index + action_size]
            trues = tf.one_hot(y_true[:, i], action_size)

            # Do not aim for strict {0,1} as this
            # is not reachable by softmax. Instead aim for
            # something reasonable, e.g. 0.995.
            # This is called "label smoothing" in Keras and Tensorflow
            trues = trues * target_value + ((1 - target_value) / action_size)

            # Clipping
            epsilon = K.epsilon()
            preds = tf.clip_by_value(preds, epsilon, 1.0)
            trues = tf.clip_by_value(trues, epsilon, 1.0)
            # Include weighting to both positive and negative samples,
            # such that variables with high weighting should happen more
            # often
            option_weights = np.array(weights[i])

            # KL loss
            loss = trues * tf.log(trues / preds)
            # Apply weighting and sum over support
            loss = tf.reduce_sum(loss * option_weights, axis=-1)

            losses.append(loss)
            current_index += action_size
        # Sum over different actions and then mean over batch elements
        loss = K.mean(K.sum(losses, axis=0))
        return loss

    return multicategorical_loss
