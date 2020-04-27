#!/usr/bin/env python3
#
# train_keras_v.py
#
# Train Keras value estimation on MineRL data
#

from argparse import ArgumentParser
import numpy as np
import keras
import minerl
import time
import random

from itertools import cycle
from collections import deque
from multiprocessing import Process, Queue
import queue

from keras_utils.models import policy_net, nature_dqn_head, resnet_head, IMPALA_resnet_head
from keras_utils.losses import create_multicategorical_loss
from wrappers.observation_wrappers import ObtainDiamondObservation
from wrappers.action_wrappers import ObtainDiamondActions
from utils.minerl_utils import unzip_states_or_actions
from utils.replay_memory import ArbitraryReplayMemory


# Limit memory usage
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

parser = ArgumentParser("Train Keras models to do imitation learning.")
parser.add_argument("data_dir", type=str, help="Path to MineRL dataset.")
parser.add_argument("model", type=str, default=None, help="Path where to store trained model.")
parser.add_argument("datasets", type=str, nargs="+", help="List of datasets to use for the training. First one should include biggest action space")
parser.add_argument("--workers", type=int, default=16, help="Number of dataset workers")
parser.add_argument("--max-seqlen", type=int, default=32, help="Max length per loader")
parser.add_argument("--seqs-per-update", type=int, default=1, help="How many sequences are loaded per one update (mini-batch) train")
parser.add_argument("--replay-size", type=int, default=500000, help="Maximum number of individual training samples to store in replay memory.")
parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to train.")
parser.add_argument("--save-every-updates", type=int, default=250000, help="How many iterations between saving a snapshot of the model")
parser.add_argument("--batch-size", type=int, default=32, help="Ye' olde batch size.")
parser.add_argument("--lr", type=float, default=0.00005, help="Adam learning rate.")
parser.add_argument("--lr-decay", type=float, default=0.0, help="Decay for learning rate.")
parser.add_argument("--target-value", type=float, default=0.995, help="Target value where cross-entropy aims.")
parser.add_argument("--l2", type=float, default=0.0, help="L2 regularizer weight.")
parser.add_argument("--gamma", type=float, default=1.0, help="Additional gamma correction (on top of the regular correction).")
parser.add_argument("--numeric-df", action="store_true", help="Use scalars for representing inventory rather than one-hot encoding.")
parser.add_argument("--cnn-head", type=str, default="nature", choices=["nature", "resnet", "impala"], help="CNN head to be used in networks")
parser.add_argument("--nn-size", type=str, default="small", choices=["small", "large"], help="Main NN size to be used")
parser.add_argument("--no-augmentation", action="store_true", help="Do not use augmentation for images.")
parser.add_argument("--no-flipping", action="store_true", help="Do not do horizontal flipping for augmentation.")

CNN_HEADS = {
    "nature": nature_dqn_head,
    "resnet": resnet_head,
    "impala": IMPALA_resnet_head
}


def trajectories_to_replay_memory(trajectories, replay_memory, args):
    """
    Turn bunch of trajectories into individual training samples
    and store them in replay_memory.
    """

    for trajectory in trajectories:
        states, actions, rewards = trajectory

        # We tried some trickery here by tracking
        # rewards were obtained, hence for-loop
        # starting from the end.
        for i in range(len(states) - 1, -1, -1):
            # Skip no-ops. Since we are doing
            # one-step imitation learning, no
            # ops just confuse our training.
            if sum(actions[i]) == 0:
                continue

            replay_memory.add((states[i], actions[i]))


def get_training_batch(replay_memory, batch_size):
    """
    Returns a single training batch suitable for
    training Keras models (i.e, tuple (inputs, outputs))
    """
    # Raw_batch is List of (inputs, outputs)
    raw_batch = replay_memory.get_batch(batch_size)

    # We need to stack individual samples
    inputs = None
    outputs = None

    # Inputs can be a list/tuple of ndarrays
    if isinstance(raw_batch[0][0], tuple) or isinstance(raw_batch[0][0], list):
        inputs = []
        # For each different input type
        for i in range(len(raw_batch[0][0])):
            # For each element in batch
            inputs.append(np.array([raw_batch[b][0][i] for b in range(batch_size)]))
    else:
        inputs = np.array([raw_batch[b][0] for b in range(batch_size)])

    # Turn outputs to ints (for cross-entropy/etc losses)
    outputs = np.array([raw_batch[b][1] for b in range(batch_size)], dtype=np.int32)

    return inputs, outputs


def data_preprocessor_worker(
    obs_processor,
    act_processor,
    in_queue,
    out_queue,
    do_flipping
):
    """
    Data preprocessor worker: Takes in MineRL data samples
    from in_queue and outputs ready stuff to out_queue

    If do_flipping is True, apply horizontal flipping on the
    samples randomly.
    """
    in_sample = None
    while True:
        try:
            in_sample = in_queue.get(timeout=60)
        except Exception:
            # Timeout or queue was closed -> quit
            break

        # Do preprocessing
        states = in_sample[0]
        actions = in_sample[1]
        rewards = in_sample[2]

        states = unzip_states_or_actions(states)
        actions = unzip_states_or_actions(actions)

        states = list(map(lambda state: obs_processor.dict_to_tuple(state), states))
        actions = list(map(lambda action: act_processor.dict_to_multidiscrete(action), actions))

        # Flipping augmentation
        if do_flipping:
            for state, action in zip(states, actions):
                if random.random() < 0.5:
                    # Flip both observation (image)
                    # and action left-right
                    obs_processor.flip_left_right(state)
                    act_processor.flip_left_right(action)

        try:
            out_queue.put([states, actions, rewards], timeout=30)
        except Exception:
            # Something went wrong. Quit the loop so
            # process dies
            break


def main(args):
    workers_per_loader = args.workers // len(args.datasets)
    data_loaders = [
        minerl.data.make(dataset, data_dir=args.data_dir, num_workers=workers_per_loader)
        for dataset in args.datasets
    ]

    # Use first dataloader as info for observation space and action space
    obs_processor = ObtainDiamondObservation(
        data_loaders[0].observation_space,
        augmentation=not args.no_augmentation,
        gamma_correction=args.gamma,
        numeric_df=args.numeric_df
    )

    act_processor = None
    action_nvec = None

    act_processor = ObtainDiamondActions(data_loaders[0].action_space)
    action_nvec = act_processor.action_space.nvec

    image_shape = obs_processor.observation_space[0].shape
    direct_shape = obs_processor.observation_space[1].shape

    cnn_head_func = CNN_HEADS[args.cnn_head]

    model, individual_outputs, (image_input, direct_input) = policy_net(
        image_shape[:2],
        image_shape[2],
        action_nvec=action_nvec,
        num_direct=direct_shape[0],
        head_func=cnn_head_func,
        body_size=args.nn_size,
        l2_weight=args.l2
    )

    # No weighting of the actions
    weights = [[1 for j in range(n_actions)] for n_actions in action_nvec]

    model = keras.models.Model(inputs=(image_input, direct_input), output=model)
    model.compile(
        loss=create_multicategorical_loss(action_nvec, weights, target_value=args.target_value),
        optimizer=keras.optimizers.Adam(lr=args.lr, decay=args.lr_decay),
    )

    # Create iterators and alternate between them
    data_iterators = cycle([
        data.sarsd_iter(num_epochs=args.epochs, max_sequence_len=args.max_seqlen)
        for data in data_loaders
    ])

    # Replay memory where we store recently loaded
    # samples. Used to balance out the bias from
    # loading samples only limited number of trajectories
    # at a time
    replay_memory = ArbitraryReplayMemory(args.replay_size)

    # Create data processors
    data_workers = []
    # NAUGHTY HARDCODING:
    # Fixed Queue sizes and number of data preprocessors
    raw_data_queue = Queue(50)
    processed_data_queue = Queue(50)
    for i in range(4):
        worker = Process(
            target=data_preprocessor_worker,
            args=(
                obs_processor,
                act_processor,
                raw_data_queue,
                processed_data_queue,
                not args.no_flipping
            )
        )
        worker.start()
        data_workers.append(worker)

    num_updates = 1
    start_time = time.time()
    average_losses = deque(maxlen=1000)
    last_save_updates = 0

    states = None
    acts = None
    rewards = None
    state_primes = None
    dones = None

    for data_iterator in data_iterators:
        try:
            states, acts, rewards, state_primes, dones = next(data_iterator)
        except StopIteration:
            # Reached end of the iterator -> Enough epochs
            # NOTE: Since some datasets are smaller, this will happen
            #       on the smallest dataset, and thus does not
            #       reflect when whole data has been visited
            #       N times
            break

        # First check if there are enough trajectories to be added to replay
        # memory before doing an timeout
        if processed_data_queue.qsize() >= args.seqs_per_update:
            trajectories = [processed_data_queue.get(timeout=30) for i in range(args.seqs_per_update)]
            # Add trajectory to replay memory
            trajectories_to_replay_memory(trajectories, replay_memory, args)

            # Check if we have enough samples in the replay memory to do an update
            if len(replay_memory) > args.batch_size:
                # Get batch
                train_inputs, train_outputs = get_training_batch(replay_memory, args.batch_size)
                average_losses.append(model.train_on_batch(
                    train_inputs,
                    train_outputs
                ))
                num_updates += 1

            # Do not print status too often
            if (num_updates % 1000) == 0:
                time_passed = int(time.time() - start_time)
                print("Time: {:<8} Updates: {:<8} AvrgLoss: {:.4f}".format(
                    time_passed, num_updates, np.mean(average_losses)
                ))

            # Check if we should save a snapshot
            if (num_updates - last_save_updates) >= args.save_every_updates:
                model.save(args.model + "_steps_{}".format(num_updates))
                last_save_updates = num_updates

        # Put new data for workers to handle
        try:
            raw_data_queue.put([states, acts, rewards], timeout=30)
        except queue.Full:
            # Raw_data_queue was full, so just skip this sample.
            # If other exceptions arise, crash to it (e.g. queues shouldn't die)
            continue

    model.save(args.model)

    # Close queues
    raw_data_queue.close()
    processed_data_queue.close()
    # Join processes
    for worker in data_workers:
        worker.terminate()
        worker.join()


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
