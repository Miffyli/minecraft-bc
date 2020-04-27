#
#  replay_memory.py
#  Here we go once again...
#
import random
import numpy as np


class ArbitraryReplayMemory:
    """
    A replay memory for storing any type of elements
    """

    def __init__(self, max_size):
        self.replay_size = max_size
        self.replay_memory = [None for i in range(max_size)]

        # Index for the next sample
        self.ctr = 0
        # How many items are in the replay memory
        self.max_size = 0

    def __len__(self):
        """
        Return current number of samples in the
        replay memory
        """
        return self.max_size

    def add(self, item):
        """
        Add an item to the replay memory
        """
        self.replay_memory[self.ctr] = item

        self.ctr += 1
        self.max_size = max(
            self.max_size,
            self.ctr
        )
        if self.ctr == self.replay_size:
            self.ctr = 0

    def get_batch(self, batch_size):
        """
        Return batch_size random elements from the replay_memory,
        """
        random_idxs = random.sample(range(self.max_size), batch_size)
        return_values = [self.replay_memory[idx] for idx in random_idxs]
        return return_values
