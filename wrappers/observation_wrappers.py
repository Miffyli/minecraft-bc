from collections import OrderedDict
import numpy as np
import random
from copy import deepcopy

import gym
from gym import spaces
from gym import Wrapper

# Default value for how long one-hot vectors we have for inventory counts
MAX_INV_ONEHOT = 8


def one_hot_encode(value, num_possibilities):
    """Returns one-hot list for value with max number of possibilities"""
    one_hot = [0 for i in range(num_possibilities)]
    one_hot[value] = 1
    return one_hot


def observation_data_augmentation(obs, do_flipping=False):
    """Modify observation in-place with data-augumentation"""
    pov_obs = obs[0]
    # Small amount of gaussian noise to the image
    pov_obs += np.random.normal(scale=0.005, size=pov_obs.shape)
    # Crude adjustment of contrast
    pov_obs *= random.uniform(0.98, 1.02)
    # Crude adjustment of brightness
    pov_obs += np.random.uniform(-0.02, 0.02, size=(1, 1, 3))

    if do_flipping:
        # With random chance, mirror the image (fliplr)
        if random.random() < 0.5:
            pov_obs = np.fliplr(pov_obs)

    # Clip
    np.clip(pov_obs, 0.0, 1.0, out=obs[0])


class ObtainDiamondObservation:
    """Turns observation space of ObtainDiamond into tuple of (image, direct_features):

    Direct features:
        - Inventory counts are turned into one-hot coding of 0-N to tell how much stuff is carried
        - Mainhand item is turned into one-hot
        - Mainhand item's damage is ratio between current damage and max damage
    
    If numeric_df is True, use single scalar for representing counts in the 
    inventory rather than one-hot encodings. 

    This is the core, Wrapless class
    """
    def __init__(self, observation_space, max_inventory_count=MAX_INV_ONEHOT,
                 augmentation=False, augmentation_flip=False, just_pov=False,
                 gamma_correction=1.0, numeric_df=False):
        self.augmentation = augmentation
        self.augmentation_flip = augmentation_flip
        self.max_inventory_count = max_inventory_count
        self.just_pov = just_pov
        self.inventory_eyes = np.eye(max_inventory_count + 1)
        self.inverse_gamma = 1 / gamma_correction
        self.numeric_df = numeric_df

        old_space = observation_space

        # Calculate how much stuff we will have in direct features
        # 1 from mainhand item damage
        self.direct_features_len = 1

        # Number of possible items in hand
        self.num_hand_items = old_space["equipped_items"]["mainhand"]["type"].n
        self.direct_features_len += self.num_hand_items 
        # Inventory sizes
        self.inventory_keys = []
        for key, space in old_space["inventory"].spaces.items():
            self.inventory_keys.append(key)
            # Include zero
            if not numeric_df:
                self.direct_features_len += max_inventory_count + 1
            else:
                self.direct_features_len += 1

        if not self.just_pov:
            self.observation_space = spaces.Tuple(spaces=(
                spaces.Box(low=0, high=1, shape=old_space["pov"].shape, dtype=np.float32),
                spaces.Box(low=0, high=1, shape=(self.direct_features_len,), dtype=np.float32)
            ))
        else:
            self.observation_space = spaces.Box(low=0, high=1, shape=old_space["pov"].shape, dtype=np.float32)


    def flip_left_right(self, obs):
        """Flip observation left-right for data augmentation"""
        # obs is a tuple so we can not replace it directly,
        # so we can change the internals instead
        obs[0][:] = np.fliplr(obs[0])


    def dict_to_tuple(self, dict_obs):
        """Convert dict observation into tuple one"""
        # Normalize the image for first observation
        pov_obs = dict_obs["pov"].astype(np.float32) / 255.0
        # Apply gamma correction
        if self.inverse_gamma != 1.0:
            pov_obs = pov_obs ** self.inverse_gamma
        # Construct direct features
        # TODO this is the lazy way...
        direct_features = []

        # Main-hand damage
        damage = dict_obs["equipped_items"]["mainhand"]["damage"]
        max_damage = max(1, dict_obs["equipped_items"]["mainhand"]["maxDamage"])
        direct_features.append([damage / max_damage, ])
        # Main-hand item type
        mainhand_item = dict_obs["equipped_items"]["mainhand"]["type"]

        # Workaround for a bug:
        # If player destroys dirt and pickups the dirt block,
        # mainhand_item will be "dirt" (not an int).
        # -> Replace all strings with integer 0
        if type(mainhand_item) == str:
            mainhand_item = 0

        direct_features.append(one_hot_encode(mainhand_item, self.num_hand_items))

        # Inventory counts
        for key, count in dict_obs["inventory"].items():
            count = min(count, self.max_inventory_count)
            if not self.numeric_df:
                # One-hot encodings
                direct_features.append(self.inventory_eyes[count])
            else:
                # Numeric encodings [0,1]
                direct_features.append([count / self.max_inventory_count])

        direct_features = np.concatenate(direct_features).astype(np.float32)

        obs = (pov_obs, direct_features)

        # If data-augmentation is enabled, add some noise or
        # modifications to the image
        if self.augmentation:
            observation_data_augmentation(obs, self.augmentation_flip)

        return obs


class FrameSkipWrapper(Wrapper):
    """Wrapper to implement frameskip for ObtainDiamond environment"""

    def __init__(self, env, frame_skip=4):
        super().__init__(env)
        self.env = env
        self.frame_skip = frame_skip

        # Check that we have dictionary action space
        # so we can do our trickery with crafting/etc options
        if not isinstance(self.env.action_space, spaces.Dict):
            raise RuntimeError("FrameSkipWrapper needs Dict action space")

    def step(self, action):
        # Take copy of the action as we are going to modify it soon
        action = deepcopy(action)
        # First, normalize camera actions according to frameskip,
        # otherwise we might go haywire.
        if "camera" in action.keys():
            action["camera"] = action["camera"] // self.frame_skip

        reward_sum = 0
        for i in range(self.frame_skip):
            obs, reward, terminal, info = self.env.step(action)
            reward_sum += reward
            if terminal:
                break
            if i == 0:
                # After first step, remove craft/smelt/nearbyCraft etc commands
                # as we might not want to play those many times
                action["craft"] = 0
                action["nearbyCraft"] = 0
                action["nearbySmelt"] = 0
                action["equip"] = 0
                action["place"] = 0

        return obs, reward_sum, terminal, info
