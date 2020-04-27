from collections import OrderedDict
import termcolor
import random

import numpy as np
import gym
from gym import spaces

# Default speed for camera movement per tick
DEFAULT_MOUSE_SPEED = 10
# Margin for mouse actions when reading human
# actions: Mouse speed has to be above this
# be considered movement when converting
# floats to discretes
DEFAULT_MOUSE_MARGIN = 1

# Dictionary mapping actions that are forced
# to zeros when we want to use minimal actions.
# (I.e. we remove those actions for set of possible actions)
FORCED_ACTIONS_IN_MINIMAL_ACTIONS = {
    "back": 0,
    "left": 0,
    "right": 0,
    "sneak": 0,
    "sprint": 0
}


def sign(x):
    """
    Return sign of x (-1 if neg, 1 if pos and 0 if zero)
    """
    if x < 0:
        return -1
    elif x > 0:
        return 1
    else:
        0


def mouse_action_to_discrete(camera_action):
    """Turns mouse action from dict into discrete one

    Takes in one value (float) and returns one value in {0,1,2}.
    0 being no movement, 1 neg movement and 2 pos movement.
    """
    if camera_action < -DEFAULT_MOUSE_MARGIN:
        return 1
    elif camera_action > DEFAULT_MOUSE_MARGIN:
        return 2
    else:
        return 0


def discrete_to_mouse_action(discrete_action, mouse_speed):
    """Turn multidiscrete mouse action to float.

    Takes in value {0,1,2} and mouse speed, and returns one float scalar.
    """

    if discrete_action == 1:
        return -mouse_speed
    elif discrete_action == 2:
        return mouse_speed
    else:
        return 0


class ObtainDiamondActions:
    """Turn action space of ObtainDiamond to something more managable by networks.

    Converts the dict into a multidiscrete array
    """
    def __init__(self, action_space, mouse_speed=DEFAULT_MOUSE_SPEED,
                 minimal_actions=True, force_actions=None, replace_actions=None):
        """
        minimal_actions: If True, reduce action space to only actions we need
        force_actions: Mapping of discrete_names -> integer we use to force that action
                       on each step. Does not work for camera!
        replace_actions: Dict mapping from action name to list of (original_action, replaced_action).
                         If discrete variable of that name takes original_action, then replace
                         it with replaced_action. Used to remove some actions
        """
        self.mouse_speed = mouse_speed
        self.old_action_space = action_space
        self.force_actions = force_actions
        self.replace_actions = replace_actions

        if self.force_actions is None:
            self.force_actions = {}

        if self.replace_actions is None:
            self.replace_actions = {}

        if minimal_actions:
            # Set "unneeded actions" to no-ops
            self.force_actions.update(FORCED_ACTIONS_IN_MINIMAL_ACTIONS)
        self.forced_action_keys = set(self.force_actions.keys())

        # Create lengths for discrete sizes
        self.original_keys = list(self.old_action_space.spaces.keys())

        self.discrete_sizes = []
        # This will contain names for all discrete items
        self.discrete_names = []
        for key, space in self.old_action_space.spaces.items():
            if key in self.forced_action_keys:
                # This key is not part of our action space, skip
                continue
            if key == "camera":
                # Special handling: For both axis, add
                # nul/neg/pos actions.
                # We need to do the forced_action check here
                # too
                if "camera_x" not in self.forced_action_keys:
                    self.discrete_sizes.append(3)
                    self.discrete_names.append("camera_x")
                if "camera_y" not in self.forced_action_keys:
                    self.discrete_sizes.append(3)
                    self.discrete_names.append("camera_y")
            else:
                # Enum or Discrete, but both work with same principle
                self.discrete_sizes.append(space.n)
                self.discrete_names.append(key)

        self.num_discretes = len(self.discrete_sizes)
        self.action_space = spaces.MultiDiscrete(self.discrete_sizes)

    def get_no_op(self):
        """Return multidiscrete action representing no-op"""
        return [0] * self.num_discretes

    def flip_left_right(self, action):
        """Flip action horizontally, in-place"""
        # Flip camera turning
        if "camera_x" in self.discrete_names:
            camera_x_idx = self.discrete_names.index("camera_x")
            camera_x_action = action[camera_x_idx]
            if camera_x_action == 1:
                action[camera_x_idx] = 2
            elif camera_x_action == 2:
                action[camera_x_idx] = 1

        # Flip left/right movement
        # Assume that if left is available, then right
        # is available as well
        if "left" in self.discrete_names:
            left_idx = self.discrete_names.index("left")
            right_idx = self.discrete_names.index("right")
            # Swap these actions
            left_action = action[left_idx]
            action[left_idx] = action[right_idx]
            action[right_idx] = left_action

    def dict_to_multidiscrete(self, action_dict):
        """Convert ObtainDiamond action dict into multidiscrete

        Takes in dict from MineRL, returns a multidiscrete
        """
        discrete_actions = [None for i in range(self.num_discretes)]
        for i, key in enumerate(self.discrete_names):
            if key == "camera_x":
                discrete_actions[i] = mouse_action_to_discrete(action_dict["camera"][1])
            elif key == "camera_y":
                discrete_actions[i] = mouse_action_to_discrete(action_dict["camera"][0])
            else:
                discrete_actions[i] = int(action_dict[key])
        return discrete_actions

    def multidiscrete_to_dict(self, action_multi):
        """Convert multidiscrete action into ObtainDiamond action dict

        Takes in multidiscrete action, returns MineRL action dict
        """
        # Initialize actions to invalid values so we crash if something
        # in code goes wrong.
        action_dict = OrderedDict([(key, None) for key in self.original_keys])
        # Initialize camera action to zeros too
        action_dict["camera"] = np.zeros((2,), dtype=np.float32)

        # Update action dict with actions
        for discrete_i in range(len(self.discrete_names)):
            discrete_name = self.discrete_names[discrete_i]
            discrete_value = action_multi[discrete_i]
            # Special care for camera actions
            if discrete_name == "camera_x":
                action_dict["camera"][1] = discrete_to_mouse_action(discrete_value, self.mouse_speed)
            elif discrete_name == "camera_y":
                action_dict["camera"][0] = discrete_to_mouse_action(discrete_value, self.mouse_speed)
            else:
                action_dict[discrete_name] = int(discrete_value)

        # Go through replace actions to see if any actions require replacing
        for replace_name, replace_mappings in self.replace_actions.items():
            # Check if replace_name even is in our dictionary
            if replace_name in action_dict.keys():
                # Go through the hardcoded actions
                original_action = action_dict[replace_name]
                for replace_mapping in replace_mappings:
                    if original_action == replace_mapping[0]:
                        # Replace action
                        action_dict[replace_name] = replace_mapping[1]
                        break

        # Update dictionary with forced actions
        for forced_name, forced_value in self.force_actions.items():
            if forced_name == "camera_x":
                action_dict["camera"][1] = discrete_to_mouse_action(forced_value, self.mouse_speed)
            elif forced_name == "camera_y":
                action_dict["camera"][0] = discrete_to_mouse_action(forced_value, self.mouse_speed)
            else:
                action_dict[forced_name] = int(forced_value)

        return action_dict

    def probabilities_to_multidiscrete(self, predictions, greedy=False, epsilon=0.0):
        """
        Turn probabilities (1D vector of length sum(self.discrete_sizes))
        to multidiscrete actions

        If greedy is False, sample actions, else use action with highest probability.

        Choose random action per discrete variable with probability epsilon.
        """
        multidiscrete_action = []
        idx = 0
        for discrete_size in self.discrete_sizes:
            probabilities = predictions[idx:idx + discrete_size]
            action = None
            if greedy:
                # Take one with highest probability
                action = np.argmax(probabilities)
            elif random.random() < epsilon:
                # Take random action
                action = random.choice(range(discrete_size))
            else:
                # Randomly sample action
                action = np.random.choice(range(discrete_size), p=probabilities)
            multidiscrete_action.append(action)
            idx += discrete_size
        return multidiscrete_action

    def q_values_to_multidiscrete(self, all_q_values, epsilon=0.0):
        """Turn q_values (1D vector of length sum(self.discrete_sizes))
        to multidiscrete actions

        Use epsilon-greedy policy according to epsilon value.
        """
        multidiscrete_action = []
        idx = 0
        for discrete_size in self.discrete_sizes:
            q_values = all_q_values[idx:idx + discrete_size]
            action = None
            # Epsilon-greedy
            if random.random() > epsilon:
                # Take one with highest q_value
                action = np.argmax(q_values)
            else:
                # Random action
                action = random.randint(0, discrete_size - 1)
            multidiscrete_action.append(action)
            idx += discrete_size
        return multidiscrete_action

    def print_probabilities(self, probabilities):
        """Print probabilities/predictions with keys for debugging purposes"""
        idx = 0
        for i, discrete_size in enumerate(self.discrete_sizes):
            action_probs = probabilities[idx:idx + discrete_size]
            # Turn into more convenient format
            # Highlight highest value
            highest_prob = max(action_probs)
            prob_string = ""
            for action_prob in action_probs:
                action_string = "{:.3f}".format(action_prob)
                # Highlight
                if action_prob == highest_prob:
                    action_string = termcolor.colored(action_string, on_color="on_green")
                prob_string += action_string + " "
            print("{:<20} {}".format(self.discrete_names[i], prob_string))
            idx += discrete_size

    def print_q_values(self, q_values):
        """Print q_values from branching_q for debugging purposes"""
        idx = 0
        for i, discrete_size in enumerate(self.discrete_sizes):
            action_probs = q_values[idx:idx + discrete_size]
            # Turn into more convenient format
            # Highlight highest value
            highest_prob = max(action_probs)
            prob_string = ""
            for action_prob in action_probs:
                action_string = "{:>6.3f}".format(action_prob)
                # Highlight
                if action_prob == highest_prob:
                    action_string = termcolor.colored(action_string, on_color="on_green")
                prob_string += action_string + " "
            print("{:<20} {}".format(self.discrete_names[i], prob_string))
            idx += discrete_size
