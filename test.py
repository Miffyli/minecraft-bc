import logging
import os

import numpy as np
import gym
import minerl

from wrappers.observation_wrappers import FrameSkipWrapper, ObtainDiamondObservation
from wrappers.action_wrappers import ObtainDiamondActions

# All the evaluations will be evaluated on MineRLObtainDiamond-v0 environment
MINERL_GYM_ENV = os.getenv('MINERL_GYM_ENV', 'MineRLObtainDiamond-v0')
MINERL_MAX_EVALUATION_EPISODES = int(os.getenv('MINERL_MAX_EVALUATION_EPISODES', 5))


def run_imitation_test():
    import keras

    # Some hardcoded values for the submission
    MOUSE_SPEED = 8
    FRAMESKIP = 4

    # Some action-space limitation on the multi-discrete,
    # similar to what was done baselines (removing/forcing actions).
    # Removed in evaluation phase by replacing with forced actions
    REPLACE_ACTIONS = {
        "equip": [
            (1, 0)  # Do not unequip items
        ],
        "nearbyCraft": [
            (1, 0),  # Do not craft wooden axe
            (3, 0),  # Do not craft stone axe
            (5, 0),  # Do not craft iron axe
        ]
    }

    # We need to replace the loss contained in the model
    # with something else, as it is not available here.
    pi_model = keras.models.load_model(
        "train/imitation_impala_epochs25",
        custom_objects={
            "multicategorical_loss": keras.losses.binary_crossentropy,
        }
    )
    env = gym.make(MINERL_GYM_ENV)

    if FRAMESKIP > 1:
        env = FrameSkipWrapper(env, FRAMESKIP)

    obs_handler = ObtainDiamondObservation(env.observation_space)
    act_handler = ObtainDiamondActions(
        env.action_space,
        mouse_speed=MOUSE_SPEED,
        replace_actions=REPLACE_ACTIONS
    )

    for game_i in range(MINERL_MAX_EVALUATION_EPISODES):
        obs = env.reset()

        # Modification since submission: Track episodic rewards
        # and print them out
        episodic_reward = 0.0

        done = False
        while not done:
            keras_obs = obs_handler.dict_to_tuple(obs)
            model_input = [
                np.array(keras_obs[0])[None], np.array(keras_obs[1])[None]
            ]

            action = None

            predicts = pi_model.predict(model_input)[0]

            action = act_handler.probabilities_to_multidiscrete(
                predicts
            )

            action_dict = act_handler.multidiscrete_to_dict(action)
            obs, reward, done, info = env.step(action_dict)

            episodic_reward += reward

        print("Episode {}. Reward {}".format(game_i, episodic_reward))

    env.close()


def main():
    """
    This function will be called for testing phase.
    """
    run_imitation_test()


if __name__ == "__main__":
    main()
