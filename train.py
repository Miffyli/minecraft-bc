# Main training code
import logging
import os

import train_keras_imitation

# All the evaluations will be evaluated on MineRLObtainDiamond-v0 environment
MINERL_GYM_ENV = os.getenv('MINERL_GYM_ENV', 'MineRLObtainDiamond-v0')
# You need to ensure that your submission is trained in under MINERL_TRAINING_MAX_STEPS steps
MINERL_TRAINING_MAX_STEPS = int(os.getenv('MINERL_TRAINING_MAX_STEPS', 8000000))
# You need to ensure that your submission is trained by launching less than MINERL_TRAINING_MAX_INSTANCES instances
MINERL_TRAINING_MAX_INSTANCES = int(os.getenv('MINERL_TRAINING_MAX_INSTANCES', 5))
# You need to ensure that your submission is trained within allowed training time.
# Round 1: Training timeout is 15 minutes
# Round 2: Training timeout is 4 days
MINERL_TRAINING_TIMEOUT = int(os.getenv('MINERL_TRAINING_TIMEOUT_MINUTES', 4 * 24 * 60))
# The dataset is available in data/ directory from repository root.
MINERL_DATA_ROOT = os.getenv('MINERL_DATA_ROOT', 'data/')


def train_pure_imitation_learning():
    """
    Submission with pure imitation learning
    """
    # Yay for hardcoded inputs!
    imitation_arguments = [
        "--epochs", "25",
        "--cnn-head", "impala",
        "--l2", "1e-5",
        # Do not save unnecessary snapshot-files of the model
        "--save-every-updates", "100000000",
        MINERL_DATA_ROOT,
        "train/imitation_impala_epochs25",
        "MineRLObtainDiamond-v0", "MineRLObtainIronPickaxe-v0",
    ]

    os.makedirs("train", exist_ok=True)

    imitation_train_args = train_keras_imitation.parser.parse_args(imitation_arguments)

    train_keras_imitation.main(imitation_train_args)


def main():
    """
    This function will be called for training phase.
    """
    train_pure_imitation_learning()


if __name__ == "__main__":
    main()
