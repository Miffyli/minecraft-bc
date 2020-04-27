# UEFDRL submission to MineRL 2019 challenge
This repository contains the final ranked submission to the [MineRL 2019 challenge](https://www.aicrowd.com/challenges/neurips-2019-minerl-competition),
reaching fifth place. 

A paper detailing the submission and related experiments: To be added.

Short story in shorter format: Behavioural cloning on the provided dataset. No RNNs.

## Contents

Code is in the submission format, and can be ran with the instructions at [submission template repository](https://github.com/minerllabs/competition_submission_starter_template).
`requirements.txt` contains Python modules required to run the code, and `apt.txt` includes any Debian packages required (used by the Docker image in AICrowd evaluation server).

The core of our submission resides in `train_keras_imitation.py`, which contains the main training loop. 

## Running

[Download](http://minerl.io/dataset/) and place MineRL dataset under `./data`. Alternatively point environment variable `MINERL_DATA_ROOT` to the downloaded dataset.

Run `train.py` to train the model, after run `test.py` to run the evaluation used in the AICrowd platform. This code prints out per-episode rewards.

After 200 games, the average episodic reward should be around 10-13. The results very from to run / depending on when training is stopped, and we also
noticed our local evaluations having consistently lower score than on AICrowd platform (achieved +15 results).
