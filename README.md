# Reachout_triage
We provide code to finetune a GPT model, train on the Reachout forum data and make predictions on the associated held-out test data as reported in [Transfer Learning for Risk Classification of Social Media Posts: Model Evaluation Study](https://arxiv.org/abs/1907.02581).

This repository contains no task data. The training data is from the CLPsych 2017 Shared task: Triaging content in online peer-support forums. Additional evaluation was performed on the [The University of Maryland Reddit Suicidality Dataset](http://users.umiacs.umd.edu/~resnik/umd_reddit_suicidality_dataset.html).

#### To recreate the environment used for feature feature generation:
``` 
conda env create -f environment.yml
source activate reachout
```

To install autosklearn, we recommend following the instructions in [autosklearn's documentation](https://automl.github.io/auto-sklearn/master/installation.html#) depending on your system.
