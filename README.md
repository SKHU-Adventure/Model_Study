# AI-Place-Recognition

For all main contributors, please check [contributing](#contributing).

## Introduction

This repository is dedicated to research and development of AI-based place recognition.

## How To Use

### Clone 

Clone this GitHub repository:

```
git clone https://github.com/SKHU-Adventure/ai-place-recognition.git
cd ai-place-recognition
```

### Requirements

The main branch works with **CUDA 11.6**, **CUDNN 8.9.5** with **Python 3.8**.
Refer to the following command to install python requirements:

```bash
pip install -r requirements.txt
```

### Prepare Datasets

1. Prepare dataset for training: 

For public datasets, you may refer to following links.
- [Nordland](https://drive.google.com/drive/folders/1CzzLo-t9iLYOszcHAnB3KaWwkP5jyJn1?usp=sharing)
- [Tokyo](https://www.di.ens.fr/willow/research/netvlad/) (available on request)

### How to Train

1. Create a directory for your experiment (e.g., `experiments/sample`)

2. Write a setup.ini file in the directory (e.g., `experiments/sample/setup.ini`).

2. Run:
```bash
torchrun --nproc_per_node=[NUM_GPU] train.py [EXPERIMENT_DIR]
```
where `[NUM_GPU]` is the number of GPUs for distributed training and `[EXPERIMENT_DIR]` is the directory created above.

### How to Evaluate


### How to Demo


## Contributing

Main contributors:

- [Mujae Park](https://github.com/Mujae), ``mujae9837[at]gmail.com``

Advisior:
- [Sangyun Lee](https://sylee-skhu.github.io), ``sylee[at]skhu.ac.kr``
