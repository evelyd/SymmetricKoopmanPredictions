<h1 align="center">
SKooP: Symmetric Koopman Predictions for Faster and More Generalizable Legged Robot Locomotion with Reinforcement Learning
</h1>

<div align="center">
Evelyn D'Elia<sup>1,2</sup>, Weishu Zhan<sup>2</sup>, Giulio Turrisi<sup>3</sup>, Giulio Romualdi<sup>4</sup>, Giuseppe L'Erario<sup>4</sup>, Raffaello Camoriano<sup>5,6</sup>, Wei Pan<sup>7</sup>, Daniele Pucci<sup>4</sup> <br> <br>
</div>

<div align="center">
  <span class="author-block" style="font-size: 0.95em;"><sup>1</sup> IIT@MIT, Italian Institute of Technology (IIT), Genoa, Italy<br><sup>2</sup> Machine Learning and Optimisation, University of Manchester, U.K.<br><sup>3</sup> Dynamic Legged Systems Laboratory, IIT, Genoa, Italy<br><sup>4</sup> Generative Bionics S.R.L, Genoa, Italy<br><sup>5</sup> DAUIN, Politecnico di Torino, Turin, Italy<br><sup>6</sup> Rehab Technologies Lab, IIT, Genoa, Italy<br><sup>7</sup> School of Engineering, Newcastle University, U.K.</span>
  <span class="eql-cntrb" style="font-size: 0.8em;"><small><br>E. D'Elia, G. Romualdi, G. L'Erario, and D. Pucci contributed to this work while at the Artificial and Mechanical Intelligence lab, IIT, Italy. W. Pan contributed to this work while at the Machine Learning and Optimisation group, University of Manchester, U.K.</small></span>
</div>


<div align="center">
    📅 Accepted to the 2026 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)
</div>

## Overview

We propose, SKooP, an RL-based approach which leverages Koopman model predictions and morphological symmetries to achieve faster policy convergence and better sample efficiency. This repository allows the reproduction of the paper results for the following approaches:

- **PPO** (`<task-name>`)
- **PPOeqic** (`<task-name>_emlp`)
- **SKooP-NoSym-NoPred** (`<task-name>_cdae_online`)
- **SKooP-NoSym** (`<task-name>_cdae_online_next_latent`)
- **SKooP-NoPred** (`<task-name>_emlp_ecdae_online`)
- **SKooP** (`<task-name>_emlp_ecdae_online_next_latent`)

### Supported Tasks

- Stand Dance
- Walk Slope
- Push Door

## Installation

### Prerequisites

- Python 3.8+
- CUDA 11.3 compatible GPU (optional, can run on CPU)

### Step 1: Create Conda Environment

```bash
conda create -n symmloco python==3.8
conda activate symmloco
```

### Step 2: Install PyTorch

```bash
pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
```

### Step 3: Install Isaac Gym

1. Download Isaac Gym Preview 4 from [NVIDIA Developer](https://developer.nvidia.com/isaac-gym)
2. Extract the archive:
    ```bash
    tar -xf IsaacGym_Preview_4_Package.tar.gz
    ```
3. Install the Python package:
    ```bash
    cd isaacgym/python && pip install -e .
    ```

### Step 4: Install Project Dependencies

```bash
# Install MorphoSymm (symmetry utilities)
cd SymmLoco/MorphoSymm && pip install -e .

# Install rsl_rl (reinforcement learning)
cd ../rsl_rl && pip install -e .

# Install legged_gym (environment framework)
cd ../legged_gym && pip install -e .
```

### Step 5: Install Additional Dependencies
```bash
# Install MorphoSymm (symmetry utilities)
pip install tensorboard wandb
```

## Training and Inference

### Sample Training Command

```bash
cd legged_gym
python legged_gym/scripts/train.py --task=cyber2_push_door_emlp_ecdae_online_next_latent --headless --right
```

### Sample Play Command

```bash
cd legged_gym
python legged_gym/scripts/play.py --task=cyber2_push_door_emlp_ecdae_online_next_latent --headless --right
```

### Command-line Arguments
* `--task`: Task name
* `--headless`: Run without rendering, e.g. for headless server
* `--left`: Mirror push_door task environment
* `--seed`: Specify non-default seed, use `-1` for random
* `--load_run`: Specify the name of the saved log to load
* `--checkpoint`: Specify the checkpoint number to load

## Data analysis

The `analysis_scripts` directory contains scripts for reproducing selected paper results.

1. Install Hugging Face CLI and authenticate:
    ```bash
    conda install -c conda-forge huggingface_hub
    hf auth login
    ```
2. Download the dataset:
   ```bash
   hf download evelyd/SymmetricKoopmanPredictions --repo-type dataset --local-dir .
   ```
3. Run an analysis script, for example:
    ```bash
    cd analysis_scripts/
    python plot_joint_angles.py
    ```

## Citations

This project uses Isaac Gym and builds upon the following libraries:
* [MorphoSymm](https://github.com/Danfoa/MorphoSymm)
* [SymmLoco](https://github.com/HybridRobotics/SymmLoco)
* [DynamicsHarmonicsAnalysis](https://github.com/Danfoa/DynamicsHarmonicsAnalysis)
* [RSL-RL](https://github.com/leggedrobotics/rsl_rl)

If you use SKooP in your research, please cite our paper:

```bibtex
@inproceedings{delia2026symmetrickoopmanpredictions,
      title={SKooP: Symmetric Koopman Predictions for Faster and More Generalizable Legged Robot Locomotion with Reinforcement Learning}, 
      author={Evelyn D'Elia and Weishu Zhan and Giulio Turrisi and Giulio Romualdi and Giuseppe L'Erario and Raffaello Camoriano and Wei Pan and Daniele Pucci},
      year={2026},
      booktitle={2026 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
}
```

## Maintainer

<table align="left">
    <tr>
        <td><a href="https://github.com/evelyd"><img src="https://github.com/evelyd.png" width="40"></a></td>
        <td><a href="https://github.com/evelyd"> @evelyd</a></td>
    </tr>
</table>