# Policy Distillation for MuJoCo Environments

This repository implements **policy distillation** techniques for transferring knowledge from a state-based teacher policy to a **vision-based student policy** in MuJoCo continuous control tasks (e.g., Pusher-v5).

The student model can process both **rendered image frames** and **internal robot state variables** (e.g., joint positions, velocities), enabling multimodal learning from pixel observations combined with proprioceptive information.

---

## Distillation Algorithms

Four distillation methods are implemented, each with different trade-offs:

### 1. Policy Proximal Distillation (PPD)
An on-policy method that combines PPO's policy gradient objective with a distillation loss from the teacher. The student collects its own rollouts in the environment and optimizes a combined objective:

$$L = L_{\text{PPO}} + \lambda \cdot L_{\text{distill}}$$

where $\lambda$ (`PPD_coef`) controls the balance between RL exploration and teacher imitation.

### 2. Teacher Distillation (Online)
The **teacher** selects actions in the environment while the student learns to match them. The student is trained using PPO but with the teacher driving the data collection. This provides a stable training signal since the teacher generates high-quality trajectories.

### 3. Student Distillation (Online)
The **student** selects actions in the environment and the teacher provides corrective supervision. The student optimizes a combination of RL rewards and distillation from the teacher's policy evaluated on the same states. This method encourages the student to explore while learning from the teacher.

### 4. Behavioural Cloning + DAgger

A two-phase offline-to-online approach:

- **Phase 1 (BC)**: The student is trained purely offline on a pre-collected dataset of teacher demonstrations using behavioural cloning (supervised learning on state-action pairs).
- **Phase 2 (DAgger)**: The student is deployed in the environment to collect new data, which is aggregated with the original dataset. The teacher labels the student-visited states with the correct actions, and training continues on the expanded dataset.

---

## Prioritized Experience Replay Buffer

For offline training (BC and DAgger phases), the repository includes a **Prioritized Experience Replay** buffer with **importance sampling** correction:

- **Priority-based sampling**: Samples with higher prediction error are drawn more frequently (controlled by `alpha`).
- **Importance sampling weights**: Corrects the bias introduced by non-uniform sampling (controlled by `beta`, annealed from an initial value toward 1.0).
- **Alpha annealing modes**: Supports `constant`, `linear`, `dynamic_mean`, and `dynamic_max` schedules to adapt priorities during training.
- **Balanced batch sampling**: An optional sampler that draws 50% old data and 50% new data per batch, ensuring both are represented.

---

### Visual Results

| PPD | StudentDistillation | TeacherDistillation |
| :---: | :---: | :---: |
| <video src="https://github.com/user-attachments/assets/b5d740b4-4f5d-4caf-bc31-6128925be980" autoplay loop muted playsinline width="100%"></video> | <video src="https://github.com/user-attachments/assets/b42d35b9-05b1-434b-b04e-e49e576f4e83" autoplay loop muted playsinline width="100%"></video> | <video src="https://github.com/user-attachments/assets/5d600667-10cd-4b67-8b87-8cb86174841f" autoplay loop muted playsinline width="100%"></video> |

| BC-10k dataset | BC-50k dataset |
| :---: | :---: |
| <video src="https://github.com/user-attachments/assets/bd35eb17-8d80-474e-9ea0-0660843eaedd" autoplay loop muted playsinline width="100%"></video> | <video src="https://github.com/user-attachments/assets/3f3a8531-14a9-457c-bbed-63281ae816bc" autoplay loop muted playsinline width="100%"></video> |

---

## Student Model Architectures

Three student model sizes are available, all based on the **IMPALA** CNN backbone:

| Model         | CNN Filters        | Description        |
|---------------|--------------------|--------------------|
| `ImpaalaSmall` | (16, 32)          | Lightweight model  |
| `ImpaalaMid`   | (16, 32, 32)      | Medium model       |
| `ImpaalaBig`   | (32, 64, 64)      | Full-capacity model|

All models use the same MLP actor head (128→128→action_dim) on top of the CNN backbone.

### IMPALA Architecture

The IMPALA (Importance Weighted Actor-Learner Architecture) backbone is a residual CNN designed for efficient visual feature extraction. It consists of:

1. **ConvSequence blocks**: Each block has a convolutional layer followed by max-pooling and two residual blocks.
2. **Residual blocks**: Standard residual connections with two 3×3 convolutions and ReLU activations, enabling deeper networks without degradation.
3. **Average pooling**: A spatial pooling layer at the end reduces the feature map to a fixed-size vector regardless of input dimensions.

This architecture strikes a good balance between computational efficiency and representational capacity, making it well-suited for RL tasks with visual observations.

---

## Repository Structure

```
├── Algorithm/                  # Core training algorithms
│   ├── PPD.py                  # Policy Proximal Distillation
│   ├── PPO.py                  # Proximal Policy Optimization (base)
│   ├── StudentTeacher_distillation.py  # Teacher/Student online distillation
│   ├── BehaviouralCloning.py   # BC + DAgger phases
│   ├── CollectDataset.py       # Teacher dataset collection
│   ├── Test_model.py           # Model evaluation utilities
│   ├── Utils.py                # DataLoader, Prioritized Buffer, helpers
│   └── seeds.py                # Centralized seed management
├── Models/                     # Neural network architectures
│   ├── Base_Model.py           # Base model class (actor-critic)
│   ├── Impoola.py              # IMPALA backbone + actor/critic builders
│   ├── Model_Pusher.py         # Pusher-specific model configurations
│   └── Utils.py                # Visual input manager, screen processing
├── Enviroment/                 # Environment wrappers
│   ├── My_wrapper.py           # Custom Gymnasium wrappers
│   └── Utils.py                # Environment creation utilities
├── Distillation/               # Experiment entry points
│   ├── Pusher.py               # Main script for Pusher distillation experiments
│   └── Utils.py                # Experiment configuration helpers
├── Train_Teacher_Pusher/       # Teacher training (SAC via Stable-Baselines3)
│   ├── Mujoco_main.py          # SAC training script
│   ├── callbacks.py            # Custom evaluation callbacks
│   ├── SaveWeights.py          # Weight extraction from SB3 to custom model
│   └── Test_Model.py           # Teacher model testing
├── Results/                    # Saved models and datasets
├── requirements.txt
└── README.md
```

---

## Installation

```bash
pip install -r requirements.txt
```

> **Note**: MuJoCo environments require a working MuJoCo installation. The `gymnasium[mujoco]` package handles this automatically.

---

## Usage

### 1. Train the Teacher (SAC)

```bash
cd Train_Teacher_Pusher
python Mujoco_main.py --device cuda:0 --total-timesteps 5000000
```

### 2. Collect a Dataset from the Teacher

```bash
python Distillation/Pusher.py --C_data --num_data 100000 --device cuda:0
```

### 3. Run Distillation

**PPD:**
```bash
python Distillation/Pusher.py --PPD --mode ImpaalaMid --PPD_parameter 5 --device cuda:0
```

**Teacher Distillation:**
```bash
python Distillation/Pusher.py --Tdistillation --mode ImpaalaMid --device cuda:0
```

**Student Distillation:**
```bash
python Distillation/Pusher.py --Sdistillation --mode ImpaalaMid --device cuda:0
```

**Behavioural Cloning (Phase 1):**
```bash
python Distillation/Pusher.py --BC_phase --mode ImpaalaMid --dataset 100k --alpha 0.0 --device cuda:0
```

**DAgger (Phase 2):**
```bash
python Distillation/Pusher.py --Dagger_phase --folder_Dagger ./Results/Pusher/OurAlgorithm1_NLL/Phase_One/<run_folder> --device cuda:0
```

### Key Arguments

| Argument        | Description                                           | Default    |
|-----------------|-------------------------------------------------------|------------|
| `--mode`        | Student architecture: `State`, `ImpaalaSmall`, `ImpaalaMid`, `ImpaalaBig` | `State` |
| `--dataset`     | Dataset size: `5k`, `10k`, `50k`, `100k`              | `100k`     |
| `--alpha`       | PER priority exponent (0.0 = uniform sampling)         | `0.0`      |
| `--loss_type`   | BC loss: `KL` (KL divergence) or `NLL` (negative log-likelihood) | `NLL` |
| `--PPD_parameter` | PPD distillation coefficient λ                       | `5`        |
| `--run_index`   | Run index for seed selection (0–5)                     | `0`        |

---

## Logging

All experiments are logged to [Weights & Biases](https://wandb.ai). Metrics include:
- Episode rewards (student and teacher)
- Action accuracy (student vs teacher)
- Policy entropy
- Training loss
- Buffer statistics (for prioritized replay)
- Evaluation videos
