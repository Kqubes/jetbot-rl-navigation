# Reinforcement Learning for JetBot Navigation in NVIDIA Isaac Sim

## Overview

This project investigates reinforcement-learning-based autonomous
navigation using NVIDIA JetBot in NVIDIA Isaac Sim and Isaac Lab.

A Proximal Policy Optimization (PPO) policy is trained to perform
goal-directed navigation using RGB camera observations while avoiding
obstacles. The project subsequently investigates whether Adaptive Domain
Randomization (ADR) improves policy generalisation and robustness under
variations in sensing and physical parameters.

The implementation uses Isaac Lab's `DirectRLEnv` workflow and RSL-RL
for PPO training.

------------------------------------------------------------------------

## Project Objectives

- Develop a vision-based navigation environment for NVIDIA JetBot.
- Train a PPO baseline policy for goal navigation and obstacle avoidance.
- Implement Adaptive Domain Randomization for improved generalisation.
- Evaluate trained policies in unseen environments.
- Test policy robustness under camera and physical perturbations.
- Compare baseline and ADR policies using success, collision, and
  timeout rates.

------------------------------------------------------------------------

## System Overview

### Hardware

| Component | Specification |
|---|---|
| GPU | NVIDIA GeForce RTX 4090 |
| Robot | NVIDIA JetBot (simulated) |

### Software

| Component | Version / Configuration |
|---|---|
| NVIDIA Isaac Sim | 6.0.1 |
| NVIDIA Isaac Lab | 3.0 |
| RSL-RL | 5.0 |
| RL Algorithm | PPO |
| Environment Workflow | `DirectRLEnv` |

### Perception and Navigation

| Component | Configuration |
|---|---|
| Camera | RGB |
| Image Resolution | 64 × 64 |
| Temporal History | 3 RGB frames |
| Obstacles | 5 per environment |
| Goal Generation | Random |
| Robot Control | Differential-drive wheel control |
| Policy Actions | Continuous left and right wheel commands |

The hardware and software specifications are reported to improve
experimental reproducibility. Training performance and computational
throughput may vary when the experiments are reproduced using different
GPU hardware.

------------------------------------------------------------------------

## Baseline Environment

The baseline task consists of vision-based goal navigation with obstacle
avoidance.

The policy receives:

- RGB camera observations from the JetBot.
- A temporal history of three camera frames.
- Relative goal-direction information.
- Distance to the navigation goal.

The policy outputs two continuous actions controlling the left and right
wheels.

An episode terminates when:

1. The robot reaches the navigation goal.
2. The robot collides with an obstacle.
3. The maximum episode duration is reached.

------------------------------------------------------------------------

## Policy Architecture

Camera observations are processed using a convolutional neural network
(CNN).

Each temporal RGB frame is encoded using a shared CNN encoder. The
extracted visual features from the three frames are concatenated with
the relative navigation-goal state and subsequently processed using
fully connected layers.

Separate actor and critic models are used through the RSL-RL PPO
implementation. The actor generates continuous wheel-control actions,
while the critic estimates the value of the current state during policy
optimisation.

------------------------------------------------------------------------

## Adaptive Domain Randomization

Following baseline training, Adaptive Domain Randomization was
introduced to investigate whether exposure to simulation variability
improves policy generalisation.

The ADR experiments progressively introduced variations in:

- Wheel-related physical parameters.
- Camera/perception parameters.
- Floor-friction parameters.

Two principal ADR configurations were investigated:

1. **ADR Wheel + Camera**
2. **ADR Wheel + Camera + Floor**

Multiple training checkpoints were evaluated rather than assuming that
the final training checkpoint represented the best-performing policy.

------------------------------------------------------------------------

## Evaluation

Three trained policy configurations were compared:

1. **Baseline**
2. **ADR Wheel + Camera**
3. **ADR Wheel + Camera + Floor**

### Nominal Unseen Evaluation

Policies were evaluated using unseen random seeds under nominal
simulation conditions.

The principal evaluation metrics were:

- Success rate
- Collision rate
- Timeout rate
- Mean episode length
- Variation across unseen seeds

### Robustness Evaluation

Policies were additionally evaluated under controlled simulation
perturbations involving:

- Camera noise
- Wheel-friction variation
- Floor-friction variation
- Combined perturbations

These experiments were used to determine whether policies trained using
ADR retained their performance when exposed to conditions different from
the nominal simulation environment.

------------------------------------------------------------------------

## Final Results

### Nominal Unseen-Environment Performance

| Model | Success Rate | Collision Rate | Timeout Rate |
|---|---:|---:|---:|
| **Baseline** | 77.0% | 10.5% | 12.5% |
| **ADR Wheel + Camera** | 80.0% | 7.0% | 13.0% |
| **ADR Wheel + Camera + Floor** | **82.0%** | **1.5%** | 16.5% |

Under nominal unseen conditions, the **ADR Wheel + Camera + Floor**
policy achieved the highest success rate of **82.0%** and the lowest
collision rate of **1.5%**.

### Average Robustness Performance

| Model | Average Success Rate |
|---|---:|
| **Baseline** | **62.25%** |
| **ADR Wheel + Camera** | 61.75% |
| **ADR Wheel + Camera + Floor** | 45.57% |

Although ADR improved nominal unseen-environment performance, the
robustness experiments produced a different result. The baseline policy
achieved the highest average success rate across the tested
perturbations.

These experiments demonstrate that improved nominal performance does not
necessarily correspond to improved robustness across all physical and
perceptual perturbations. In particular, introducing additional
randomisation dimensions during training did not consistently improve
performance under every evaluated perturbation.

------------------------------------------------------------------------

## Relation to Prior Work

The initial vision-based JetBot navigation implementation was informed
by the open-source `jetbot-nav-rl` project:

[shahizat/jetbot-nav-rl — Training a Vision-Based Navigation Policy
using NVIDIA Isaac Lab](https://github.com/shahizat/jetbot-nav-rl)

The reference implementation provided guidance for selected components
of the initial navigation task, including aspects of the reward
formulation, environment reset strategy, and convolutional visual
encoder architecture.

The implementation in this repository was subsequently reworked and
extended for the requirements of this project.

Major differences and extensions include:

- Migration from SKRL to the RSL-RL PPO framework.
- Integration with the RSL-RL actor/critic model interface.
- Replacement of the original combined observation representation with
  separate image and navigation-state observations.
- Use of three temporal RGB frames for vision-based navigation.
- Replacement of contact-sensor-based collision detection with
  distance-based obstacle collision detection.
- Migration from `TiledCamera` to the Isaac Lab `Camera` interface.
- Restructuring of obstacle and environment management.
- Reduction to five obstacles for the experimental navigation
  environment.
- Addition of training and episode-level performance instrumentation.
- Development of Adaptive Domain Randomization for camera noise, wheel
  friction, and floor friction.
- Development of two ADR configurations: Wheel + Camera and Wheel +
  Camera + Floor.
- Development of checkpoint evaluation and policy selection.
- Development of unseen-environment evaluation.
- Development of controlled robustness experiments involving camera,
  wheel, floor-friction, and combined perturbations.
- Development of interactive robustness testing and parameter
  visualisation.

The reference implementation was therefore used as a starting point for
selected components of the baseline navigation system rather than as the
complete experimental pipeline.

------------------------------------------------------------------------
## How to Run

Run all scripts from the project root using the Isaac Lab Python environment.

### Training

Baseline training:

```bash
~/IsaacLab/isaaclab.sh -p scripts/train_baseline.py --enable_cameras --headless
```

ADR training:

```bash
~/IsaacLab/isaaclab.sh -p scripts/train_adr.py --enable_cameras --headless
```

### Checkpoint Evaluation

Evaluate baseline checkpoints:

```bash
~/IsaacLab/isaaclab.sh -p scripts/evaluate_checkpoints.py --enable_cameras --headless
```

Evaluate ADR checkpoints:

```bash
~/IsaacLab/isaaclab.sh -p scripts/evaluate_adr_checkpoints.py --enable_cameras --headless
```

### Final Evaluation

Evaluate the trained policies on unseen environments:

```bash
~/IsaacLab/isaaclab.sh -p scripts/evaluate_unseen.py --enable_cameras --headless
```

Run the final policy comparison:

```bash
~/IsaacLab/isaaclab.sh -p scripts/evaluate_final_comparison.py --enable_cameras --headless
```

### Robustness Testing

Run the complete robustness evaluation:

```bash
~/IsaacLab/isaaclab.sh -p scripts/evaluate_robustness.py --enable_cameras --headless
```

The robustness evaluation tests the trained policies under camera-noise,
wheel-friction, floor-friction, and combined perturbations.

### Interactive Robustness Testing

Run the robustness play script with Isaac Sim visualisation:

```bash
~/IsaacLab/isaaclab.sh -p scripts/play_robustness.py \
    --enable_cameras \
    --visualizer kit \
    --livestream 2
```

This allows the trained policy to be observed interactively while testing
camera-noise, wheel-friction, and floor-friction perturbations. The
parameter-monitoring visualisation is displayed during testing.

### Policy Visualisation

Run a trained policy with Isaac Sim visualisation:

```bash
~/IsaacLab/isaaclab.sh -p scripts/play.py \
    --enable_cameras \
    --visualizer kit \
    --livestream 2
```
------------------------------------------------------------------------

## Reproducibility

Separate scripts are provided for baseline training, ADR training,
checkpoint evaluation, unseen-environment evaluation, and robustness
testing.

Trained checkpoints are stored under:

```text
logs/jetbot_baseline/
logs/jetbot_adr/
```

Final experimental results and generated plots are stored under:

```text
final_results/
```

The exact numerical results obtained during retraining may vary
depending on random seeds, software versions, simulation configuration,
and available hardware.

------------------------------------------------------------------------

## License

GPL-3.0 — see [`LICENSE`](LICENSE) for details.

------------------------------------------------------------------------

## Acknowledgements and References

- [NVIDIA Isaac Sim](https://developer.nvidia.com/isaac/sim)
- [NVIDIA Isaac Lab](https://github.com/isaac-sim/IsaacLab)
- [RSL-RL](https://github.com/leggedrobotics/rsl_rl)
- [Isaac Lab — Training the JetBot: Ground
  Truth](https://isaac-sim.github.io/IsaacLab/main/source/setup/walkthrough/training_jetbot_gt.html)
  — referenced for the JetBot environment workflow and visualization
  marker implementation.
- [shahizat/jetbot-nav-rl](https://github.com/shahizat/jetbot-nav-rl)
  — reference implementation used during the initial development of
  selected components of the vision-based JetBot navigation
  environment.

The `jetbot-nav-rl` reference implementation is licensed under the GNU
General Public License v3.0. This project acknowledges the original
implementation and documents the modifications and experimental
extensions introduced as part of the present work.