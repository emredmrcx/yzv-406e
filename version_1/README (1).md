# Kindergarten Toy Sorting Robot 🤖🧸

A high-fidelity robotics simulation featuring a mobile manipulator designed for autonomous toy collection and sorting in a kindergarten environment. This project transition from kinematic teleportation to a fully physics-driven control system.

## 🚀 Overview

The **Kindergarten Robot** is a mobile base equipped with a **Frank Emika Panda 7-DOF** robotic arm. It is designed to navigate a cluttered room, identify colored toys (balls, cubes, ducks), and sort them into color-coordinated bins using advanced perception and manipulation algorithms.

## 🛠️ Technology Stack

- **Physics Engine**: [PyBullet](https://pybullet.org/) - Real-time multibody physics simulation with contact dynamics and forward/inverse kinematics.
- **Manipulation**: 7-DOF Panda Arm with a parallel-jaw gripper.
- **Navigation**: Physics-based velocity control with dynamic obstacle avoidance and target-aware docking.
- **Vision**: 
    - **OwlViT (Vision Transformer)**: Zero-shot text-conditioned object detection for identifying toys and obstacles.
    - **Color-based Fallback**: Robust OpenCV-style color segmentation for environments without GPU support.
- **Decision Making**: Probabilistic exploration grid and state-machine-driven autonomous collection loops.

## ✨ Key Features

- **Physics-Based Control**: Every movement—from base rotation to arm extension—is driven by velocity commands (`resetBaseVelocity`), ensuring realistic inertia and interactions.
- **Intelligent Exploration**: The robot utilizes a probabilistic search grid to efficiently patrol and map the environment.
- **Target-Aware Navigation**: Advanced avoidance logic allows the robot to bypass obstacles while seamlessly docking with target toys and bins.
- **Dynamic Tracking**: Real-time closed-loop tracking ensures the robot adapts if toys are moved during an approach phase.
- **Interactive Command Interface**: Supports natural language commands for specific tasks (e.g., "collect red toys", "patrol", "auto").

## 🚦 Getting Started

### Prerequisites

```bash
pip install pybullet numpy pillow
# Optional: for AI vision support
pip install transformers torch
```

### Execution

Run the robot in autonomous mode (Headless):
```bash
python3 main.py --headless
```

Run in interactive mode (GUI):
```bash
python3 main.py interactive
```

## 🎮 Interactive Commands

The robot features a natural language command parser. You can combine **colors** and **keywords** to filter tasks.

### Supported Filters
- **Colors**: `red`, `green`, `blue`
- **Objects**: `ball`, `duck`, `cube`, `teddy` (bear)

### Command Examples
- `collect all toys`: Full autonomous cleanup of every visible and hidden toy.
- `collect red ducks`: Targeted search and retrieval for a specific category.
- `collect green balls`: Focus on a single toy color/type combination.
- `collect blue toys`: Retrieve all toys of a specific color.
- `patrol` / `explore`: Initiate the probabilistic search grid without picking up items.
- `auto`: Starts the fully autonomous continuous collection loop.
- `help`: Displays the command guide within the terminal.
- `quit`: Safely shuts down the simulation and disconnects from PyBullet.


## 📐 Implementation Details

- **Inverse Kinematics (IK)**: Real-time IK solver for precise 7-DOF arm positioning based on world-frame targets.
- **Obstacle Avoidance**: Potential field-based avoidance combined with raycast-based local adjustments.
- **Stability Control**: Active vertical stabilization to prevent base tipping during heavy manipulation tasks.

---
*Developed with focus on realistic physical interaction and autonomous mobile manipulation.*
