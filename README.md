# Stretch Simulation Workflow

Version 2 files are too large to upload here, therefore they are uploaded to this [link](https://drive.google.com/drive/folders/1CzBpMjR5p07C_zGG9bKV9HuhQYNIuIMT?usp=sharing). Below explains the workflow for the version 2. Readme file for version 1 can be found in version_1 folder.

## Setup
1. Clone the repository.
2. Follow the Docker setup instructions: [README_DOCKER.md](https://github.com/hello-robot/stretch_ros2/blob/humble/stretch_simulation/README_DOCKER.md)

## Environment & Features
On top of the base simulation, we have built:
-   **Kindergarten Environment:** 2-room layout with toy shelf and drop-off zones.
-   **Navigation Stack:** Uses Nav2 for path planning and localization.
-   **LLM Interaction Services:** Built under `src/stretch_ros2/stretch_simulation_interfaces`.

### Capabilities (ROS Services)
The system exposes high-level services for LLM agents to control the robot:

1.  **traverse_to_location** (`/publish_goal_pose`)
    *   **Description:** Autonomously traverses the robot to designated locations.
    *   **Locations:** `toy_box`, `delivery_location1`, `delivery_location2` (3 play areas).
    *   **Behavior:** Publishes a goal pose to Nav2.

2.  **pick_object** (`/attach_object`)
    *   **Description:** Fakes the grasp logic.
    *   **Behavior:**
        *   Lowers arm to object's exact Z-position (keeping X/Y fixed).
        *   Attaches object physics to gripper.
        *   **Post-Grasp:** Lifts object by 0.3m and retracts arm to body.

3.  **place_object** (`/detach_object`)
    *   **Description:** Releases the object from the gripper.

## Execution

### 1. Start Simulation & Services
Run the simulation which includes the custom kindergarten scene and services:
```bash
ros2 launch stretch_simulation stretch_mujoco_driver.launch.py \
  use_mujoco_viewer:=true \
  mode:=navigation \
  use_cameras:=true \
  use_robocasa:=false \
  broadcast_odom_tf:=True
```

### 2. Start Navigation
Run the navigation stack:
```bash
ros2 launch stretch_nav2 navigation.launch.py \
  use_slam:=True \
  params_file:=/home/emre/ament_ws/src/stretch_ros2/stretch_nav2/config/nav2_params_slam.yaml \
  use_sim_time:=true
```

### 3. Run LLM Agent
Run any LLM agent to control the robot, just make it read this README.
