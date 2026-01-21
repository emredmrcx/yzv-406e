#!/usr/bin/env python3
"""
Kindergarten Toy Sorting Robot
================================
An intelligent mobile manipulator that:
- Patrols the kindergarten room
- Detects toys using OwlViT AI vision
- Picks up toys and sorts them by color into correct bins
- Responds to natural language commands

Author: Expert Robotics Engineer
Date: January 2026
"""

import numpy as np
import pybullet as p
import pybullet_data
import cv2
import time
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from enum import Enum
import sys
import os
import random


# Try to import AI libraries for OwlViT
try:
    import torch
    from transformers import pipeline
    from PIL import Image
    OWLVIT_AVAILABLE = True
    print("✅ OwlViT available")
except ImportError:
    OWLVIT_AVAILABLE = False
    print("⚠️  OwlViT not available, using color detection")


class RobotState(Enum):
    """Robot state machine"""
    IDLE = 0
    PATROLLING = 1
    DETECTING = 2
    APPROACHING = 3
    PICKING = 4
    TRANSPORTING = 5
    PLACING = 6
    COMPLETED = 7


@dataclass
class Toy:
    """Toy object data"""
    name: str
    color: str
    toy_type: str
    position: np.ndarray
    id: int
    size: float
    collected: bool = False


@dataclass
class Bin:
    """Storage bin data"""
    color: str
    position: np.ndarray
    id: int


class VisionSystem:
    """AI-powered vision system using OwlViT"""
    
    def __init__(self, use_ai: bool = True):
        self.use_ai = use_ai and OWLVIT_AVAILABLE
        
        if self.use_ai:
            print("🔧 Loading OwlViT model...")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            try:
                self.detector = pipeline(
                    "zero-shot-object-detection",
                    model="google/owlvit-base-patch32",
                    device=0 if device == "cuda" else -1
                )
                print(f"✅ OwlViT loaded on {device}")
            except Exception as e:
                print(f"⚠️  Failed to load OwlViT: {e}")
                print("📦 Falling back to color-based detection")
                self.use_ai = False
        
        # Color ranges for fallback
        self.color_ranges = {
            'red': [(0, 120, 70), (10, 255, 255)],
            'green': [(40, 40, 40), (80, 255, 255)],
            'blue': [(100, 50, 50), (130, 255, 255)],
        }
    
    def detect_objects(self, rgb_image: np.ndarray, query: str) -> List[Dict]:
        """
        Detect objects in image
        
        Args:
            rgb_image: RGB image
            query: What to look for (e.g., "toy car", "ball")
        
        Returns:
            List of detections with boxes and scores
        """
        if self.use_ai:
            return self._detect_ai(rgb_image, query)
        else:
            return self._detect_color(rgb_image)
    
    def _detect_ai(self, rgb_image: np.ndarray, query: str) -> List[Dict]:
        """Detect using OwlViT"""
        pil_img = Image.fromarray(rgb_image)
        
        # Define what we're looking for
        queries = [query, f"a {query}", f"toy {query}", "toy", "object"]
        
        predictions = self.detector(pil_img, candidate_labels=queries)
        
        detections = []
        for pred in predictions:
            if pred['score'] > 0.1:  # Confidence threshold
                box = pred['box']
                detections.append({
                    'label': pred['label'],
                    'score': pred['score'],
                    'box': [box['xmin'], box['ymin'], box['xmax'], box['ymax']],
                    'center': [(box['xmin'] + box['xmax']) // 2, 
                              (box['ymin'] + box['ymax']) // 2]
                })
        
        return detections
    
    def _detect_color(self, rgb_image: np.ndarray) -> List[Dict]:
        """Fallback color-based detection"""
        hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
        detections = []
        
        for color, (lower, upper) in self.color_ranges.items():
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for cnt in contours:
                if cv2.contourArea(cnt) > 200:
                    x, y, w, h = cv2.boundingRect(cnt)
                    detections.append({
                        'label': color,
                        'score': 1.0,
                        'box': [x, y, x+w, y+h],
                        'center': [x + w//2, y + h//2]
                    })
        
        return detections


class KindergartenRobot:
    """Mobile manipulator robot for kindergarten"""
    
    def __init__(self, use_gui: bool = True, use_ai: bool = True):
        print("=" * 70)
        print("🏫 KINDERGARTEN TOY SORTING ROBOT")
        print("=" * 70)

        # Known static layout (room limits and bin locations)
        self.room_half = 3.0
        self.bin_blueprint = {
            'half_extents': np.array([0.22, 0.22, 0.02]),
            'wall_height': 0.18
        }
        self.toy_scale = {
            'cube_half': 0.045,  # 9 cm cube
            'ball_radius': 0.04,  # 8 cm diameter ball
            'urdf_scale': 0.65
        }

        # Termination limits (tuned after grid init)
        self.max_search_rounds_without_detection = 6
        self.max_pick_failures = 3
        
        # Initialize PyBullet
        print(f"🔌 Connecting to PyBullet (GUI: {use_gui})...")
        if use_gui:
            self.p_id = p.connect(p.GUI)
        else:
            self.p_id = p.connect(p.DIRECT)
            
        if self.p_id < 0:
            print("❌ Failed to connect to PyBullet")
            sys.exit(1)
            
        print("✅ Connected to PyBullet")
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(1/240)

        
        # Create kindergarten environment
        print("\n🏗️  Building kindergarten room...")
        self._create_kindergarten()
        
        # Create mobile base
        print("🤖 Creating mobile robot...")
        self._create_mobile_base()
        
        # Create robotic arm on mobile base
        print("🦾 Attaching robotic arm...")
        self._attach_arm()
        
        # Vision system
        print("👁️  Initializing vision system...")
        self.vision = VisionSystem(use_ai=use_ai)
        
        # Camera parameters
        self.camera_width = 640
        self.camera_height = 480
        self.fov = 60
        
        # Robot state
        self.state = RobotState.IDLE
        self.current_task = None
        self.held_object = None

        # Toy memory (last seen positions)
        self.last_seen = {}
        self.last_seen_expiry = 25.0

        # Base and bin safety margins (needed before exploration grid)
        self.base_radius = 0.22
        self.bin_keepout_margin = 0.12
        
        # Exploration targets
        self.explore_points = []
        self.visited_points = set()
        self._init_exploration_grid()
        self.current_waypoint = 0

        # Probabilistic search map (heat over explore points)
        self.search_scores = np.zeros(len(self.explore_points))

        # Termination limits based on coverage
        self.max_search_rounds_without_detection = max(30, len(self.explore_points) + 5)

        # Navigation tuning (smoother motion)
        self.nav_step = 0.018
        self.nav_dt = 1 / 120
        self.max_turn_rate = 0.04
        self.avoidance_gain = 0.6
        self._last_desired = np.zeros(2)
        self.scan_dt = 1 / 120
        self.scan_rate = 0.25

        # End-effector top-down orientation (gripper facing downward)
        self.ee_down_orientation = p.getQuaternionFromEuler([np.pi, 0, 0])
        
        print("✅ Robot initialized!\n")
    
    def _create_kindergarten(self):
        """Create kindergarten environment with toys and bins"""
        # Floor
        self.plane_id = p.loadURDF("plane.urdf")
        floor_visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[3.2, 3.2, 0.02],
            rgbaColor=[0.95, 0.9, 0.8, 1]
        )
        floor_collision = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[3.2, 3.2, 0.02]
        )
        p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=floor_collision,
            baseVisualShapeIndex=floor_visual,
            basePosition=[0, 0, -0.02]
        )
        
        # Walls (simple boxes)
        room_half = self.room_half
        wall_height = 1.2
        wall_thickness = 0.1
        wall_color = [0.9, 0.9, 0.7, 1]  # Cream color

        # Right/Left walls (along Y)
        for x in [room_half, -room_half]:
            visual = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=[wall_thickness, room_half, wall_height / 2],
                rgbaColor=wall_color
            )
            collision = p.createCollisionShape(
                p.GEOM_BOX,
                halfExtents=[wall_thickness, room_half, wall_height / 2]
            )
            p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=collision,
                baseVisualShapeIndex=visual,
                basePosition=[x, 0, wall_height / 2]
            )

        # Front/Back walls (along X)
        for y in [room_half, -room_half]:
            visual = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=[room_half, wall_thickness, wall_height / 2],
                rgbaColor=wall_color
            )
            collision = p.createCollisionShape(
                p.GEOM_BOX,
                halfExtents=[room_half, wall_thickness, wall_height / 2]
            )
            p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=collision,
                baseVisualShapeIndex=visual,
                basePosition=[0, y, wall_height / 2]
            )
        
        # Create colored bins (red/green/blue in corners) - open-top
        self.bins = []
        bin_positions = [
            ([2.2, 2.2, 0.15], 'red'),
            ([2.2, -2.2, 0.15], 'green'),
            ([-2.2, 2.2, 0.15], 'blue'),
        ]
        
        bin_colors = {
            'red': [1, 0, 0, 1],
            'green': [0, 1, 0, 1],
            'blue': [0, 0, 1, 1],
        }
        
        for pos, color in bin_positions:
            # Bin floor
            half_ext = self.bin_blueprint['half_extents']
            floor_visual = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=half_ext.tolist(),
                rgbaColor=bin_colors[color]
            )
            floor_collision = p.createCollisionShape(
                p.GEOM_BOX,
                halfExtents=half_ext.tolist()
            )
            bin_floor_id = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=floor_collision,
                baseVisualShapeIndex=floor_visual,
                basePosition=[pos[0], pos[1], 0.02]
            )

            # Bin walls
            wall_half = self.bin_blueprint['half_extents'][0]
            wall_thick = 0.02
            wall_height = self.bin_blueprint['wall_height']
            wall_positions = [
                ([pos[0] + wall_half, pos[1], wall_height / 2], [wall_thick, wall_half, wall_height / 2]),
                ([pos[0] - wall_half, pos[1], wall_height / 2], [wall_thick, wall_half, wall_height / 2]),
                ([pos[0], pos[1] + wall_half, wall_height / 2], [wall_half, wall_thick, wall_height / 2]),
                ([pos[0], pos[1] - wall_half, wall_height / 2], [wall_half, wall_thick, wall_height / 2]),
            ]
            for wpos, half_ext in wall_positions:
                w_visual = p.createVisualShape(
                    p.GEOM_BOX,
                    halfExtents=half_ext,
                    rgbaColor=bin_colors[color]
                )
                w_collision = p.createCollisionShape(
                    p.GEOM_BOX,
                    halfExtents=half_ext
                )
                p.createMultiBody(
                    baseMass=0,
                    baseCollisionShapeIndex=w_collision,
                    baseVisualShapeIndex=w_visual,
                    basePosition=wpos
                )

            self.bins.append(Bin(color=color, position=np.array(pos), id=bin_floor_id))
        
        print(f"  📦 Created {len(self.bins)} colored bins")
        
        # Create toys (red/green/blue)
        self.toys = []
        toy_colors = bin_colors  # Same colors as bins
        toy_configs = [
            # Red toys
            ([0.5, 0.8, 0.04], 'red', 'ball'),
            ([0.7, 0.5, 0.05], 'red', 'duck_vhacd.urdf'),
            ([0.9, 0.2, 0.05], 'red', 'cube'),
            # Green toys
            ([0.3, -0.6, 0.05], 'green', 'teddy_vhacd.urdf'),
            ([0.8, -0.8, 0.05], 'green', 'cube'),
            ([0.2, -0.2, 0.04], 'green', 'ball'),
            # Blue toys
            ([-0.5, 0.7, 0.05], 'blue', 'duck_vhacd.urdf'),
            ([-0.7, 0.4, 0.05], 'blue', 'cube'),
            ([-0.3, 1.1, 0.05], 'blue', 'teddy_vhacd.urdf'),
        ]

        for pos, color, toy_type in toy_configs:
            if toy_type == 'cube':
                half = self.toy_scale['cube_half']
                visual = p.createVisualShape(
                    p.GEOM_BOX,
                    halfExtents=[half, half, half],
                    rgbaColor=toy_colors[color]
                )
                collision = p.createCollisionShape(
                    p.GEOM_BOX,
                    halfExtents=[half, half, half]
                )
                toy_id = p.createMultiBody(
                    baseMass=0.05,
                    baseCollisionShapeIndex=collision,
                    baseVisualShapeIndex=visual,
                    basePosition=pos
                )
                name = f"{color}_cube"
                type_name = "cube"
                toy_size = half * 2
            elif toy_type == 'ball':
                radius = self.toy_scale['ball_radius']
                visual = p.createVisualShape(
                    p.GEOM_SPHERE,
                    radius=radius,
                    rgbaColor=toy_colors[color]
                )
                collision = p.createCollisionShape(
                    p.GEOM_SPHERE,
                    radius=radius
                )
                toy_id = p.createMultiBody(
                    baseMass=0.03,
                    baseCollisionShapeIndex=collision,
                    baseVisualShapeIndex=visual,
                    basePosition=pos
                )
                name = f"{color}_ball"
                type_name = "ball"
                toy_size = radius * 2
            else:
                toy_id = p.loadURDF(toy_type, pos, globalScaling=self.toy_scale['urdf_scale'])
                p.changeVisualShape(toy_id, -1, rgbaColor=toy_colors[color])
                base_name = toy_type.replace('.urdf', '')
                name = f"{color}_{base_name}"
                if 'duck' in base_name:
                    type_name = 'duck'
                elif 'teddy' in base_name:
                    type_name = 'teddy'
                else:
                    type_name = base_name
                toy_size = 0.09

            self.toys.append(
                Toy(name=name, color=color, toy_type=type_name, position=np.array(pos), id=toy_id, size=toy_size)
            )
        
        print(f"  🧸 Created {len(self.toys)} toys")

        # Create obstacles
        self.obstacles = []
        obstacle_specs = [
            ([1.2, 0.0, 0.15], [0.2, 0.4, 0.15]),
            ([-1.0, -0.6, 0.15], [0.25, 0.25, 0.15]),
            ([0.0, -1.4, 0.2], [0.5, 0.15, 0.2]),
            ([1.5, 1.2, 0.2], [0.35, 0.2, 0.2]),
        ]
        for pos, half_extents in obstacle_specs:
            visual = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=half_extents,
                rgbaColor=[0.6, 0.6, 0.6, 1]
            )
            collision = p.createCollisionShape(
                p.GEOM_BOX,
                halfExtents=half_extents
            )
            obs_id = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=collision,
                baseVisualShapeIndex=visual,
                basePosition=pos
            )
            self.obstacles.append((obs_id, np.array(pos), np.array(half_extents)))
        print(f"  🧱 Created {len(self.obstacles)} obstacles")
    
    def _create_mobile_base(self):
        """Create mobile base"""
        self.base_pos = [0, 0, 0.1]
        base_visual = p.createVisualShape(
            p.GEOM_CYLINDER,
            radius=0.22,
            length=0.18,
            rgbaColor=[0.25, 0.25, 0.35, 1]
        )
        base_collision = p.createCollisionShape(
            p.GEOM_CYLINDER,
            radius=0.22,
            height=0.18
        )
        
        self.robot_id = p.createMultiBody(baseMass=10,
                                         baseCollisionShapeIndex=base_collision,
                                         baseVisualShapeIndex=base_visual,
                                         basePosition=self.base_pos)

        p.changeDynamics(self.robot_id, -1,
                         lateralFriction=1.0,
                         rollingFriction=0.01,
                         spinningFriction=0.01,
                         linearDamping=0.1,
                         angularDamping=0.1,
                         activationState=p.ACTIVATION_STATE_DISABLE_SLEEPING)

    
    def _attach_arm(self):
        """Attach Panda robotic arm with stability improvements"""
        try:
            # Load Panda arm on top of mobile base
            robot_pos, robot_orn = p.getBasePositionAndOrientation(self.robot_id)
            arm_pos = [robot_pos[0], robot_pos[1], robot_pos[2] + 0.15]
            self.arm_offset = [0, 0, 0.15]
            self.arm_id = p.loadURDF(
                "franka_panda/panda.urdf",
                arm_pos,
                useFixedBase=False,
                flags=p.URDF_USE_SELF_COLLISION
            )
            
            # Create strong constraint between base and arm
            self.arm_base_constraint = p.createConstraint(
                self.robot_id, -1,
                self.arm_id, -1,
                p.JOINT_FIXED,
                [0, 0, 0],
                [0, 0, 0.15],
                [0, 0, 0]
            )
            # High max force prevents arm from wobbling
            p.changeConstraint(self.arm_base_constraint, maxForce=10000)
            
            # Get joint info
            self.num_joints = p.getNumJoints(self.arm_id)
            
            # Find gripper joints and arm joints
            self.arm_joints = list(range(7))  # First 7 joints are arm
            self.gripper_joints = []
            for i in range(self.num_joints):
                joint_info = p.getJointInfo(self.arm_id, i)
                joint_name = joint_info[1].decode('utf-8')
                if 'finger' in joint_name:
                    self.gripper_joints.append(i)
            
            # Set arm to home position with damping
            home_joints = [0, -0.785, 0, -2.356, 0, 1.571, 0.785]
            for i in range(7):
                p.resetJointState(self.arm_id, i, home_joints[i])
                # Add velocity damping to reduce jitter
                p.setJointMotorControl2(
                    self.arm_id, i,
                    p.VELOCITY_CONTROL,
                    targetVelocity=0,
                    force=50  # Damping force
                )
            
            # Get end-effector link
            self.ee_link_index = 11  # Panda hand link
            
            self.gripper_constraint = None
            
            print(f"  ✅ Panda arm attached ({self.num_joints} joints)")
            print(f"  ✅ Gripper joints: {len(self.gripper_joints)}")
            
        except Exception as e:
            print(f"  ⚠️  Could not load Panda: {e}")
            print(f"  ⚠️  Using simplified gripper")
            self.arm_id = None
            self.gripper_constraint = None
            self.arm_offset = [0, 0, 0.15]

    def _sync_arm_base(self):
        """Keep arm base aligned with the mobile base"""
        if not self.arm_id or self.arm_id < 0:
            return
        try:
            base_pos, base_orn = p.getBasePositionAndOrientation(self.robot_id)
            arm_pos = [base_pos[0] + self.arm_offset[0],
                       base_pos[1] + self.arm_offset[1],
                       base_pos[2] + self.arm_offset[2]]
            p.resetBasePositionAndOrientation(self.arm_id, arm_pos, base_orn)
        except:
            pass
    
    def get_robot_pose(self) -> Tuple[float, float, float]:
        """Get robot position and orientation"""
        pos, orn = p.getBasePositionAndOrientation(self.robot_id)
        euler = p.getEulerFromQuaternion(orn)
        return pos[0], pos[1], euler[2]
    
    def get_camera_image(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get camera view from robot"""
        x, y, theta = self.get_robot_pose()
        
        # Camera on robot
        cam_height = 0.4
        cam_pos = [x + 0.2 * np.cos(theta), y + 0.2 * np.sin(theta), cam_height]
        target_pos = [
            cam_pos[0] + np.cos(theta),
            cam_pos[1] + np.sin(theta),
            cam_height - 0.1
        ]
        
        view_matrix = p.computeViewMatrix(cam_pos, target_pos, [0, 0, 1])
        proj_matrix = p.computeProjectionMatrixFOV(
            self.fov, self.camera_width/self.camera_height, 0.1, 10
        )
        
        _, _, rgb, depth, _ = p.getCameraImage(
            self.camera_width, self.camera_height,
            view_matrix, proj_matrix
        )
        
        rgb_array = np.array(rgb).reshape(self.camera_height, self.camera_width, 4)[:, :, :3]
        depth_array = np.array(depth).reshape(self.camera_height, self.camera_width)
        
        return rgb_array, depth_array

    def _is_point_clear(self, point: np.ndarray) -> bool:
        """Check if a point is not inside an obstacle or wall buffer"""
        margin = 0.30
        if abs(point[0]) > self.room_half - margin or abs(point[1]) > self.room_half - margin:
            return False

        for _, obs_pos, half_extents in self.obstacles:
            dx = abs(point[0] - obs_pos[0])
            dy = abs(point[1] - obs_pos[1])
            if dx < (half_extents[0] + margin) and dy < (half_extents[1] + margin):
                return False

        # Bin keep-out zones to prevent base entering bins
        bin_half = float(self.bin_blueprint['half_extents'][0])
        keepout = bin_half + self.base_radius + self.bin_keepout_margin
        for bin_obj in self.bins:
            dx = abs(point[0] - bin_obj.position[0])
            dy = abs(point[1] - bin_obj.position[1])
            if dx < keepout and dy < keepout:
                return False
        return True

    def _init_exploration_grid(self):
        """Initialize exploration grid points across the room"""
        spacing = 0.7
        xs = np.arange(-self.room_half + spacing, self.room_half - spacing + 1e-3, spacing)
        ys = np.arange(-self.room_half + spacing, self.room_half - spacing + 1e-3, spacing)

        # Sweep pattern (snake) to cover the room systematically
        for i, y in enumerate(ys):
            row = []
            for x in xs:
                pt = np.array([x, y])
                if self._is_point_clear(pt):
                    row.append([x, y])
            if i % 2 == 1:
                row.reverse()
            self.explore_points.extend(row)

    def _choose_explore_target(self) -> Optional[List[float]]:
        """Choose the next exploration target using probabilistic scores"""
        if not self.explore_points:
            return None

        # Prefer highest-score unvisited points
        candidates = [i for i, p in enumerate(self.explore_points) if tuple(p) not in self.visited_points]
        if not candidates:
            return None

        # Probabilistic choice (softmax over scores)
        scores = np.array([self.search_scores[i] for i in candidates], dtype=float)
        scores = scores - np.max(scores)
        weights = np.exp(scores)
        weights = weights / (np.sum(weights) + 1e-9)
        chosen = int(np.random.choice(len(candidates), p=weights))
        return self.explore_points[candidates[chosen]]

    def _compute_pick_base_pose(self, toy_xy: np.ndarray) -> Optional[Tuple[float, float, float]]:
        """Compute a safe base pose to grasp a toy from top-down."""
        angles = np.linspace(0, 2 * np.pi, 12, endpoint=False)
        for preferred_dist in [0.14, 0.16, 0.18, 0.20]:
            for ang in angles:
                offset = np.array([np.cos(ang), np.sin(ang)]) * preferred_dist
                candidate = np.array(toy_xy) - offset
                if not self._is_point_clear(candidate):
                    continue
                yaw = np.arctan2(toy_xy[1] - candidate[1], toy_xy[0] - candidate[0])
                return float(candidate[0]), float(candidate[1]), float(yaw)
        return None

    def _compute_avoidance_vector(self, robot_pos: np.ndarray) -> np.ndarray:
        """Compute repulsive vector from nearby obstacles"""
        avoid = np.zeros(2)
        for _, obs_pos, half_extents in self.obstacles:
            vec = robot_pos[:2] - obs_pos[:2]
            dist = np.linalg.norm(vec)
            safe_radius = max(half_extents[0], half_extents[1]) + 0.5
            if dist < safe_radius and dist > 1e-3:
                strength = (safe_radius - dist) / safe_radius
                avoid += (vec / dist) * strength

        # Bin avoidance
        bin_half = float(self.bin_blueprint['half_extents'][0])
        bin_safe = bin_half + self.base_radius + 0.25
        for bin_obj in self.bins:
            vec = robot_pos[:2] - bin_obj.position[:2]
            dist = np.linalg.norm(vec)
            if dist < bin_safe and dist > 1e-3:
                strength = (bin_safe - dist) / bin_safe
                avoid += (vec / dist) * strength * 1.2

        # Lidar-like ray checks (local obstacle avoidance)
        ray_len = 0.8
        ray_origin = [robot_pos[0], robot_pos[1], 0.2]
        ray_dirs = []
        for i in range(12):
            ang = i * (2 * np.pi / 12)
            ray_dirs.append([np.cos(ang), np.sin(ang)])
        ray_from = [ray_origin] * len(ray_dirs)
        ray_to = [[ray_origin[0] + d[0] * ray_len,
                   ray_origin[1] + d[1] * ray_len,
                   ray_origin[2]] for d in ray_dirs]
        results = p.rayTestBatch(ray_from, ray_to)
        
        toy_ids = [t.id for t in self.toys]
        for d, hit in zip(ray_dirs, results):
            hit_id = hit[0]
            hit_fraction = hit[2]
            # Ignore floor, walls (if they have no ID), self, and toys
            if hit_id >= 0 and hit_id != self.robot_id and hit_id != self.arm_id and hit_id not in toy_ids:
                if 0.0 < hit_fraction < 1.0:
                    dist = hit_fraction * ray_len
                    strength = max(0.0, (ray_len - dist) / ray_len)
                    avoid -= np.array(d) * strength * 1.5


        # Wall avoidance
        wall_margin = self.room_half - 0.4
        if abs(robot_pos[0]) > wall_margin:
            avoid[0] += -np.sign(robot_pos[0]) * 0.8
        if abs(robot_pos[1]) > wall_margin:
            avoid[1] += -np.sign(robot_pos[1]) * 0.8
        return avoid

    def _approach_point(self, target_xy: np.ndarray, standoff: float = 0.35, timeout: float = 12.0) -> bool:
        """Move near a target with a safety standoff to reduce collisions"""
        robot_x, robot_y, _ = self.get_robot_pose()
        dx, dy = target_xy[0] - robot_x, target_xy[1] - robot_y
        dist = np.linalg.norm([dx, dy])
        if dist < 1e-3:
            return True

        scale = max(dist - standoff, 0.0) / dist
        approach = np.array([robot_x, robot_y]) + np.array([dx, dy]) * scale
        if not self._is_point_clear(approach):
            # Try small angular offsets to find a safe approach point
            for ang in np.linspace(-np.pi / 3, np.pi / 3, 7):
                rot = np.array([
                    np.cos(ang) * dx - np.sin(ang) * dy,
                    np.sin(ang) * dx + np.cos(ang) * dy
                ])
                rot_dist = np.linalg.norm(rot)
                if rot_dist < 1e-3:
                    continue
                scale = max(rot_dist - standoff, 0.0) / rot_dist
                candidate = np.array([robot_x, robot_y]) + rot * scale
                if self._is_point_clear(candidate):
                    approach = candidate
                    break
        return self.move_to(float(approach[0]), float(approach[1]), timeout=timeout, aggressive=True)

    def _apply_base_velocity(self, linear_vel: np.ndarray, angular_vel: float):
        """Apply velocity to mobile base using physics"""
        # Get current orientation
        _, orn = p.getBasePositionAndOrientation(self.robot_id)
        euler = p.getEulerFromQuaternion(orn)
        yaw = euler[2]
        
        # Transform velocity to world frame
        world_vx = linear_vel[0] * np.cos(yaw) - linear_vel[1] * np.sin(yaw)
        world_vy = linear_vel[0] * np.sin(yaw) + linear_vel[1] * np.cos(yaw)
        
        # Apply velocity
        p.resetBaseVelocity(
            self.robot_id,
            linearVelocity=[world_vx, world_vy, 0],
            angularVelocity=[0, 0, angular_vel]
        )
    
    def _navigate_to_grasp_position(self, target_xy: np.ndarray, timeout: float = 10.0, desired_dist: float = 0.30, tolerance: float = 0.10) -> bool:
        """Move to a position from which the toy is reachable and face it."""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            rx, ry, ryaw = self.get_robot_pose()
            pos = np.array([rx, ry])
            to_toy = target_xy - pos
            dist = np.linalg.norm(to_toy)
            
            toy_yaw = np.arctan2(to_toy[1], to_toy[0])
            yaw_err = np.arctan2(np.sin(toy_yaw - ryaw), np.cos(toy_yaw - ryaw))
            
            # Dynamic acceptance range based on arguments
            if (desired_dist - tolerance) < dist < (desired_dist + tolerance) and abs(yaw_err) < 0.45:
                p.resetBaseVelocity(self.robot_id, [0, 0, 0], [0, 0, 0])
                for _ in range(30): 
                    self._sync_arm_base()
                    p.stepSimulation()
                return True

            
            # Simple control: Rotate and move
            ang_vel = np.clip(yaw_err * 3.0, -1.2, 1.2)
            
            # Move towards a point desired_dist away from toy
            if dist > 0.01:
                dir_unit = to_toy / dist
                # Goal is to be at desired_dist from toy
                goal_pos = target_xy - dir_unit * desired_dist
                to_goal = goal_pos - pos
                dist_to_goal = np.linalg.norm(to_goal)
                
                lin_speed = min(0.3, dist_to_goal * 2.0)
                if dist_to_goal < 0.05: lin_speed = 0
                
                move_dir = to_goal / (dist_to_goal + 1e-6)
                p.resetBaseVelocity(
                    self.robot_id,
                    linearVelocity=[move_dir[0] * lin_speed, move_dir[1] * lin_speed, 0],
                    angularVelocity=[0, 0, ang_vel * 1.5]
                )

            
            self._sync_arm_base()
            p.stepSimulation()
            time.sleep(1/240)
            
        return False


    
    def move_to(self, target_x: float, target_y: float, timeout: float = 10.0, aggressive: bool = False):
        """Navigate to target position with velocity-based physics control"""
        # Clamp targets to safe space if needed
        if not self._is_point_clear(np.array([target_x, target_y])):
            rx, ry, _ = self.get_robot_pose()
            vec = np.array([rx, ry]) - np.array([target_x, target_y])
            dist = np.linalg.norm(vec)
            if dist < 1e-3:
                vec = np.array([1.0, 0.0])
                dist = 1.0
            dir_unit = vec / dist
            candidate = np.array([target_x, target_y]) + dir_unit * 0.35
            if self._is_point_clear(candidate):
                target_x, target_y = float(candidate[0]), float(candidate[1])
            else:
                # Try small offsets
                for ang in np.linspace(0, 2 * np.pi, 12, endpoint=False):
                    candidate = np.array([target_x, target_y]) + np.array([np.cos(ang), np.sin(ang)]) * 0.35
                    if self._is_point_clear(candidate):
                        target_x, target_y = float(candidate[0]), float(candidate[1])
                        break

        start_time = time.time()
        max_linear_speed = 0.5  # m/s
        max_angular_speed = 1.5  # rad/s
        
        while time.time() - start_time < timeout:
            x, y, theta = self.get_robot_pose()
            pos = np.array([x, y, 0.0])

            dx = target_x - x
            dy = target_y - y
            distance = np.sqrt(dx**2 + dy**2)

            if distance < 0.12:
                # Stop and settle
                p.resetBaseVelocity(self.robot_id, [0, 0, 0], [0, 0, 0])
                for _ in range(30):
                    self._sync_arm_base()
                    p.stepSimulation()
                    time.sleep(1/240)
                return True

            # Obstacle avoidance
            avoidance = self._compute_avoidance_vector(pos)
            avoid_gain = 0.3 if aggressive or distance < 0.6 else self.avoidance_gain
            desired = np.array([dx, dy]) + avoidance * avoid_gain
            if np.linalg.norm(desired) < 1e-3:
                desired = np.array([dx, dy])

            # Smooth desired direction
            desired = 0.6 * self._last_desired + 0.4 * desired
            self._last_desired = desired

            # Compute velocity direction and magnitude
            if np.linalg.norm(desired) > 1e-3:
                move_dir = desired / np.linalg.norm(desired)
            else:
                move_dir = np.array([dx, dy]) / (distance + 1e-6)
            
            # Speed proportional to distance (slow down as we approach)
            speed = min(max_linear_speed, distance * 1.5)
            if distance < 0.3:
                speed = min(speed, distance * 0.8)
            
            # Desired heading
            target_angle = np.arctan2(desired[1], desired[0])
            angle_diff = np.arctan2(np.sin(target_angle - theta), np.cos(target_angle - theta))
            
            # Angular velocity proportional to angle error
            angular_vel = np.clip(angle_diff * 3.0, -max_angular_speed, max_angular_speed)
            
            # Linear velocity in world frame
            linear_vel = move_dir * speed
            
            # Check if next position would be safe
            next_pos = np.array([x, y]) + linear_vel * (1/60)
            if not self._is_point_clear(next_pos):
                # Slow down and try to avoid
                speed *= 0.3
                linear_vel = move_dir * speed
            
            # Apply velocities using physics
            # Extra stability: Zero out roll/pitch velocity to keep base perfectly level
            curr_lin_vel, _ = p.getBaseVelocity(self.robot_id)
            p.resetBaseVelocity(
                self.robot_id,
                linearVelocity=[linear_vel[0], linear_vel[1], curr_lin_vel[2]],
                angularVelocity=[0, 0, angular_vel]
            )
            
            self._sync_arm_base()
            p.stepSimulation()
            time.sleep(1/240)


        # Timeout - stop robot
        p.resetBaseVelocity(self.robot_id, [0, 0, 0], [0, 0, 0])
        return False

    def _compute_safe_bin_approach(self, bin_xy: np.ndarray) -> Optional[np.ndarray]:
        """Compute a safe approach point outside the bin keep-out zone."""
        bin_half = float(self.bin_blueprint['half_extents'][0])
        standoff = bin_half + self.base_radius + self.bin_keepout_margin + 0.05
        base_x, base_y, _ = self.get_robot_pose()
        vec = np.array([base_x, base_y]) - np.array(bin_xy)
        dist = np.linalg.norm(vec)
        if dist < 1e-3:
            vec = np.array([1.0, 0.0])
            dist = 1.0
        dir_unit = vec / dist
        candidate = np.array(bin_xy) + dir_unit * standoff
        if self._is_point_clear(candidate):
            return candidate

        # Try cardinal directions if direct line is blocked
        directions = [np.array([1.0, 0.0]), np.array([-1.0, 0.0]), np.array([0.0, 1.0]), np.array([0.0, -1.0])]
        for d in directions:
            candidate = np.array(bin_xy) + d * standoff
            if self._is_point_clear(candidate):
                return candidate
        return None

    def patrol(self):
        """Patrol the room"""
        print("🚶 Exploring...")
        target = self._choose_explore_target()
        if target is None:
            # All visited, restart sweep
            self.visited_points.clear()
            self.current_waypoint = 0
            target = self._choose_explore_target()

        if target and self.move_to(target[0], target[1], timeout=14.0):
            self.visited_points.add(tuple(target))
            # Decay score for visited target
            idx = self.explore_points.index(target)
            self.search_scores[idx] *= 0.2
            print(f"  ✅ Reached point {target}")
        elif target:
            # If navigation failed, do a small recovery step and keep sweep going
            rx, ry, _ = self.get_robot_pose()
            nudge = np.array([rx, ry]) + np.random.uniform(-0.25, 0.25, size=2)
            if self._is_point_clear(nudge):
                self.move_to(float(nudge[0]), float(nudge[1]), timeout=6.0, aggressive=True)

    def _approach_last_seen(self, color_filter: Optional[str], type_filter: Optional[str]) -> bool:
        """Approach a remembered target location before continuing sweep."""
        now = time.time()
        candidates = []
        for toy in self.toys:
            if toy.collected:
                continue
            if color_filter and toy.color != color_filter:
                continue
            if type_filter and toy.toy_type != type_filter:
                continue
            mem = self.last_seen.get(toy.id)
            if not mem:
                continue
            if now - mem["time"] > self.last_seen_expiry:
                continue
            candidates.append((toy, mem["pos"]))

        if not candidates:
            return False

        # Prefer nearest remembered toy
        rx, ry, _ = self.get_robot_pose()
        candidates.sort(key=lambda t: (t[1][0] - rx) ** 2 + (t[1][1] - ry) ** 2)
        toy, pos = candidates[0]
        print(f"🧭 Going to last seen {toy.name} at {pos[:2]}")
        return self._approach_point(np.array(pos[:2]), standoff=0.25, timeout=12.0)
    
    def _scan_in_place(self, steps: int = 120):
        """Rotate in place to scan the environment"""
        x, y, theta = self.get_robot_pose()
        dtheta = self.scan_rate * self.scan_dt
        for _ in range(steps):
            theta += dtheta
            p.resetBasePositionAndOrientation(
                self.robot_id,
                [x, y, 0.1],
                p.getQuaternionFromEuler([0, 0, theta])
            )
            self._sync_arm_base()
            p.stepSimulation()
            time.sleep(self.scan_dt)

    def search_for_toys(self, color_filter: Optional[str] = None, type_filter: Optional[str] = None) -> List[Toy]:
        """Search for toys in current view"""
        target_desc = color_filter or "any"
        if type_filter:
            target_desc = f"{target_desc} {type_filter}"
        print(f"🔍 Searching for {target_desc} toys...")
        
        rgb, _ = self.get_camera_image()
        
        # Detect objects
        detections = self.vision.detect_objects(rgb, "toy")
        use_geometry_fallback = len(detections) == 0
        
        found_toys = []
        seen_ids = set()
        if not use_geometry_fallback:
            for _ in detections:
                # Match with known toys
                for toy in self.toys:
                    if toy.collected:
                        continue
                    if color_filter and toy.color != color_filter:
                        continue
                    if type_filter and toy.toy_type != type_filter:
                        continue
                    
                    # Check if toy is in view (simplified)
                    toy_pos, _ = p.getBasePositionAndOrientation(toy.id)
                    robot_x, robot_y, robot_theta = self.get_robot_pose()
                    
                    # If toy is roughly in front of robot
                    dx = toy_pos[0] - robot_x
                    dy = toy_pos[1] - robot_y
                    angle_to_toy = np.arctan2(dy, dx)
                    angle_diff = abs(angle_to_toy - robot_theta)
                    
                    if angle_diff < np.pi/3 and np.sqrt(dx**2 + dy**2) < 2.0:
                        if toy.id not in seen_ids:
                            seen_ids.add(toy.id)
                            found_toys.append(toy)
                            self.last_seen[toy.id] = {
                                "pos": np.array(toy_pos),
                                "time": time.time(),
                                "name": toy.name,
                            }
                            # Boost nearby search scores
                            for i, pt in enumerate(self.explore_points):
                                dist = np.linalg.norm(np.array(pt) - np.array(toy_pos[:2]))
                                if dist < 1.2:
                                    self.search_scores[i] += max(0.0, 1.2 - dist)
                            print(f"  ✅ Found {toy.name} at {toy_pos[:2]}")
        else:
            # Fallback: geometric visibility without detections
            robot_x, robot_y, robot_theta = self.get_robot_pose()
            for toy in self.toys:
                if toy.collected:
                    continue
                if color_filter and toy.color != color_filter:
                    continue
                if type_filter and toy.toy_type != type_filter:
                    continue

                toy_pos, _ = p.getBasePositionAndOrientation(toy.id)
                dx = toy_pos[0] - robot_x
                dy = toy_pos[1] - robot_y
                dist = np.sqrt(dx**2 + dy**2)
                angle_to_toy = np.arctan2(dy, dx)
                angle_diff = abs(np.arctan2(np.sin(angle_to_toy - robot_theta), np.cos(angle_to_toy - robot_theta)))

                if angle_diff < np.pi/2 and dist < 2.5:
                    if toy.id not in seen_ids:
                        seen_ids.add(toy.id)
                        found_toys.append(toy)
                        self.last_seen[toy.id] = {
                            "pos": np.array(toy_pos),
                            "time": time.time(),
                            "name": toy.name,
                        }
                        for i, pt in enumerate(self.explore_points):
                            dist = np.linalg.norm(np.array(pt) - np.array(toy_pos[:2]))
                            if dist < 1.2:
                                self.search_scores[i] += max(0.0, 1.2 - dist)
                        print(f"  ✅ Found {toy.name} at {toy_pos[:2]} (fallback)")
        
        return found_toys

    def collect_requested(self, color_filter: Optional[str], type_filter: Optional[str], max_steps: int = 1500):
        """Search and collect requested toys"""
        print(f"🎯 Task: Collect {color_filter or 'all'} {type_filter or 'toys'}\n")
        steps = 0
        rounds_without_detection = 0
        pick_failures = 0
        while steps < max_steps:
            steps += 1

            if self.held_object is None:
                # Scan and search
                self._scan_in_place(steps=80)
                toys = self.search_for_toys(color_filter, type_filter)

                if toys:
                    rounds_without_detection = 0
                    # Pick the nearest visible toy
                    rx, ry, _ = self.get_robot_pose()
                    toys.sort(key=lambda t: (p.getBasePositionAndOrientation(t.id)[0][0] - rx) ** 2 +
                                            (p.getBasePositionAndOrientation(t.id)[0][1] - ry) ** 2)
                    target_toy = toys[0]
                    if self.pick_toy(target_toy):
                        self.place_in_bin(target_toy.color)
                        pick_failures = 0
                    else:
                        pick_failures += 1
                        if not self._approach_last_seen(color_filter, type_filter):
                            self.patrol()
                else:
                    rounds_without_detection += 1
                    # Go to last seen location before sweeping
                    if not self._approach_last_seen(color_filter, type_filter):
                        self.patrol()
            else:
                rounds_without_detection = 0
                self.place_in_bin(self.held_object.color)

            if rounds_without_detection >= self.max_search_rounds_without_detection:
                print("\n⏹️  Arama sınırı aşıldı, durduruluyor.")
                return

            if pick_failures >= self.max_pick_failures:
                print("\n⏹️  Çoklu kavrama hatası, durduruluyor.")
                return

            # Finish if all requested toys collected
            done = True
            for toy in self.toys:
                if toy.collected:
                    continue
                if color_filter and toy.color != color_filter:
                    continue
                if type_filter and toy.toy_type != type_filter:
                    continue
                done = False
                break

            if done:
                print("\n🎉 Requested toys collected!")
                return

        print("\n⚠️  Süre doldu, hedef oyuncaklar tam toplanamadı.")
    
    def move_arm_to(self, target_pos: np.ndarray, steps: int = 100, target_orn: Optional[Tuple[float, float, float, float]] = None):
        """Move arm end-effector to target position using IK"""
        if not self.arm_id:
            return False

        base_pos, base_orn = p.getBasePositionAndOrientation(self.robot_id)
        if target_orn is None:
            target_orn = self.ee_down_orientation
        joint_damping = [0.05] * self.num_joints
        
        for _ in range(steps):
            # Keep base stable during arm motion
            p.resetBasePositionAndOrientation(self.robot_id, base_pos, base_orn)
            self._sync_arm_base()
            p.resetBaseVelocity(self.robot_id, [0, 0, 0], [0, 0, 0])
            # Get current robot base position
            base_pos, base_orn = p.getBasePositionAndOrientation(self.robot_id)
            
            # Inverse kinematics
            joint_poses = p.calculateInverseKinematics(
                self.arm_id,
                self.ee_link_index,
                target_pos,
                targetOrientation=target_orn,
                jointDamping=joint_damping,
                maxNumIterations=120,
                residualThreshold=0.01
            )
            
            # Set joint positions
            for i in range(min(7, len(joint_poses))):
                p.setJointMotorControl2(
                    self.arm_id,
                    i,
                    p.POSITION_CONTROL,
                    targetPosition=joint_poses[i],
                    force=300,
                    positionGain=0.05,
                    velocityGain=0.9
                )
            
            p.stepSimulation()
            time.sleep(1/240)
        
        return True
    
    def open_gripper(self):
        """Open gripper"""
        if not self.arm_id or not self.gripper_joints:
            return
        
        for joint in self.gripper_joints:
            p.setJointMotorControl2(
                self.arm_id,
                joint,
                p.POSITION_CONTROL,
                targetPosition=0.04,  # Open
                force=20
            )
        
        for _ in range(50):
            p.stepSimulation()
            time.sleep(1/240)
    
    def close_gripper(self):
        """Close gripper"""
        if not self.arm_id or not self.gripper_joints:
            return
        
        for joint in self.gripper_joints:
            p.setJointMotorControl2(
                self.arm_id,
                joint,
                p.POSITION_CONTROL,
                targetPosition=0.0,  # Closed
                force=20
            )
        
        for _ in range(50):
            p.stepSimulation()
            time.sleep(1/240)
    
    def pick_toy(self, toy: Toy) -> bool:
        """Pick up a toy using physics-based navigation"""
        print(f"🤏 Picking up {toy.name}...")
        
        # Get current toy position
        toy_pos, _ = p.getBasePositionAndOrientation(toy.id)
        # Navigate to grasp position
        # Navigate to grasp position
        print(f"  → Navigating to grasp position...")
        # Initial approach: looser tolerance
        if not self._navigate_to_grasp_position(np.array(toy_pos[:2]), timeout=12.0, desired_dist=0.35, tolerance=0.15):
            print(f"  ❌ Could not approach {toy.name}")
            return False

        
        if self.arm_id:
            # Refresh toy position after stabilization
            toy_pos, _ = p.getBasePositionAndOrientation(toy.id)
            
            # Use real arm
            print(f"  → Opening gripper...")
            self.open_gripper()
            
            # Move arm above toy
            print(f"  → Moving arm to toy...")
            self._sync_arm_base()
            above_pos = np.array([toy_pos[0], toy_pos[1], toy_pos[2] + 0.20])
            self.move_arm_to(above_pos, steps=100, target_orn=self.ee_down_orientation)
            
            # Descend to toy
            print(f"  → Descending...")
            grasp_pos = np.array([toy_pos[0], toy_pos[1], toy_pos[2] + 0.03])
            self.move_arm_to(grasp_pos, steps=90, target_orn=self.ee_down_orientation)
            
            # Check proximity
            ee_state = p.getLinkState(self.arm_id, self.ee_link_index)
            ee_pos = np.array(ee_state[0])
            toy_pos_current = np.array(p.getBasePositionAndOrientation(toy.id)[0])
            grasp_dist = np.linalg.norm(ee_pos - toy_pos_current)

            if grasp_dist > 0.20:
                # Try to get closer by moving base
                print(f"  → Adjusting for better reach (grasp dist: {grasp_dist:.2f}m)...")
                toy_xy = np.array(toy_pos[:2])
                # STRICTER approach on retry: aim for 0.28m with tightly bound
                self._navigate_to_grasp_position(toy_xy, timeout=5.0, desired_dist=0.28, tolerance=0.08)
                p.resetBaseVelocity(self.robot_id, [0, 0, 0], [0, 0, 0])
                for _ in range(40):
                    self._sync_arm_base()
                    p.stepSimulation()
                    time.sleep(1/240)
                
                # Retry arm motion
                toy_pos, _ = p.getBasePositionAndOrientation(toy.id)
                above_pos = np.array([toy_pos[0], toy_pos[1], toy_pos[2] + 0.18])
                grasp_pos = np.array([toy_pos[0], toy_pos[1], toy_pos[2] + 0.02])
                self.move_arm_to(above_pos, steps=80, target_orn=self.ee_down_orientation)
                self.move_arm_to(grasp_pos, steps=80, target_orn=self.ee_down_orientation)
                
                ee_state = p.getLinkState(self.arm_id, self.ee_link_index)
                ee_pos = np.array(ee_state[0])
                toy_pos_current = np.array(p.getBasePositionAndOrientation(toy.id)[0])
                grasp_dist = np.linalg.norm(ee_pos - toy_pos_current)
                
                if grasp_dist > 0.20:
                    print(f"  ❌ Grasp failed (too far: {grasp_dist:.2f}m)")
                    return False

            # Close gripper
            print(f"  → Closing gripper...")
            self.close_gripper()

            # Create constraint for stable grasp
            self.gripper_constraint = p.createConstraint(
                self.arm_id, self.ee_link_index,
                toy.id, -1,
                p.JOINT_FIXED,
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0]
            )

            if self.gripper_constraint < 0:
                print("  ❌ Grasp failed (constraint)")
                return False
            
            # Lift
            print(f"  → Lifting...")
            lift_pos = np.array([toy_pos[0], toy_pos[1], toy_pos[2] + 0.30])
            self.move_arm_to(lift_pos, steps=80, target_orn=self.ee_down_orientation)
            
        else:
            # Fallback: simple constraint
            robot_pos, _ = p.getBasePositionAndOrientation(self.robot_id)
            self.gripper_constraint = p.createConstraint(
                self.robot_id, -1, toy.id, -1,
                p.JOINT_FIXED,
                [0, 0, 0],
                [0, 0, 0.2],
                [0, 0, 0]
            )
        
        self.held_object = toy
        print(f"  ✅ Picked up {toy.name}")
        return True
    
    def place_in_bin(self, bin_color: str) -> bool:
        """Place held object in bin"""
        if not self.held_object:
            print("❌ No object in gripper")
            return False
        
        # Find bin
        target_bin = None
        for bin_obj in self.bins:
            if bin_obj.color == bin_color:
                target_bin = bin_obj
                break
        
        if not target_bin:
            print(f"❌ No {bin_color} bin found")
            return False
        
        print(f"📦 Placing {self.held_object.name} in {bin_color} bin...")
        
        # Move base near bin with a safe standoff (never enter bin keep-out)
        bin_xy = target_bin.position[:2]
        approach_point = self._compute_safe_bin_approach(bin_xy)
        
        start_dist = np.linalg.norm(np.array(self.get_robot_pose()[:2]) - np.array(bin_xy))
        
        if approach_point is not None:
             # If we are already reasonably close to a good spot, don't force a full move
            move_success = self.move_to(float(approach_point[0]), float(approach_point[1]), timeout=12.0, aggressive=False)
            
            if not move_success:
                 # Check if we are "close enough" regardless of move_to result
                 # "Close enough" means we are within reaching distance ~0.7m depending on bin size
                 # but NOT inside the keepout zone.
                 curr_pos = np.array(self.get_robot_pose()[:2])
                 dist_to_bin = np.linalg.norm(curr_pos - bin_xy)
                 
                 # Bin reach check: Arm can reach ~0.8m. Bin center might be far.
                 # If we are within 0.8m of bin center and > keepout
                 bin_half = float(self.bin_blueprint['half_extents'][0])
                 keepout = bin_half + self.base_radius + self.bin_keepout_margin
                 
                 if keepout < dist_to_bin < 0.90:
                     print(f"  ⚠️ Exact approach failed, but within reach ({dist_to_bin:.2f}m). Proceeding...")
                 else:
                     print(f"  ❌ Could not reach bin vicinity (dist: {dist_to_bin:.2f}m, keepout: {keepout:.2f}m)")
                     return False
        else:
             print("  ❌ Could not find safe bin approach")
             return False

        # Face the bin for a clean top-down drop
        base_x, base_y, base_yaw = self.get_robot_pose()
        base_xy = np.array([base_x, base_y])
        dir_vec = np.array(bin_xy) - base_xy
        face_yaw = np.arctan2(dir_vec[1], dir_vec[0])
        
        # Rotate to face bin
        print(f"  → Rotating to face {bin_color} bin...")
        start_rot = time.time()
        while time.time() - start_rot < 4.0:
            _, _, curr_yaw = self.get_robot_pose()
            yaw_error = np.arctan2(np.sin(face_yaw - curr_yaw), np.cos(face_yaw - curr_yaw))
            if abs(yaw_error) < 0.05:
                break
            angular_vel = np.clip(yaw_error * 3.0, -1.0, 1.0)
            p.resetBaseVelocity(self.robot_id, [0, 0, 0], [0, 0, angular_vel])
            self._sync_arm_base()
            p.stepSimulation()
            time.sleep(1/240)
        p.resetBaseVelocity(self.robot_id, [0, 0, 0], [0, 0, 0])

        
        # Proceed with place using the arm
        if True:
            if self.arm_id:
                # Move arm over bin
                print(f"  → Moving arm over bin...")
                for _ in range(60):
                    self._sync_arm_base()
                    p.stepSimulation()
                    time.sleep(1/240)
                over_bin = target_bin.position.copy()
                over_bin[2] += 0.38
                self.move_arm_to(over_bin, steps=90, target_orn=self.ee_down_orientation)

                # Descend for top-down placement
                drop_pos = target_bin.position.copy()
                drop_pos[2] += 0.18
                self.move_arm_to(drop_pos, steps=80, target_orn=self.ee_down_orientation)
                
                # Open gripper to release
                print(f"  → Releasing...")
                
                # Remove constraint first
                if self.gripper_constraint:
                    p.removeConstraint(self.gripper_constraint)
                    self.gripper_constraint = None
                
                self.open_gripper()
                
                # Retract arm
                print(f"  → Retracting arm...")
                retract_pos = over_bin.copy()
                retract_pos[2] += 0.12
                self.move_arm_to(retract_pos, steps=70, target_orn=self.ee_down_orientation)
                
            else:
                # Simple release
                if self.gripper_constraint:
                    p.removeConstraint(self.gripper_constraint)
                    self.gripper_constraint = None

            # Object is released. It will fall into bin by gravity. 
            # We don't teleport the toy anymore, just let it fall.
            self.held_object.collected = True
            
            self.held_object = None
            print(f"  ✅ Placed in {bin_color} bin")
            return True

        
        return False
    
    def execute_command(self, command: str):
        """Execute natural language command"""
        print(f"\n{'='*70}")
        print(f"💬 Command: '{command}'")
        print(f"{'='*70}\n")
        
        command = command.lower()
        
        # Parse command
        color_filter = None
        if 'red' in command:
            color_filter = 'red'
        elif 'green' in command:
            color_filter = 'green'
        elif 'blue' in command:
            color_filter = 'blue'

        type_filter = None
        if 'ball' in command:
            type_filter = 'ball'
        elif 'duck' in command:
            type_filter = 'duck'
        elif 'teddy' in command or 'bear' in command:
            type_filter = 'teddy'
        elif 'cube' in command:
            type_filter = 'cube'
        
        if 'help' in command or 'commands' in command:
            print("Available commands (examples):")
            print("  - collect red toys")
            print("  - collect green toys")
            print("  - collect blue toys")
            print("  - collect all toys")
            print("  - collect red balls")
            print("  - collect blue ducks")
            print("  - collect green cubes")
            print("  - patrol")
            print("  - auto")
            print("  - quit")
            return

        if 'patrol' in command or 'explore' in command or 'search' in command:
            self.patrol()
            return

        if 'auto' in command or 'autonomous' in command:
            self.run_autonomous()
            return
        
        if 'collect' in command or 'find' in command or 'get' in command or 'pick' in command:
            self.collect_requested(color_filter, type_filter)
            return

        # Fallback: any unknown command triggers autonomous search
        print("🤖 Command not recognized, switching to autonomous search...")
        self.run_autonomous()
    
    def run_interactive(self):
        """Interactive mode"""
        print("\n💬 INTERACTIVE MODE")
        print("Example commands (English only):")
        print("  - 'collect red toys'")
        print("  - 'collect green toys'")
        print("  - 'collect blue toys'")
        print("  - 'collect all toys'")
        print("  - 'collect red balls'")
        print("  - 'collect blue ducks'")
        print("  - 'collect green cubes'")
        print("  - 'patrol'")
        print("  - 'auto' (continuous search)")
        print("  - 'help' (show command examples)")
        print("  - 'quit' to exit\n")
        
        while True:
            try:
                cmd = input("🎤 Command: ").strip()
                
                if cmd.lower() in ['quit', 'exit', 'q']:
                    break
                
                if not cmd:
                    continue
                
                self.execute_command(cmd)
                
            except KeyboardInterrupt:
                break
    
    def close(self):
        """Clean up"""
        try:
            if p.isConnected():
                p.resetSimulation()
                p.disconnect()
        except Exception:
            pass
        print("\n👋 Robot shutdown")

    def run_autonomous(self, max_steps: int = 2000):
        """Autonomous search and collect loop"""
        print("\n🤖 AUTONOMOUS MODE")
        print("(Press Ctrl+C to stop)\n")
        steps = 0
        rounds_without_detection = 0
        pick_failures = 0
        while steps < max_steps:
            steps += 1

            if self.held_object is None:
                # Scan and search
                self._scan_in_place(steps=90)
                toys = self.search_for_toys()
                if toys:
                    rounds_without_detection = 0
                    # Pick the nearest visible toy
                    rx, ry, _ = self.get_robot_pose()
                    toys.sort(key=lambda t: (p.getBasePositionAndOrientation(t.id)[0][0] - rx) ** 2 +
                                            (p.getBasePositionAndOrientation(t.id)[0][1] - ry) ** 2)
                    target_toy = toys[0]
                    if self.pick_toy(target_toy):
                        self.place_in_bin(target_toy.color)
                        pick_failures = 0
                    else:
                        pick_failures += 1
                        if not self._approach_last_seen(None, None):
                            self.patrol()
                else:
                    rounds_without_detection += 1
                    # Approach last seen target before sweeping
                    if not self._approach_last_seen(None, None):
                        self.patrol()
            else:
                rounds_without_detection = 0
                # If holding, go to bin
                self.place_in_bin(self.held_object.color)

            if rounds_without_detection >= self.max_search_rounds_without_detection:
                print("\n⏹️  Search limit reached, stopping autonomous mode.")
                break

            if pick_failures >= self.max_pick_failures:
                print("\n⏹️  Multiple grasp failures, stopping autonomous mode.")
                break

            # End if all collected
            if all(t.collected for t in self.toys):
                print("\n🎉 All toys collected!")
                break


def main():
    """Main entry point"""
    try:
        # Check for AI support
        use_ai = OWLVIT_AVAILABLE
        
        if not use_ai:
            print("\n⚠️  AI detection not available")
            print("To enable: pip install transformers torch")
            print("Continuing with color-based detection...\n")
            time.sleep(2)
        
        headless_env = os.environ.get("HEADLESS", "").lower() in ["1", "true", "yes"]
        headless_arg = any(arg.lower() in ["headless", "nogui", "--headless", "--nogui"] for arg in sys.argv[1:])
        use_gui = not (headless_env or headless_arg)

        # Create robot
        robot = KindergartenRobot(use_gui=use_gui, use_ai=use_ai)

        # Default to autonomous; add "interactive" arg for commands
        if any("interactive" in arg.lower() for arg in sys.argv[1:]):
            robot.run_interactive()
        else:
            robot.run_autonomous()
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'robot' in locals():
            robot.close()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
