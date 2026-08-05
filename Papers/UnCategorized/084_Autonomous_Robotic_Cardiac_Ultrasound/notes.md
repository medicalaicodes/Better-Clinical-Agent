# Anatomical Prior-Driven Framework for Autonomous Robotic Cardiac Ultrasound Standard View Acquisition
**Authors:** N/A
**Year:** 2026 | **Venue:** ICRA
**ArXiv/DOI:** https://arxiv.org/abs/2603.21134

# Problem Statement
Acquiring high-quality cardiac ultrasound (Echocardiogram) views requires years of specialized sonographer training, severely limiting access in rural areas.

# Key Contributions
1. Integrates an AI Agent with a robotic ultrasound arm to autonomously scan a patient's chest and acquire standard clinical views (e.g., Apical 4-Chamber).
2. The agent processes the live 2D ultrasound video feed and outputs 6-DOF (Degree of Freedom) movement commands to the robot arm.
3. Uses anatomical priors (3D models of the heart) to guide the robotic arm, rather than blindly searching the chest.

# Methodology / Architecture
A "Vision-to-Action" Agent. The Vision Module (VLM) assesses the quality of the current live ultrasound frame. If the view is suboptimal, the Action Module calculates the geometric transformation needed (e.g., "tilt probe 10 degrees anteriorly") and sends the motor command.

# Datasets & Benchmarks Used
| Dataset/Benchmark | Description | Size/Scale |
|-------------------|-------------|------------|
| Live Phantom Test | Physical torso phantoms | 50 trials |

# Key Results & Metrics
| Task/Benchmark | Metric | This Paper (Agent) | Novice Sonographer | Improvement |
|----------------|--------|--------------------|--------------------|-------------|
| View Acquisition| Time to Standard View| 45 seconds | 120 seconds | 2.6x faster |

# Limitations & Future Work
Tested primarily on healthy phantoms. Patients with obesity or severe lung disease scatter the ultrasound beam, confusing the agent's spatial mapping.

# Relevance to Our Survey
Highly relevant for **Medical Vision** and **Embodied Agents**. Represents the physical embodiment of medical AI agents (Robotics). Modality: Live Video, Motor Control.
