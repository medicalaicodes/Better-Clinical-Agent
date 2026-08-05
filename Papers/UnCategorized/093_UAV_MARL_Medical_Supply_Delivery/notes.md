# UAV-MARL: Multi-Agent Reinforcement Learning for Time-Critical and Dynamic Medical Supply Delivery
**Authors:** N/A
**Year:** 2026 | **Venue:** ArXiv
**ArXiv/DOI:** https://arxiv.org/abs/2603.10528

# Problem Statement
Delivering time-critical medical supplies (blood, antivenom, organs) via a swarm of drones requires real-time routing adaptation to weather, battery levels, and changing emergency priorities.

# Key Contributions
1. Applies Multi-Agent Reinforcement Learning (MARL) to physical drone swarms for medical logistics.
2. Drones communicate their local states (wind, battery, payload) to nearby drones to dynamically re-route or hand off payloads if an emergency triage priority shifts mid-flight.
3. Outperforms static traveling-salesman routing algorithms by 40% in dynamic weather scenarios.

# Methodology / Architecture
Decentralized MARL (MAPPO algorithm). Each drone is an independent agent. They optimize a joint reward function that heavily penalizes late delivery of high-triage items (e.g., O-negative blood) while balancing swarm battery life.

# Datasets & Benchmarks Used
| Dataset/Benchmark | Description | Size/Scale |
|-------------------|-------------|------------|
| Med-Drone-Sim | Urban logistics simulation | 10k flight scenarios |

# Key Results & Metrics
| Task/Benchmark | Metric | This Paper (MARL) | Static Routing | Improvement |
|----------------|--------|-------------------|----------------|-------------|
| Critical Delivery | % On-Time (Dynamic env) | 94.2% | 65.8% | +28.4% |

# Limitations & Future Work
Simulated environments do not perfectly capture the complex aerodynamics of physical drone-to-drone payload handoffs.

# Relevance to Our Survey
Relevant for **Embodied Agents / Operational**. Highlights how multi-agent logic translates into physical hospital logistics and public health supply chains. Modality: Operational/Logistics Data.
