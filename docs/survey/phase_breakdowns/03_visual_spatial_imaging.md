# Phase 3: Visual and Spatial Data Ingestion

# Core Concept
Acting as the "eyes" of the agent by encoding pixel and volumetric data into functional embeddings (compressed mathematical representations) that can be processed jointly with text in a single semantic space.

# 1. 2D Medical Imaging
Not all 2D images are treated equally; strategy depends on scale.
*   **Single Global Views (Chest X-rays)**: The agent ingests the entire image at once to analyze overall structure for macro-level anomalies (enlarged heart, fluid).
    *   **SOTA Architecture (MedRAX)**: Medical Reasoning Agent for Chest X-ray. Uses LLM as a planner that orchestrates specialized tools: Segmentation (MedSAM), VQA (CheXagent), Grounding (Maira-2), and Classification (DenseNet-121). Fuses heterogeneous outputs.
*   **Active, Multi-Scale Exploration (Pathology WSIs)**: Gigapixel Whole-Slide Images (e.g., 100,000 x 100,000 pixels) cannot be ingested at once (computationally O(N^2) complexity would crash models).
    *   **Algorithm**: Macro pass (segment tissue, discard glass background) -> Patching (256x256) -> Multiple Instance Learning (MIL) scoring -> Active Zooming on top 1% most suspicious patches.
    *   **SOTA Architecture (SlideSeek & PathChat+)**: Employs hierarchical structure. Supervisor Agent (e.g., OpenAI o1) inspects low-res overview and formulates hypothesis. Explorer Agents (PathChat+, powered by CONCH vision encoder and Qwen2.5 backbone) zoom into specific coordinates to analyze high-res patches.
    *   **SOTA Architecture (CPathAgent)**: Uses Multi-scale Multi-view Sequence Reasoning. Navigates autonomously (panning, zooming at 10x, 40x) and integrates visual features to generate an interpretable diagnostic narrative.

# 2. 3D Medical Imaging
Adds the dimension of depth (CT scans, MRIs).
*   **Token Constraints**: A 3D scan contains hundreds of 2D slices, quickly exceeding token limits. Addressed via Spatial Cropping (Sparse Tokenization) to mask out irrelevant tissues/air.
*   **Cross-Slice Spatial Continuity**: The agent must capture depth mathematically to recognize continuous 3D anatomies (e.g., a blood vessel across slices). SOTA models use 3D Positional Embeddings and 3D Convolutional/Swin Transformer architectures.
*   **SOTA Examples**:
    *   **AgentMRI (Quality Control Agent)**: VLM-powered system for autonomous MRI reconstruction. Uses multi-query consensus to diagnose artifact corruption (motion, noise) and dynamically selects the correct deep learning correction model.
    *   **CTAgent (Modular Orchestration)**: Bypasses token limits by invoking region-specific tools (e.g., Lung Segmenter, then Nodule Classifier on the cropped region) and synthesizes a structured report.

# 3. Real-Time Acoustic & Optical Topography
*   **Ultrasound & Doppler**: High-noise, operator-dependent. SOTA uses Multi-Device Domain Adversarial Training to strip device-specific noise and extract pure anatomical geometry. Doppler uses spatial-temporal 3D Convolutions for continuous velocity signals.
*   **OCT (Optical Coherence Tomography)**: Massive resolution cross-sectional layers. SOTA (EyeMVP, MIRAGE) uses cross-modal masked reconstruction to predict 3D depth directly from 2D Color Fundus Photos.

# 4. Metabolic and Molecular Fusion
*   **PET-CT, SPECT, MRS**: Shows metabolic physiology instead of just anatomy. 
*   **SOTA (SDF-HOLO for PET/CT)**: Systemic Dual-stream Fusion Holo Model. Decouples anatomy (CT) and functional signals (PET), then uses Cross-Modal Interaction to let anatomy refine PET signal aggregation and let metabolic glow guide morphological inspection. Agentic orchestration uses Python tools to calculate SUVs and Metabolic Tumor Volumes.

# 5. Functional 4D Brain Mapping
*   **fMRI / Perfusion**: Tracks BOLD signals over time (4D: 3D space + Time).
*   **SOTA (Meta TRIBE v2)**: Tri-Modal Foundation Model combining vision (V-JEPA), language (LLaMA), and audio (Wav2Vec-BERT) encoders, projected via a Temporal Transformer to predict 4D fMRI activation maps in-silico.

# 6. Specialized Topological and Dense 3D
*   **Angiography**: SOTA uses Graph Neural Networks (GNNs). Treats vessels as "edges" and bifurcations as "nodes" to find network blockages.
*   **Mammography & CBCT**: SOTA uses Physics-Informed Neural Networks (PINNs) to mathematically reverse X-ray scatter artifacts based on tissue density physics, reconstructing high-fidelity images.
