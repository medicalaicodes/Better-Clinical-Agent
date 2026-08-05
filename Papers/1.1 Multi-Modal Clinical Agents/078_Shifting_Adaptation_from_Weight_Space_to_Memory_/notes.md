# Shifting Adaptation from Weight Space to Memory Space: A Memory-Augmented Agent for Medical Image Segmentation
**Authors:** N/A
**Year:** 2026 | **Venue:** ArXiv
**ArXiv/DOI:** https://arxiv.org/abs/2603.05873

# 1. Problem Statement
Medical image segmentation models suffer from poor generalization across domains (institutions, scanners, patients) due to data heterogeneity. While foundation models help, they typically require task-specific fine-tuning (e.g., PEFT). This necessitates significant communication overhead in federated learning (FL) and impedes continuous, dynamic knowledge updates during clinical deployment.

# 2. Key Contributions
- **MemSeg-Agent:** Unifies few-shot learning, federated supervised learning, and test-time adaptation without modifying the model's weights.
- **Static Memory via Supervision:** Projects parameterized pseudo-observations into SAM2's memory space to capture task/site-specific priors compactly (only 2M parameters).
- **Test-Time Working Memory:** Online, feedback-driven adaptation that updates via a gated delta rule. 
- **Communication-Efficient FL:** Transmitting memories rather than weights drops per-round communication by ~98.65% (~74.3x reduction).

# 3. Methodology
- **Frozen SAM2 Backbone:** Uses SAM2's native memory conditioning without fine-tuning.
- **Static Memory:** Optimized using Dice and BCE loss on pseudo-observations to learn a stable prior.
- **Few-Shot Memory:** Non-parametric encoding of a small support set retrieved via similarity.
- **Test-Time Working Memory:** When predictions are poor, human-in-the-loop corrections are mapped into working memory. DINOv3 similarity and Dice scores gate the exponential moving average (EMA) updates to prevent noise accumulation.
- **Agentic Controller:** Orchestrates retrieval, first checking static memory, then augmenting with few-shot or test-time working memories depending on domain shifts.

# 4. Datasets & Benchmarks
- **CHAOS:** Abdominal MRI (Kidneys, Spleen, Liver).
- **ACDC:** Cardiac MRI (RV, Myo, LV).
- **CAMUS:** 2D Echocardiography / Ultrasound (LV, Myo, LA).
- **CardiacUDA:** Cross-domain external benchmark (tested on CAMUS-trained models).

# 5. Key Results
- **Cross-Domain:** Working memory boosts performance dramatically under domain shift (CAMUS to CardiacUDA), hitting 77.30% Avg Dice vs MedSAM2 (42.75%) and UNet (10.15%).
- **In-Domain Efficiency:** Static memory alone matches fully supervised networks (UNet, nnU-Net, SwinUNETR) while being highly parameter-efficient.
- **FL Costs:** Communication overhead slashed drastically without losing segmentation accuracy.

# 6. Unsolved Gaps & Benchmark Failures (Hyper-Critical)
- **Modality Failure on Ultrasound:** The paper explicitly fails to achieve state-of-the-art results on the CAMUS benchmark. Ultrasound boundaries are ambiguous, and the SAM2 backbone relies too heavily on stark edge-aware features. Without explicit semantic DINO-like fine-tuning, the model performs poorly here.
- **Human-in-the-Loop Dependency:** The highly-touted "test-time working memory" requires an oracle or human clinician to provide corrected annotations ($y_t^*$) when the prediction fails. This entirely undermines the prospect of a fully autonomous clinical agent.
- **Redundant Aggregation in In-Domain Scenarios:** The ablation study reveals that retrieving more working memories (larger $k$) degrades in-domain performance. The agent lacks a robust filtering mechanism for redundant or noisy memory accumulation.
- **2D Slice-Wise Limitation:** Medical imaging (MRI/CT) is inherently 3D. The method operates purely in a 2D slice-wise manner, discarding crucial Z-axis spatial dependencies. 
- **Reliance on External Embeddings (DINOv3):** DINOv3 is used to measure memory similarity and route the controller. DINOv3 is trained on natural images and is sub-optimal for mapping domain distances in noisy medical modalities (e.g., echocardiograms).

# 7. Modalities & Key Models
- **Modalities:** Abdominal MRI, Cardiac MRI, 2D Echocardiography (Ultrasound).
- **Key Models:** SAM2 (frozen backbone), DINOv3 (similarity router), MedSAM2, UNet, nnU-Net, SwinUNETR.
