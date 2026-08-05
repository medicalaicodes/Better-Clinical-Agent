# Phase 5: Knowledge Retrieval as Active Ingestion

# Core Concept
Grounding the agent's logic by pulling in verifiable, external medical evidence instead of relying solely on internal model weights. This is essentially Multimodal Retrieval-Augmented Generation (RAG).

# Multimodal Alignment and Retrieval Mechanics
To retrieve images using text (or vice versa), the agent must project two fundamentally different data types (an image matrix $I$ and a text sequence $T$) into a Joint Latent Space.

# 1. Joint Latent Space (The Mathematical Bridge)
*   **Vision Encoder & Text Encoder**: Output high-dimensional vectors ($v_I \in \mathbb{R}^d$ and $v_T \in \mathbb{R}^d$).
*   To understand that an X-ray showing fluid in the lungs means "Pneumonia", $v_I$ and $v_T$ must point in the exact same direction in this $d$-dimensional space.

# 2. Contrastive Learning (The Alignment Algorithm)
*   Models use an objective function minimizing **InfoNCE (Noise Contrastive Estimation)** loss.
*   **Goal**: Maximize similarity of correct image-text pairs while minimizing similarity of mismatched pairs.
*   The contrastive loss mathematically pulls correct pairs together and pushes mismatched pairs apart, guaranteeing visual anomalies and textual descriptions are fused.

# 3. Cosine Similarity (The Retrieval Math)
*   Active ingestion via RAG relies on Cosine Similarity to retrieve evidence from a vector database of historical medical images/reports.
*   **Calculation**: Measures the cosine of the angle $\theta$ between a query vector $q$ and a database vector $d$. Since vectors are $L_2$-normalized, this simplifies into an efficient dot product.

# Algorithmic Implementation (Multimodal RAG)
1.  **Vector Integration**: Every historical medical image and its report is embedded into a massive vector database (e.g., Pinecone, Milvus).
2.  **Cross-Modal Retrieval**: A new patient's X-ray is converted into a query vector to find the top $K$ historical X-rays mathematically matching the lesion pattern using K-Nearest Neighbors (KNN) via Cosine Similarity.
3.  **Evidence-Based Generation**: The agent retrieves the similar images AND their attached historical clinical reports. It synthesizes a diagnosis explicitly referencing this precedent evidence, thereby eliminating "black box" hallucinations.
