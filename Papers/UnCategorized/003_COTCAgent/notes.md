# COTCAgent: Preventive Consultation via Probabilistic Chain-of-Thought Completion
**Authors:** Michael Zhang, Emily Chen et al.
**Year:** 2026 | **Venue:** ArXiv
**ArXiv/DOI:** https://arxiv.org/abs/2605.15016

# Problem Statement
Current medical dialogue agents are reactive; they answer questions but fail to proactively anticipate patient needs or guide the consultation toward a differential diagnosis, often leading to missed critical information.

# Key Contributions
1. Proposes COTCAgent, which uses Probabilistic Chain-of-Thought (CoT) Completion to proactively ask preventive and clarifying questions.
2. Models the dialogue as a Partially Observable Markov Decision Process (POMDP).
3. Reduces diagnostic errors in simulated patient-doctor interactions by 18%.

# Methodology / Architecture
The agent runs a background CoT process that generates multiple hypothetical diagnostic paths based on current symptoms. It calculates the information gain for potential follow-up questions. Instead of just answering the user, the agent selects the question with the highest expected information gain to steer the conversation.

# Datasets & Benchmarks Used
| Dataset/Benchmark | Description | Size/Scale |
|-------------------|-------------|------------|
| SymCAT | Symptom-disease dataset | 800 diseases |
| MedDialog | Medical conversations | 3.4M dialogues |

# Key Results & Metrics
| Task/Benchmark | Metric | This Paper | Previous SOTA | Improvement |
|----------------|--------|------------|---------------|-------------|
| Diagnostic Acc | Top-3 Accuracy | 76.5% | 68.2% | +8.3% |
| Turn Count | Avg Turns to Dx | 5.2 | 7.8 | -2.6 turns (Faster) |

# Limitations & Future Work
The probabilistic completion can sometimes be overly aggressive, asking too many questions before providing reassurance. Needs better empathy modeling.

# Relevance to Our Survey
Important for **Clinical Decision Support** and **Patient-Facing Copilots**. Demonstrates proactive reasoning in conversational agents. Modality: Clinical Text.
