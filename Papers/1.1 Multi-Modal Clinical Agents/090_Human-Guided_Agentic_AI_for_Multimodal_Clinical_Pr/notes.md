# Human-Guided Agentic AI for Multimodal Clinical Prediction: Lessons from the AgentDS Healthcare Benchmark
**Authors:** N/A
**Year:** 2026 | **Venue:** ArXiv
**ArXiv/DOI:** https://arxiv.org/abs/2602.19502

# 1. Unsolved Gaps (Why this method fails in real clinical deployment)
1. **Synthetic Data Dependency**: The methodology was developed and tested on synthetic benchmark data. It lacks the real-world clinical noise, missing values, measurement errors, and data entry inconsistencies found in actual EHR systems, making its real-world robustness unproven.
2. **High Manual Effort (Scalability Issue)**: The proposed human-guided workflow required 10-15 hours of human effort per challenge for feature engineering (parsing PDFs, clinical notes, and time-series data). This extensive manual labor bottlenecks scalability and rapid prototyping in dynamic clinical environments.
3. **Task-Specific Feature Engineering**: There is no universal strategy for multimodal data extraction. For example, TF-IDF worked for long discharge notes but completely failed for short daily progress notes (which required custom clinical keyword matching).
4. **Missing Clinical Trust Features**: The current models lack uncertainty quantification (calibrated probability predictions) which is crucial for clinicians to assess model confidence in high-stakes decisions. Furthermore, there is no fairness analysis or demographic stratification to ensure the models are free of biases regarding race, age, or insurance.

# 2. Benchmark Failures (Where the model falls short quantitatively)
1. **Overall Ranking**: Despite the human guidance, the approach did not secure 1st place in any challenge, ranking 5th (Readmission), 6th (ED Cost), and 3rd (Discharge Readiness), with an overall domain rank of 5th.
2. **Failure of Pure Automation**: The fully autonomous agentic system consistently failed to achieve optimal results on its own. For instance, the baseline automated feature selection degraded performance by an average of -0.014 F1 and +$11 MAE compared to human-curated domain features.
3. **Deep Learning Failure**: Deep learning architectures (LSTMs for vitals, BERT for text) were effectively unusable and failed to perform well because they overfit on the benchmark's small sample sizes (N=1,000-5,000). The team had to fall back to tree-based ensembles (XGBoost, Random Forest).
4. **Modality Dependence**: Relying solely on multimodal data without integrating structured tabular data led to a severe quantitative drop (-0.083 F1 / +$37 MAE).
