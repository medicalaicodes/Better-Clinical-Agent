# MedEyes: Learning Dynamic Visual Focus for Medical Progressive Diagnosis
**Authors:** N/A
**Year:** 2025 | **Venue:** ArXiv
**ArXiv/DOI:** https://arxiv.org/abs/2511.22018

# 1. Unsolved Gaps (Why this method fails in real clinical deployment)
1. **Quantitative Measurement Errors**: MedEyes fails at quantitative measurement tasks (e.g., tumor sizing) because it relies on relative anatomical proportion estimation rather than actual pixel-to-centimeter calibration, producing variance incompatible with clinical standards.
2. **Fine-Grained Concept Differentiation**: It struggles with fine-grained concept differentiation, frequently conflating pathological subtypes that share morphological features (e.g., confusing 'aneurysmal changes' with 'dissecting aneurysms').

# 2. Benchmark Failures (Where the model falls short quantitatively)
1. **Complex Benchmark Ceilings**: Quantitatively, MedEyes underperforms on complex datasets, achieving only 55.3% on PMC-VQA and 59.7% on MMMU (Health & Medicine). 
2. **Ablation Vulnerabilities**: Ablation studies show significant quantitative failures when components are isolated: removing the Gaze-guided Reasoning Navigator (GRN) drops performance by 8.7%, and removing off-policy learning causes a 10.5% degradation. 
3. **Accumulation of Errors**: Configuring the model with an off-policy trajectory sequence longer than 3 steps causes errors to accumulate, leading to a decrease in overall accuracy.
