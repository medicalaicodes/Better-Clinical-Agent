# MAM: Modular Multi-Agent Framework for Multi-Modal Medical Diagnosis via Role-Specialized Collaboration
**Authors:** N/A
**Year:** 2025 | **Venue:** ACL Findings
**URL:** https://aclanthology.org/2025.findings-acl.1298/

# 1. Unsolved Gaps (Why this method fails in real clinical deployment)
1. **Lack of Real-World Grounding**: The framework was tested purely on public benchmark datasets. In clinical reality, the "Medical Assistant" agent queries a standard Google API to fetch data; it cannot currently integrate with real, secure hospital databases or Electronic Health Records (EHR) systems due to privacy and architecture gaps.
2. **Cascading Hallucinations and Bias**: The system is fundamentally bottlenecked by the capabilities of the underlying foundation models. Any reasoning inaccuracies, knowledge gaps, or biases in the baseline LLMs can propagate through the inter-agent discussion, compromising the final output.
3. **Collaboration Overhead & Diminishing Returns**: Increasing the number of expert roles beyond 3 actually degraded performance across datasets, highlighting that too many agents introduce redundant noise and hinder consensus. Extending the debate to 3+ rounds often diluted accurate initial diagnoses.
4. **Poor Information Retrieval**: The retrieval agent struggles significantly to formulate effective search queries for complex medical cases. In several datasets, the framework failed to retrieve the necessary medical context.

# 2. Benchmark Failures (Where the model falls short quantitatively)
1. **Low Absolute Accuracy on Complex Visuals**: The model performs poorly on difficult multimodal tasks. Accuracy on DeepLesion is only 35.1%, PMC-VQA is 32.5%, and PathVQA is 47.6%. Text accuracy on MedQA is similarly low at 40.0%.
2. **Agent Consensus Overriding Correct Baselines**: The framework actively overrides correct baseline predictions with incorrect ones (Heartbeat drops from 70.2% to 64.0% consistency, SoundDr drops from 60.0% to 47.9%).
3. **Retrieval Module Failures**: NIH Dataset achieved a catastrophic recall rate of just 12.1%. Even when the correct context was retrieved, the downstream "Answer Correct" rate was only 46.2%. DeepLesion retrieval recall was just 31.7%.
