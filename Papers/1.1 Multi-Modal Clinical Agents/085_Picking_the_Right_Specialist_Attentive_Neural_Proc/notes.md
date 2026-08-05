# Picking the Right Specialist: Attentive Neural Process-based Selection of Task-Specialized Models
**Authors:** N/A
**Year:** 2026 | **Venue:** ArXiv
**ArXiv/DOI:** https://arxiv.org/abs/2602.14901

# 1. Unsolved Gaps (Why this method fails in real clinical deployment)
*   **Reliance on a Frozen Model Zoo:** ToolSelect treats all underlying candidate models as static, black-box specialists. In a real clinical deployment, patient demographics, imaging hardware, and institutional reporting styles drift over time. Since the method cannot fine-tune or adapt the underlying models, if the entire zoo lacks the capability to handle an out-of-distribution clinical scenario, ToolSelect will fail by default.
*   **Imperfect Clinical Routing:** The Attentive Neural Process (ANP) router uses behavioral summaries and query text to guess the best model, but it lacks deep, intrinsic medical reasoning. It cannot "read" the image with the same depth as the specialists, leading to misrouting where a sub-optimal tool is selected simply because its summary heuristically matched the query.
*   **Lack of Autonomous Safety:** The authors explicitly state in the Impact Statement that the system is not capable of autonomous clinical decision-making. The routing errors mean human-in-the-loop oversight is strictly required, preventing fully automated deployment.

# 2. Benchmark Failures (Where the model falls short quantitatively)
The paper defines an "Oracle" bound—the performance if the system perfectly selected the best possible tool from the zoo for every single query. ToolSelect consistently fails to reach this bound, demonstrating quantitative routing failures:
*   **Visual Question Answering (VQA):** This is the most glaring benchmark failure. The Oracle achieves **96.49%** accuracy, but ToolSelect only achieves **72.01%**. This massive ~24.5% absolute performance gap indicates that for open-ended medical questions, the selector frequently misjudges which VLM to trust.
*   **Disease Diagnosis:** On the Open-I dataset, ToolSelect achieves **43.80% F1**, which severely lags behind the Oracle's **58.00% F1**. On the VinDr dataset, ToolSelect achieves **59.88% F1** compared to the Oracle's **66.22% F1**.
*   **Visual Grounding:** ToolSelect achieves a Mean IoU of **50.08**, falling short of the Oracle's **56.20**.
*   **Report Generation:** ToolSelect achieves **24.17%** F1-RadGraph, while the Oracle attains **27.70%**.
