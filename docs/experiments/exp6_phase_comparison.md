# **EEG RAG System – Technical Analysis Report (Phases 1–4)**

**Project:** EEG Retrieval-Augmented Generation (RAG)
**Date:** October 20, 2025

## **1. Executive Summary**

This report presents the architectural evolution of the EEG RAG system through four development phases.
Each phase targeted two recurring challenges:

1. The system’s inability to handle complex reasoning queries, and
2. A persistent hallucination error in the GPD definition.

The transition from the early **two-source dense baseline (Phase 1)** to the **fully reranked hybrid configuration (Phase 4)** successfully resolved the GPD hallucination and established stable reasoning capability.
However, a consistent regression appeared in definition retrieval. While reasoning and factual precision improved, the system increasingly failed to return basic medical term definitions such as *LPD* and *BIRDs*—tasks that the initial version handled correctly.

In short, the later phases enhanced conceptual reasoning but compromised retrieval accuracy for short, definition-based queries.

---

## **2. Architectural Evolution**

### **Phase 1 → Phase 2: Introducing Reasoning Capability**

| Aspect                   | **Phase 1 (Baseline)**                                                                                                                   | **Phase 2 (Three-Source RAG)**                        |
| :----------------------- | :--------------------------------------------------------------------------------------------------------------------------------------- | :---------------------------------------------------- |
| **Data Sources**         | EEG + PDF Definitions                                                                                                                    | EEG + PDF + Video Reasoning                           |
| **Core Change**          | Two-source dense retrieval                                                                                                               | Added video transcripts and query-type classification |
| **Rationale**            | The baseline system failed completely on reasoning-type questions. Video transcripts were introduced to provide conceptual explanations. |                                                       |
| **Impact**               | Reasoning improved significantly, but definition retrieval degraded due to vector noise from video content.                              |                                                       |
| **GPD Result**           | Incorrect: “Generalized Spike Discharge”                                                                                                 | Incorrect: “Generalized Paroxysmal Discharge”         |
| **LPD/BIRDs Definition** | Correct                                                                                                                                  | Lost; retrieval confused by mixed embeddings          |

**Summary:**
Reasoning capability was achieved, but adding a third data source introduced semantic interference. The vector space became noisier, and the system began retrieving descriptive rather than definitional content.

---

### **Phase 2 → Phase 3: Implementing Hybrid Search (RRF)**

| Aspect                   | **Phase 2 (Dense Search)**                                                                                                                       | **Phase 3 (Hybrid RAG)**                                                |
| :----------------------- | :----------------------------------------------------------------------------------------------------------------------------------------------- | :---------------------------------------------------------------------- |
| **Core Change**          | Dense-only retrieval                                                                                                                             | Combined Dense and BM25 Sparse Search with Reciprocal Rank Fusion (RRF) |
| **Rationale**            | Dense retrieval alone failed on short, keyword-based queries. The hybrid method aimed to recover exact matches while preserving semantic recall. |                                                                         |
| **Impact**               | Recall improved but precision dropped. RRF often ranked contextually weak chunks above correct definitions.                                      |                                                                         |
| **GPD Result**           | “Generalized Paroxysmal Discharge”                                                                                                               | “Generalized Prognostic Value” (incorrect due to term co-occurrence)    |
| **LPD/BIRDs Definition** | Still incorrect                                                                                                                                  | Still incorrect                                                         |

**Summary:**
Hybrid search increased recall but decreased precision. The GPD hallucination worsened because BM25 boosted irrelevant fragments that happened to contain the same terms.

---

### **Phase 3 → Phase 4: Adding Cross-Encoder Reranking**

| Aspect                   | **Phase 3 (Hybrid RRF)**                                                                                                                                                                     | **Phase 4 (Full Hybrid + Reranker)**                  |
| :----------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :---------------------------------------------------- |
| **Core Change**          | Dense + BM25 retrieval                                                                                                                                                                       | Added BGE Cross-Encoder reranker for post-RRF ranking |
| **Rationale**            | RRF alone could not reliably rank correct results at the top. The reranker was added to rescore top candidates based on contextual relevance.                                                |                                                       |
| **Impact**               | The reranker eliminated the GPD hallucination by promoting the correct PDF definition. However, missing LPD/BIRDs definitions persisted, confirming a retrieval rather than ranking problem. |                                                       |
| **GPD Result**           | “Generalized Prognostic Value”                                                                                                                                                               | Correct: “Generalized Periodic Discharges”            |
| **LPD/BIRDs Definition** | Failed                                                                                                                                                                                       | Failed                                                |

**Summary:**
The reranking layer fixed factual precision issues and stabilized output quality. Nevertheless, the model still lacked access to concise definitional text, as the correct content never reached the reranker.

---

## **3. Comparative Summary**

| Feature              | **Phase 1**            | **Phase 2**                 | **Phase 3**                   | **Phase 4**                     |
| :------------------- | :--------------------- | :-------------------------- | :---------------------------- | :------------------------------ |
| Complex Reasoning    | Failed                 | Successful                  | Successful                    | Successful                      |
| GPD Definition       | Hallucinated (“Spike”) | Hallucinated (“Paroxysmal”) | Worsened (“Prognostic Value”) | Correct (“Periodic Discharges”) |
| LPD/BIRDs Definition | Correct                | Regression                  | Regression                    | Regression                      |
| Retrieval Stability  | High                   | Medium                      | Low                           | High                            |
| Overall Precision    | Moderate               | Decreased                   | Low                           | High                            |

**Key Observations:**

* Reasoning capability introduced in Phase 2 remained stable across all later versions.
* The GPD hallucination was fully resolved in Phase 4.
* The regression on simple definitions persisted through every stage after Phase 1.

---

## **4. Root Cause**

The main cause of the regression was **retrieval contamination**.
Once video and EEG data were embedded in the same vector index, short definitional text from PDFs became overshadowed by lengthy descriptive content containing repeated mentions of key terms.

Even with BM25 balancing and cross-encoder reranking, the search pool remained polluted. The correct definitions were rarely surfaced among top results, leaving the reranker unable to recover them.

---

## **5. Corrective Strategy**

### **Immediate Fix: Source Gating**

* When handling definition-type queries, retrieval should exclude video and EEG sources.
* Restrict these queries to the `medical_definitions` collection only.
* This will restore clean retrieval of concise, authoritative text fragments.

### **Long-Term Measures**

* Implement adaptive weighting for each source based on query classification.
* Add retrieval diagnostics to monitor source contamination in real time.
* Evaluate hybrid scoring thresholds to better balance precision and recall for short queries.

---

## **6. Conclusion**

Phase 4 represents the most capable and balanced configuration of the EEG RAG system to date.
It provides accurate reasoning and consistent factual precision while eliminating the major hallucination issues that affected earlier versions.

However, the persistent definition regression confirms that the limitation now lies within the **retrieval layer**, not in the model’s reasoning.
Applying strict source gating and retrieval diagnostics should close this gap, enabling the system to achieve both high-level reasoning and accurate definitional grounding.

---
