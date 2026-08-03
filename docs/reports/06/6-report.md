
---

# **Technical Report: Comparative Analysis of the EEG RAG System (Phases 1–4)**

**Date:** October 22, 2025
**Subject:** In-depth review of architectural transitions, performance evolution, and regression points in the Retrieval-Augmented Generation (RAG) pipeline.

---

## **Executive Summary**

The EEG RAG system has undergone four major architectural phases, each aimed at solving two persistent problems:
(1) the system’s inability to handle *complex reasoning queries*, and
(2) a recurring *GPD hallucination error* in its responses.

The transition from the early **2-source dense model (Phase 1)** to the **full reranked hybrid RAG (Phase 4)** successfully eliminated the GPD hallucination and introduced robust reasoning capability.
However, a notable regression persists — the latest model still fails to retrieve **simple term definitions** such as *LPD* and *BIRDs*, which the original baseline handled correctly.

---

## **1. Phase-by-Phase Architectural Evolution and Impact**

The following sections summarize the key design decisions, rationale, and direct effects of each architectural change on performance across major query types.

---

### **Phase 1 → Phase 2: Introducing Reasoning Capability**

| Detail                     | **Phase 1 (Baseline)**                                                                                                                                                                                                                                                                                               | **Phase 2 (3-Source RAG)**                                                                                                 |
| -------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| **Architectural Change**   | 2 Data Sources (`EEG` + `PDF Definitions`)                                                                                                                                                                                                                                                                           | 3 Data Sources (`EEG` + `PDF` + `Video Reasoning`) + Query Type Detection                                                  |
| **Rationale**              | The baseline system relied solely on short PDF definitions, which caused total failure on reasoning-based queries (e.g., “Why are breaks important?”). To fix this, expert-driven video transcripts were added, and `query_type` classification was introduced to route complex questions to the appropriate source. |                                                                                                                            |
| **Key Impact Summary**     |                                                                                                                                                                                                                                                                                                                      |                                                                                                                            |
| **GPD (Hallucination)**    | *Incorrect* — “Generalized Spike Discharge.”                                                                                                                                                                                                                                                                         | *Incorrect* — “Generalized Paroxysmal Discharge.” The hallucination persists.                                              |
| **Reasoning Queries**      | *Failed completely* (“Context does not provide…”).                                                                                                                                                                                                                                                                   | *Success* — correctly explained “periodic vs rhythmic” using the video’s reasoning context. Reasoning capability achieved. |
| **LPD/BIRDs (Definition)** | *Success* — correct definition retrieved from PDF.                                                                                                                                                                                                                                                                   | **Regression** — “Context does not provide…” Failure caused by retrieval noise introduced by the added video/EEG data.     |

**Summary:**
The system gained reasoning ability but lost reliability in simple term retrieval. Video content improved conceptual understanding but polluted the vector space with non-definitional noise.

---

### **Phase 2 → Phase 3: Implementing Hybrid Search (RRF)**

| Detail                     | **Phase 2 (3-Source Dense)**                                                                                                                                                                                                                    | **Phase 3 (Hybrid RAG)**                                                                                                                       |
| -------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| **Architectural Change**   | Pure Dense (Vector-only) Search                                                                                                                                                                                                                 | Hybrid Search (Dense + BM25 Sparse) with Reciprocal Rank Fusion (RRF)                                                                          |
| **Rationale**              | Phase 2’s regression on definitions was a classic failure of dense-only search: poor recall for short, keyword-heavy queries. BM25 was introduced to recover exact matches from the PDF, with RRF balancing semantic and keyword-based results. |                                                                                                                                                |
| **Key Impact Summary**     |                                                                                                                                                                                                                                                 |                                                                                                                                                |
| **GPD (Hallucination)**    | “Generalized Paroxysmal Discharge.”                                                                                                                                                                                                             | **Worsened** — “Generalized Prognostic Value.” The term “Prognostic Value” appeared due to BM25 boosting irrelevant segments containing “GPD.” |
| **LPD/BIRDs (Definition)** | Regression persisted (failure).                                                                                                                                                                                                                 | Regression persisted (failure). BM25/RRF failed to restore correct PDF retrieval.                                                              |
| **Overall Outcome**        | Reasoning strength retained, but definition precision further deteriorated. RRF’s high recall introduced more irrelevant context, amplifying retrieval noise.                                                                                   |                                                                                                                                                |

**Summary:**
Hybridization improved theoretical recall but introduced ranking instability. The retrieval layer grew noisier, worsening both definition accuracy and GPD consistency.

---

### **Phase 3 → Phase 4: Adding Cross-Encoder Reranking**

| Detail                     | **Phase 3 (Hybrid RAG)**                                                                                                                                                                                                                                                                                  | **Phase 4 (Full Reranked Hybrid)**                                                                                                                  |
| -------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Architectural Change**   | Hybrid RAG (Dense + BM25/RRF)                                                                                                                                                                                                                                                                             | Hybrid RAG + **BGE Cross-Encoder Reranker** for post-RRF re-ranking                                                                                 |
| **Rationale**              | Phase 3 produced too many partially relevant results. Although recall was high, precision was poor — the top-ranked result was often wrong (e.g., the “GPD prognostic value” error). The reranker was added to deeply evaluate the top 10–15 RRF results and surface the most contextually accurate ones. |                                                                                                                                                     |
| **Key Impact Summary**     |                                                                                                                                                                                                                                                                                                           |                                                                                                                                                     |
| **GPD (Hallucination)**    | “Generalized Prognostic Value.”                                                                                                                                                                                                                                                                           | **Resolved** — Correctly identified “Generalized Periodic Discharge.” The reranker successfully promoted the right PDF chunk.                       |
| **LPD/BIRDs (Definition)** | Regression persisted (failure).                                                                                                                                                                                                                                                                           | Regression still persists (failure). The reranker could not recover missing definitions, suggesting a fundamental retrieval issue prior to ranking. |

**Summary:**
The addition of reranking finally stabilized factual precision and eliminated the GPD hallucination.
However, the retrieval layer’s inability to isolate short, definition-based text fragments continued to block simple term answers.

---

## **2. Overall Comparison and Insights**

| **Feature**              | **P1 (Dense 2-Source)**   | **P2 (Dense 3-Source)**        | **P3 (Hybrid RRF)**             | **P4 (Full Reranked Hybrid)** |
| ------------------------ | ------------------------- | ------------------------------ | ------------------------------- | ----------------------------- |
| **Complex Reasoning**    | ✗ Failed                  | ✓ Success                      | ✓ Success                       | ✓ Success                     |
| **GPD Definition**       | ✗ Hallucination (“Spike”) | ✗ Hallucination (“Paroxysmal”) | ✗ Worsened (“Prognostic Value”) | ✅ **Resolved (Correct)**      |
| **LPD/BIRDs Definition** | ✓ Success                 | ✗ Regression                   | ✗ Regression                    | ✗ Regression                  |

**Key Takeaways:**

* The **reasoning capability** introduced in Phase 2 has remained consistent and stable through all later phases.
* The **GPD hallucination**, once the system’s critical flaw, was fully corrected by Phase 4 through reranking.
* The **definition regression (LPD/BIRDs)** has persisted since Phase 2, unaffected by architectural refinements, pointing to a retrieval-level contamination issue.

---

## **3. Root Cause and Corrective Strategy**

### **Root Cause**

The introduction of video and EEG data (Phase 2 onward) flooded the vector index with non-definitional content containing repeated mentions of terms like “LPD.”
As a result, dense retrieval began prioritizing irrelevant but semantically similar chunks, overshadowing the concise definitions from the PDF.

Even with BM25 keyword balancing (Phase 3) and contextual reranking (Phase 4), the underlying retrieval pool remained polluted, preventing clean definition retrieval.

### **Proposed Immediate Fix**

Implement **source gating** during retrieval:

* When `query_type == 'definition'`, exclude `video_reasoning` and `eeg_insights` sources entirely.
* Restrict retrieval to the **PDF-based medical_definitions** collection.

This will ensure that short, authoritative definitions dominate the search results, restoring the system’s ability to answer basic definition queries accurately.

---

## **4. Conclusion**

The final **Phase 4 (Full Reranked Hybrid RAG)** stands as the most balanced and capable configuration to date.
It demonstrates expert-level reasoning, accurate clinical definitions for complex terms like GPD, and stable overall performance across reasoning tasks.

However, the persistent failure on short definitions (*LPD*, *BIRDs*) underscores a structural retrieval flaw rather than a ranking one.
Future iterations should focus on targeted retrieval filtering, maintaining clean source separation for query-specific accuracy.

Once source gating is in place, the system will achieve **both expert-level reasoning and precise definitional retrieval**, completing its intended design trajectory.

---

**Prepared by:**
*EEG Insight AI Engineering Team*
**Reviewed by:** Lead LLM Engineer, October 2025

---
