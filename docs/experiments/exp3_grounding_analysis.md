# **EEG RAG System – Experimental Development Report**

**Date:** October 14, 2025
**Model:** Gemma 2B
**Experiments:** 9 iterations (exp0 – exp8)


## **1. Executive Summary**

This document outlines the iterative development of a Retrieval-Augmented Generation (RAG) system built for clinical EEG data interpretation. Over nine experiment cycles, the project evolved from an unstable prototype (**exp0**)—plagued by factual hallucinations and inconsistent retrieval—to a well-structured and reliable hybrid architecture (**exp8**).

The earliest versions lacked grounding in medical terminology and relied on naive retrieval, which led to misleading responses and misattributed patient data. Introducing a **dual-source retrieval setup** (EEG segments + medical definitions) in **exp1** laid the foundation for factual accuracy.

Subsequent improvements showed that for a small instruction-tuned model like **Gemma 2B**, **clear directive prompting** outperformed both complex multi-step reasoning and few-shot examples. However, the most important architectural advance arrived with **exp5**, where a **metadata-based filtering layer** was added before retrieval. This single change eliminated the chronic issue of contextual misattribution by enforcing data integrity at the database level.

The final version (**exp8**) uses a **“filter-then-fallback” hybrid retrieval**: it first applies exact metadata matching, and only if that fails, falls back to semantic search. This combination delivers both high precision and flexibility while staying efficient on limited hardware.

In short, this work reinforced a key lesson: in RAG systems, **retrieval intelligence matters as much as model intelligence**.

---

## **2. Iteration Analysis – From Baseline to Hybrid Retrieval**

### **exp0 – Naive Baseline**

**Architecture & Issues:**
The initial system used a single-source RAG setup (only `eeg_insights`) and a four-step pipeline — Summarize → Answer → Validate → Correct. It looked structured but behaved unpredictably.

* **Hallucinations:** When asked “What is LPD?”, the model guessed “low-power delta activity,” an entirely fabricated concept.
* **Context drift:** For invalid patient IDs, it retrieved unrelated yet medically similar data, confidently presenting it as correct.
  These failures highlighted that excessive pipeline depth cannot fix poor retrieval grounding.

---

### **exp1 – exp2 – Foundational Rework**

**Changes Introduced:**

* Added a **second data source** (`medical_definitions`) to ground terminology.
* Tested simplified pipelines: **exp1** (single pass) and **exp2** (two-step summarization + answer).

**Outcomes:**

* **exp1:** Over-simplification caused the model to almost always refuse to answer (“The context does not provide…”).
* **exp2:** Marginally better, but the summarization stage often dropped critical content, leading to incomplete answers.
  These iterations clarified that stability couldn’t come from chaining LLM calls—it needed better retrieval logic.

---

### **exp3 – exp4 – Directive Prompting Breakthrough**

**Changes Introduced:**

* **exp3:** Replaced multi-stage flow with a single, **directive prompt** containing explicit rules (e.g., “Use only the data below”).
* **exp4:** Tightened these instructions with structured formatting and exact ID quoting.

**Results:**

* The system became **noticeably more disciplined**, referencing specific patient data as instructed.
* In **exp3**, retrieval errors remained, but now it was clear the issue was upstream—the model correctly processed incorrect inputs.
* In **exp4**, stricter prompting led the model to **refuse inconsistent queries** instead of fabricating answers.

This phase proved that **explicit, rule-based directives** work better for smaller models than complex reasoning chains or few-shot examples.

---

### **exp5 – Metadata Filtering: The Turning Point**

**Implementation:**
This was the most impactful architectural change.

* Added **pre-retrieval query parsing** using regex to detect IDs in user queries.
* If an ID was detected, the system performed an **exact database lookup** instead of a semantic search.

**Impact:**

* **Misattribution eliminated.** The system stopped hallucinating patient data completely.
* Non-existent IDs returned a clean “No data found,” bypassing the LLM entirely.

By moving factual validation to the **retrieval layer**, not the generation layer, this iteration permanently solved the system’s most serious reliability issue.

---

### **exp6 – exp8 – Refinement and Final Architecture**

**exp6 – Few-Shot Regression:**
Tried replacing directive prompts with few-shot examples. It backfired—the model refused most queries. For Gemma 2B, explicit instructions clearly outperformed pattern-learning from examples.

**exp7 – Resource Optimization:**
Re-implemented the exp5 architecture with better efficiency for 8 GB RAM environments. Adjusted `num_ctx` and reduced retrieval depth to fit within memory limits while keeping precision.

**exp8 – Hybrid Retrieval (Final):**

* Combined **metadata filtering** (for precise queries) with **semantic fallback** (for open queries).
* Delivered fast, factual responses for ID-based inputs, while still supporting exploratory medical questions.
* Balanced performance and accuracy, making the system robust across diverse use cases.

---

## **3. Comparative Overview**

| Version  | Strategy                      | Factual Accuracy | Hallucination Control | Key Takeaway                                   |
| :------- | :---------------------------- | :--------------: | :-------------------: | :--------------------------------------------- |
| **exp0** | Single-source RAG             |       Poor       |          None         | Severe hallucination & wrong patient mapping   |
| **exp1** | Single-pass                   |        N/A       |      Full refusal     | Over-simplified, context starvation            |
| **exp2** | Two-step (Summarize → Answer) |        Low       |          Low          | Summarization lost key info                    |
| **exp3** | Directive Prompting           |     Moderate     |        Moderate       | Model obeyed, retriever still weak             |
| **exp4** | Stricter Directives           |       Good       |          Good         | First consistent, safe behavior                |
| **exp5** | Metadata Filtering            |   **Excellent**  |     **Excellent**     | Solved factual misattribution                  |
| **exp6** | Few-shot Prompting            |        N/A       |      Full refusal     | Misaligned with model scale                    |
| **exp7** | Optimized Filtering           |     Excellent    |       Excellent       | Lightweight, production-ready                  |
| **exp8** | **Hybrid Retrieval**          |   **Excellent**  |     **Excellent**     | Final robust solution, high precision + recall |

---

## **4. Conclusion**

The progression from exp0 to exp8 shows how reliability in RAG systems comes not from more generation steps, but from **smarter retrieval**.

Two insights stood out throughout development:

1. **Retrieval intelligence is non-negotiable.**
   Shifting accuracy control upstream—through ID parsing and metadata filtering—was the defining breakthrough. It ensured factual grounding before any text generation occurred.
2. **Prompt design must match model capacity.**
   Compact instruction-tuned models like Gemma 2B perform best when given clear, rule-based directives rather than abstract reasoning tasks or few-shot learning.

The final system (exp8) successfully integrates both lessons. It’s lightweight, factual, and dependable—ready for deployment or further scaling with larger models.

---