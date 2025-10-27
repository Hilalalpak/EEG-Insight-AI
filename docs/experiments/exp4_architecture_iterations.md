---

# **EEG RAG System – Comparative Evaluation of Six Experimental Architectures**

**Project:** Retrieval-Augmented Generation (RAG) for EEG-Based Clinical Interpretation
**Model:** Gemma 2B
**Date:** October 16, 2025
**Experiments:** exp1 → exp6 (21 standardized queries)

---

## **1. Executive Summary**

This report provides a comprehensive review of six RAG architectures developed and tested for clinical EEG interpretation.
The experiments focused on performance across terminology, patient-specific data, and comparative reasoning tasks.

A key finding emerged: **system simplicity correlated directly with reliability.**
The baseline architecture (**exp1**) achieved a **95.2% success rate** and zero timeouts — outperforming every more complex design.
In contrast, successive layers of architectural sophistication (multi-query routing, video integration, contextual prompts) consistently introduced latency, instability, and reduced factual reliability.

The results demonstrate that in RAG pipelines, **precision retrieval and prompt clarity** drive success far more than model complexity or source variety.

---

## **2. Performance Overview**

### **2.1 Accuracy and Reliability**

| System   | Success | Failure | Timeout | Success Rate |
| :------- | :------ | :------ | :------ | :----------- |
| **exp1** | 20      | 1       | 0       | **95.2%**    |
| **exp2** | 19      | 1       | 1       | 90.5%        |
| **exp3** | 16      | 2       | 3       | 76.2%        |
| **exp4** | 16      | 3       | 2       | 76.2%        |
| **exp5** | 16      | 2       | 3       | 76.2%        |
| **exp6** | 15      | -       | 7       | 71.4%        |

The simplest system (exp1) was the most stable. Each new layer of sophistication — additional data sources, multi-query generation, complex routing — degraded performance.

### **2.2 Latency Trends**

| System   | Avg Response Time | Change vs Baseline |
| :------- | :---------------- | :----------------- |
| **exp1** | 32 s              | —                  |
| **exp2** | 36 s              | +12%               |
| **exp3** | 46 s              | +44%               |
| **exp4** | 92 s              | +187%              |
| **exp5** | 84 s              | +162%              |
| **exp6** | 68 s              | +112%              |

Every added processing stage (multi-query generation, multiple source retrievals) introduced exponential delay.

---

## **3. Experiment Summaries**

### **exp1 – Baseline Architecture (⭐ Recommended)**

**Design:**

* Two collections (EEG segments + medical definitions).
* Regex-based patient ID extraction.
* Direct semantic search with metadata filtering.

**Performance:**

* **Highest reliability (95.2%)**, fastest average time (32s).
* Perfect accuracy on patient-specific queries.
* Only weak area: limited depth on terminology questions like “What are BIRDs?”.

**Verdict:**
Simple, stable, and robust — ideal for production use where reliability and speed are critical.

---

### **exp2 – Video Source Integration**

**Change:** Added ACNS educational video transcripts as a third data source.

**Result:**

* Slight drop in success rate (–4.7%) and small latency increase.
* Video data rarely contributed meaningfully; transcript chunks were too broad and semantically inconsistent.
* SapBERT embeddings mismatched against conversational transcript text.

**Verdict:**
Extra context increased overhead but added little value to factual performance.

---

### **exp3 – Contextual Prompting**

**Change:** Introduced adaptive, context-aware prompts — but mixed Turkish and English instructions.

**Result:**

* Sharp decline in reliability (–14.3%).
* Turkish directives caused parsing confusion for the English-trained Gemma model.
* Some improvements in table formatting and comparative clarity, but not enough to offset the regression.

**Verdict:**
Over-engineered; language inconsistency led to major failures.

---

### **exp4 – Multi-Query Decomposition**

**Change:** Each user query split into 2–3 sub-queries, each searching all sources independently.

**Result:**

* No accuracy gain, but latency doubled (up to 92 seconds).
* Frequent **patient ID loss** during sub-query generation.
* Occasional breakthroughs (e.g., first correct BIRDs response), but inconsistent.

**Verdict:**
Conceptually strong but practically inefficient. Multi-query cost outweighed any marginal accuracy gain.

---

### **exp5 – Video-Prioritized Multi-Query**

**Change:** Modified exp4 to prioritize video transcripts for terminology definitions.

**Result:**

* Produced the **only correct BIRDs definition** referencing ACNS standards.
* Maintained high latency and timeout rate.
* Still failed patient ID–based retrievals.

**Verdict:**
Excellent terminology recall but fragile and slow. Valuable for educational use, not production.

---

### **exp6 – Hybrid Routing System**

**Architecture:** Adaptive routing by query type:

* Patient → exp1 pipeline
* Terminology → exp5 pipeline
* Comparison → structured analysis

**Result:**

* Highest-quality responses when successful, but 33% timeout rate.
* Most complex logic layer, yet still limited by small-model constraints and fixed timeout thresholds.

**Verdict:**
Promising conceptually; needs optimization and better timeout management.

---

## **4. Diagnostic Insights**

### **4.1 Metadata Gap**

Confidence scores and vote distributions existed in raw data but were never embedded.
As a result, all systems failed queries related to expert agreement or confidence metrics.

### **4.2 Video Chunking**

5000-character chunks contained multiple unrelated terms (LPD, GPD, BIRDs, IIC), confusing the retriever.
Term-based chunking would have yielded better precision.

### **4.3 Context Window Saturation**

As context size increased, Gemma 2B’s token limit (≈3K tokens) was repeatedly exceeded, causing response truncation or timeouts.

---

## **5. System Comparison**

| Metric           | exp1   | exp2 | exp3 | exp4 | exp5 | exp6   |
| :--------------- | :----- | :--- | :--- | :--- | :--- | :----- |
| Accuracy         | 8      | 7    | 5    | 7    | 7    | **9*** |
| Reliability      | **10** | 9    | 6    | 6    | 6    | 4      |
| Response Quality | 6      | 7    | 5    | 8    | 8    | **9*** |
| Speed            | **10** | 8    | 6    | 2    | 3    | 5      |
| Coverage         | 6      | 6    | 5    | 7    | 7    | 8      |
| **Total**        | **40** | 37   | 27   | 30   | 31   | 35     |

(*Scores with asterisk = “when successful.”*)

---

## **6. Recommendations**

### **Production Use**

→ **exp1** (baseline)

* Fast, reliable, and stable.
* Ideal for real-time EEG query workflows.
* Trade-off: Limited domain depth.

### **Medical Education**

→ **Hybrid: exp6 router + exp5 terminology pipeline**

* Best contextual definitions, ACNS citations.
* Acceptable for non-critical environments where latency is tolerable.

### **Research Applications**

→ **exp1 + exp4 selective augmentation**

* Use exp4’s multi-query decomposition only for difficult terminology questions.

---

## **7. Implementation Priorities**

### **Immediate**

* Embed metadata (confidence scores, expert votes).
* Add patient ID preservation to all pipelines.
* Normalize video transcripts via term-based chunking.

### **Short-Term**

* Implement hybrid retrieval (semantic + metadata).
* Introduce timeout-aware query routing.

### **Long-Term**

* Evaluate exp6 after fixes; A/B test against exp1.
* Consider larger LLM (≥7B) for multi-source reasoning.

---

## **8. Key Takeaways**

1. **Simplicity Wins:** The baseline model outperformed every complex system.
2. **Retrieval Quality > Model Complexity:** Data cleanliness and retrieval precision determined factual accuracy.
3. **Prompt Design Over Architecture:** Consistent, concise, English-only prompts outperformed dynamic or role-based ones.
4. **Metadata Is Critical:** Confidence and expert agreement data hold untapped accuracy potential.
5. **Engineering Before Scaling:** Fixing retrieval logic and normalization yields higher ROI than scaling model size prematurely.

---

## **9. Conclusion**

The EEG RAG project’s progression demonstrates a universal truth in applied RAG engineering:
**complexity often degrades reliability.**

The exp1 baseline — fast, deterministic, and cleanly retrieved — remains the most dependable configuration.
While advanced variants like exp6 show theoretical promise, production deployment should prioritize **robust retrieval, metadata integration, and contextual efficiency** over additional architectural layers.

---