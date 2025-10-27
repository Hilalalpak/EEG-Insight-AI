---

# **EEG RAG Clinical Report**

**Project:** Retrieval-Augmented Generation (RAG) for EEG Clinical Pattern Interpretation
**Model:** Gemma 2B
**Embedding Models:** all-MiniLM-L6-v2  |  SapBERT-PubMedBERT
**Date:** October 12, 2025

---

## **1. Overview**

This report summarizes a series of case-based experiments evaluating the EEG RAG system’s ability to analyze and interpret clinical EEG segments.
Each case represents a distinct configuration of embedding models and prompt strategies tested on a standardized query set:

1. **Definition queries** – e.g., “What is a seizure?” or “What is LPD?”
2. **Patient-specific retrieval** – e.g., “Show me seizure segments from patient 1002197945.”
3. **Comparative reasoning** – e.g., “Compare seizure patterns in EEG 1001717358 vs 1002197945.”

The objective was to examine how well the RAG system integrates retrieved EEG segment summaries with general clinical reasoning, while maintaining factual accuracy and preventing hallucinated interpretations.

---

## **2. Case Summaries**

### **Case 1 – Baseline RAG (MiniLM Embeddings)**

**Setup:**
A straightforward RAG configuration using *all-MiniLM-L6-v2* for embeddings and a simple expert-style prompt (“You are an expert clinical neurophysiologist…”).

**Observations:**

* Responses accurately described retrieved segments but lacked broader clinical interpretation.
* The model produced repetitive explanations for seizures and misdefined **LPD** as *low-power discharge*.
* Patient-specific queries sometimes mixed results from unrelated EEG IDs.

**Assessment:**
Well-structured but overly literal. Functioned more as a retrieval summarizer than as a reasoning assistant.

---

### **Case 2 – Structured Three-Step Prompt**

**Prompt Framework:**
1️⃣ General definition → 2️⃣ Analyze provided context → 3️⃣ Synthesize and illustrate.

**Outcome:**

* Marked improvement in readability and organization.
* Integrated theoretical definitions (“A seizure is a sudden, involuntary spike in brain activity…”) with retrieved EEG features.
* Still generated fabricated explanations for LPD and minor factual inconsistencies.

**Takeaway:**
Clearer pedagogy and better coherence, but factual grounding remained weak without domain-specific embeddings.

---

### **Case 3 – Domain Embeddings (SapBERT)**

**Configuration:**
Replaced MiniLM with **SapBERT-PubMedBERT** for improved biomedical concept mapping.

**Findings:**

* Enhanced retrieval of medically relevant terms.
* Introduced an embedding-normalization issue: cosine similarity scores became large negative values (≈ −115 to −220).
* Despite richer terminology, clinical accuracy did not significantly improve.

**Interpretation:**
Switching to domain embeddings improved linguistic precision but exposed scaling and normalization flaws in the ChromaDB pipeline.

---

### **Case 4 – Hybrid Prompt + SapBERT**

**Goal:**
Combine the three-step reasoning structure with SapBERT embeddings to test whether better context retrieval improved factual grounding.

**Results:**

* Definitions of seizures were generally accurate.
* Patient-specific retrieval errors persisted (EEG segments from multiple IDs mixed).
* Misinterpretation of **LPD** continued (“low-power delta,” “linear power density,” etc.).

**Summary:**
Technical upgrades did not correct the architectural retrieval flaw. The model’s responses sounded confident but remained unreliable for precise medical interpretation.

---

### **Case 5 – Persona-Based Prompt (“Dr. Chen”)**

**Intent:**
Simulate domain authority by embedding a clinician persona to encourage structured, clinically reasoned answers.

**Outcome:**

* Writing style improved slightly (more natural phrasing).
* Content quality dropped—responses became shorter and less analytical.
* Higher temperature (0.75) increased verbosity and drifted from clinical focus.

**Conclusion:**
Persona prompts exceed the cognitive capacity of small models like Gemma 2B. Role framing alone does not yield factual improvements.

---

### **Case 6 – Balanced Configuration**

**Settings:**
Gemma 2B + SapBERT + Three-step prompt + optimized parameters (temperature 0.6, top p 0.9).

**Performance:**

* Most consistent behavior overall.
* Cleaner explanations, moderate coherence across all queries.
* Still suffered from patient-filtering errors and context saturation (five long segments exceeded context window).

**Final Evaluation:**
A well-balanced configuration that reached the system’s upper bound given current architecture.

---

## **3. Technical Insights**

### **3.1 Embedding Pipeline Issues**

Negative similarity values in SapBERT runs revealed missing L2 normalization. Without it, ChromaDB miscalculates cosine distances, severely distorting retrieval relevance. Normalization should be applied during both indexing and querying.

### **3.2 Patient-ID Filtering**

All versions failed to constrain retrieval to a single EEG ID. Mixed-patient outputs made downstream comparisons clinically meaningless. Adding an explicit `eeg_id` filter in the Streamlit UI would resolve this issue with minimal code change.

### **3.3 Prompt Engineering**

While step-based prompts improved structure, they could not compensate for missing context or flawed retrieval. Excessively long context windows degraded Gemma 2B performance by exceeding its attention limit (~2 k tokens).

---

## **4. Observed Hallucination Patterns**

| Query Type          | Typical Hallucination       | Example                                                       |
| ------------------- | --------------------------- | ------------------------------------------------------------- |
| Definition          | Invented acronym expansions | “LPD = Low Power Delta”                                       |
| Patient-specific    | Mis-attributed segments     | Mixed 1002197945 + 1001717358                                 |
| Comparison          | Oversimplified metrics      | “Amplitude 2263 vs 2271 → different severity”                 |
| General explanation | Redundant phrasing          | “Abnormal synchronized neuronal discharges” repeated verbatim |

---

## **5. System Limitations**

1. **No pre-validation of patient IDs** → false retrieval context.
2. **No context compression** → token overflow reduces focus.
3. **Limited medical knowledge base** → hallucinated definitions.
4. **Single-model validation** → cannot self-detect errors.
5. **Small model capacity** → weak multi-document reasoning.

---

## **6. Recommendations**

### **Immediate Fixes**

* **Normalize embeddings** before indexing.
* **Add patient-ID filtering** in the retrieval pipeline.
* **Re-index ChromaDB** after normalization.

### **Short-Term Enhancements**

* Compress EEG summaries to key metrics (power, SEF, label).
* Combine deterministic signal stats with LLM interpretation.
* Post-process LLM responses to deduplicate and flag uncertain claims.

### **Long-Term Directions**

* Introduce query-intent classification (definition / patient / comparison).
* Implement retrieval re-ranking by exact patient ID and signal quality.
* Add an evaluation framework to benchmark accuracy and consistency.

---

## **7. Conclusions**

The experiments highlight that most observed errors stemmed not from the language model itself but from **retrieval and engineering gaps**:

* **Embedding normalization** and **patient filtering** are critical foundations.
* **Prompt tuning** yields only marginal benefits without clean data retrieval.
* The small Gemma 2B model handles basic definitions well but cannot perform robust clinical reasoning or multi-document synthesis.

With these architectural fixes, the EEG RAG system can serve effectively as a **research and educational platform** for exploring EEG patterns, though it is **not yet clinically reliable**. Future work should focus on scalable context management, knowledge integration, and larger language models (≥ 7B parameters) for true clinical decision support.

---