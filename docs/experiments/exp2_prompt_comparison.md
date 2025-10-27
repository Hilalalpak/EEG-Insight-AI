# **EEG RAG System – Experimental Evaluation Report**

**Date:** October 13, 2025
**Model:** Gemma 2B
**Versions Tested:** 4 (Main1 → Main4)
**Test Set:** 5 standardized queries

## **1. Executive Summary**

This report documents four major iterations of the EEG RAG system, designed to improve factual grounding and reduce hallucinations in a medical Q&A context.
Across all versions, the same test set revealed a consistent pattern: while technical precision improved slightly with stricter prompts and better validation, **none of the models achieved reliable factual accuracy.**

The main bottlenecks were:

* **Context insufficiency:** key definitions (e.g., “What is LPD?”) were missing from the retrieved data.
* **Weak hallucination control:** all systems continued to infer nonexistent facts confidently.
* **Flawed self-validation:** using the same model for both generation and verification created circular reasoning.

Although the later versions (**Main3** and **Main4**) improved code quality and structure, they failed to address the fundamental issue—**the absence of true external validation.**

The experiments collectively underscore that multi-step pipelines don’t automatically improve reliability. Progress depends instead on **strong retrieval logic**, **knowledge base integration**, and **independent validation models.**

---

## **2. Iteration Analysis**

### **Main1 – Baseline Vanilla RAG**

**Setup:**
A straightforward RAG pipeline retrieved EEG segments from ChromaDB and fed them to the model for response generation. The prompt combined internal knowledge with retrieved context, aiming for synthesis between the two.

**Results:**

* When asked “What is LPD?”, the model guessed “very high amplitude activity,” confusing it with GPD.
* When queried with a **non-existent patient ID**, it fabricated a plausible medical narrative using a different patient’s data.
* Given a deliberately wrong statement (“LPD = low-power discharge”), it accepted the false claim and expanded on it.
* In a comparison task, it referenced patient IDs that didn’t exist in the retrieved data.

**Takeaways:**
Main1 performed fast and used relevant medical terms, but lacked any factual safeguards. The temperature (0.6) was too high, allowing confident fabrication.

**Strengths:** quick responses, correct numeric metrics.
**Weaknesses:** zero hallucination control, no validation, unreliable grounding.

---

### **Main2 – Strict Grounding RAG**

**Changes Introduced:**
To combat hallucinations, a rule-based prompt was introduced:

```
1. Use only the given segments.
2. Cite segment numbers.
3. If no answer, say “insufficient information.”
4. Never make up definitions.
```

Additional changes:

* Temperature reduced to 0.1
* Added repeat penalty and stop tokens
* Implemented a lightweight hallucination detector (`_contains_hallucination()`)

**Results:**
While the model became more conservative, it still fabricated definitions (“low-power activity”) and misattributed patients. The hallucination filter missed subtle cases, and patient ID validation failed because the query ID was never checked—only the response.

**What Worked:**

* Lower temperature improved consistency.
* Numbered segments helped partial grounding.
* Stop tokens prevented prompt leakage.

**What Failed:**

* Rule violations persisted.
* Post-check logic too fragile.
* ID validation incomplete.

**Summary:** Cleaner implementation, slightly safer behavior, but still 80% hallucination rate.

---

### **Main3 – Self-Validating RAG**

**Motivation:**
After repeated failures, the idea was to let the model **validate and correct its own outputs** via a four-step pipeline.

**Pipeline:**

1. **Summarization:** condense context.
2. **Answer Generation:** produce an explanation.
3. **Self-Validation:** check correctness and confidence.
4. **Self-Correction:** revise if validation failed.

**Outcome:**
The approach backfired. The model confidently invented new medical terms like “Low-Power Dopaminergic Activity.” Validation then marked these as *PASS* — since the same model evaluated its own hallucinations.

**Root Cause:**
Using a single model for both reasoning and validation introduced **the same-model paradox** — a model cannot detect the hallucinations it generated.

**Performance:**

* 4× slower and more expensive than Main1.
* Validation logic easily bypassed due to parsing errors.
* Hallucination rate: 100%.

**Conclusion:**
Self-validation provided only an illusion of safety.

---

### **Main4 – Hybrid Simplified RAG**

**Objective:**
A refactored version of Main3 focused on code quality and efficiency, not logic overhaul.

**Engineering Improvements:**

* Centralized LLM call function (`call_llm()`), eliminating duplication.
* Reduced token limits for each step.
* Improved error handling and readability (DRY principles, type hints).

**Results:**
Functionally similar to Main3: slight improvement in wording, but factual reliability unchanged.
Responses such as “LPD refers to high amplitude, clean signal” persisted, showing the same inference errors as before.

**Summary:**
Main4 represented the cleanest code but the same flawed reasoning pipeline.

---

## **3. Comparative Analysis**

| Version   | Pipeline                 | Hallucination Rate | Latency | Grounding | Maintainability | Outcome                          |
| :-------- | :----------------------- | :----------------- | :------ | :-------- | :-------------- | :------------------------------- |
| **Main1** | Single-pass              | 80%                | ~5s     | 3/10      | 6/10            | Naive but fast                   |
| **Main2** | Strict rules + filters   | 80%                | ~6s     | 4/10      | 7/10            | Minimal improvement              |
| **Main3** | Self-validating (4-step) | 100%               | ~20s    | 2/10      | 5/10            | Failed validation logic          |
| **Main4** | Refactored hybrid        | 80%                | ~18s    | 4/10      | 8/10            | Best engineering, poor reasoning |

**Observations:**

* Complex pipelines increased latency without improving reliability.
* Hallucination detection logic caught syntax issues, not factual ones.
* Consistent model behavior showed the limits of Gemma 2B without external fact-checking.

---

## **4. Root Causes**

1. **Missing Contextual Definitions:**
   None of the retrieved documents defined “LPD.” Without grounding, every version hallucinated a definition.

2. **Broken ID Validation:**
   The code checked IDs *after* response generation instead of before the LLM call. Missing pre-validation allowed fabricated patients.

3. **Blind Agreement with User Input:**
   All systems accepted incorrect user statements instead of correcting them. The models treated misinformation as context.

4. **Same-Model Validation:**
   Validation and generation sharing the same weights invalidated the purpose of “self-checking.”

---

## **5. Recommendations**

1. **Pre-LLM Validation Layer:**

   * Verify patient IDs before generation.
   * Detect query type (definition vs data).
   * Refuse invalid or missing-context queries early.

2. **Medical Knowledge Base Integration:**
   Link to **UMLS** or **SNOMED CT** to verify terms like “LPD” or “GPD.” Prevents made-up definitions.

3. **Independent Validation Model:**
   Use a stronger secondary model (e.g., Llama3-70B) for factual verification.

4. **Confidence Scoring:**
   Classify responses by confidence level based on grounding depth (e.g., “Context-based: High,” “Inference: Medium”).

5. **Prompt Redesign:**
   Replace “Use your medical knowledge” with explicit instructions like:

   > “If the context is insufficient, state that and stop.”

---

## **6. Conclusion**

Across four iterations, the EEG RAG system demonstrated consistent limitations of small instruction-tuned models when used in a sensitive, factual domain like clinical EEG analysis.

**Key lessons learned:**

* **Complexity ≠ Accuracy:** Multi-step reasoning and self-validation did not reduce hallucinations.
* **Validation must be external:** The same model cannot reliably critique itself.
* **Retrieval and grounding matter most:** Without complete, labeled context, even a perfect prompt fails.

**Final Verdict:**

* **Main1:** Simple baseline, functional but unreliable.
* **Main2:** Minor gains via stricter prompts.
* **Main3–4:** Over-engineered, slower, still hallucinatory.

Moving forward, future iterations should shift focus from prompting tricks to **retrieval intelligence**, **knowledge grounding**, and **independent validation layers** — the real drivers of reliability in medical RAG systems.

---