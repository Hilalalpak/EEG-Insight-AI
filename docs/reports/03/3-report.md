# **EEG RAG System \- Technical Experimental Report**

**Project:** RAG-Based Question-Answering System for Medical EEG Data  
**Date:** October 17, 2025  
**Tested Versions:** 9 distinct implementations (exp0 to exp8)  
**Core Model:** gemma:2b

## **1\. Executive Summary**

This report documents the iterative refinement of a Retrieval-Augmented Generation (RAG) system designed for clinical EEG data analysis. Through a series of nine distinct experiments, the system was evolved from a functionally flawed baseline **(exp0)**, which suffered from severe factual hallucination and contextual misattribution, to a robust and reliable final architecture **(exp8)**.  
The initial versions struggled with two core problems: the absence of a grounding knowledge base for medical terminology, and a naive retrieval strategy that led the LLM to process incorrect context. The introduction of a **dual-source retrieval** mechanism **(exp1)**, combining EEG segment summaries with a medical terminology knowledge base, was the first critical step toward resolving these issues.  
Subsequent experiments demonstrated that for a small, instruction-tuned model like gemma:2b, **explicit directive prompting (exp3, exp4)** is significantly more effective than complex multi-step pipelines or few-shot learning approaches. However, the most critical breakthrough was the implementation of a **pre-retrieval metadata filtering (exp5)** layer, which definitively solved the problem of contextual misattribution by enforcing factual constraints at the database level.  
The final architecture **(exp8)** integrates these learnings into a hybrid "filter-then-fallback" retrieval strategy, optimized for a low-resource environment. This version successfully balances the precision of metadata filtering with the flexibility of semantic search, resulting in a system that is both factually accurate and contextually relevant. The journey from exp0 to exp8 underscores the principle that in RAG systems, **the intelligence of the retriever is as critical as the generative power of the LLM.**

## **2\. Iteration Analysis: From Baseline to Hybrid Retrieval**

### **exp0: The Naive Baseline**

#### **2.1 System Architecture & Weaknesses**

The initial version was a standard, single-source RAG system using only the eeg\_insights collection. It employed an overly complex 4-step pipeline (Summarize, Answer, Validate, Correct) that created a false sense of security.

* **Factual Hallucination:** When asked to define a term not present in the context (e.g., "What is LPD?"), the model invented a definition ("low-power delta activity") based on a plausible but incorrect interpretation of the acronym.  
* **Contextual Misattribution:** When queried for a non-existent patient ID ("9999999999"), the semantic retriever fetched data for a medically similar but incorrect patient. The LLM then proceeded to confidently present this incorrect data as belonging to the requested patient. This was the system's most critical failure.

### **exp1 & exp2: Foundational Changes and Initial Failures**

#### **2.1 Implemented Improvements**

* **Dual-Source Context (exp1):** The architecture was fundamentally improved by adding the medical\_definitions collection. This provided an external knowledge base to ground the model's understanding of medical terminology.  
* **Pipeline Experiments:** exp1 attempted a radical simplification to a single-pass RAG for speed, while exp2 reinstated a two-step (Summarize \-\> Answer) pipeline.

#### **2.2 Test Results and Evaluation**

* The aggressive simplification in **exp1** was a catastrophic failure. The model, starved of context and reasoning steps, defaulted to a state of **constant refusal**, answering nearly every query with "The context does not provide...".  
* The two-step pipeline in **exp2** was a marginal improvement but remained deeply flawed. The summarization step was found to be unreliable, often discarding critical information before it reached the final generation prompt. This highlighted the fragility of chained LLM calls.

### **exp3 & exp4: The Directive Prompting Breakthrough**

#### **2.1 Implemented Improvements**

* **Directive Prompting (exp3):** The pipeline was simplified to a single LLM call, but with a new, highly structured prompt. This "directive prompt" provided the model with a clear set of rules, such as You must answer using ONLY the data below and Use specific numbers from the data.  
* **Stricter Directives (exp4):** The rules were made even more explicit, with instructions like Quote exact patient IDs and a mandated output structure.

#### **2.2 Test Results and Evaluation**

* **A Turning Point:** The impact was immediate. The model began to adhere strictly to the provided context, citing specific numerical values and patient data as instructed.  
* **Isolating the Core Problem:** The model's newfound obedience exposed the retriever's flaws. In exp3, it still misattributed data for patient "42516," but it was now clear that the LLM was correctly processing incorrect input. In exp4, the stricter rules gave the model the necessary guidance to identify this inconsistency and refuse to answer, a significant improvement in safety.  
* **Conclusion:** This phase proved that for small, instruction-tuned models, explicit, rule-based prompts are far more effective than complex, abstract pipelines.

### **exp5: The Architectural Solution \- Smart Metadata Filtering**

#### **2.1 Implemented Improvements**

This iteration introduced the most significant architectural change of the series.

* **Pre-LLM Query Parsing:** Logic was added to the API endpoint to parse the user's query for specific patient or EEG IDs using regular expressions.  
* **Metadata Filtering:** If an ID was found, the system would bypass semantic search and perform a direct, exact-match lookup in ChromaDB using a where filter.

#### **2.2 Test Results and Evaluation**

* **Problem Solved:** This change completely eliminated the contextual misattribution error. A query for a non-existent patient now returned zero results from the database, allowing the system to provide an immediate and factually correct "No data found" response without ever invoking the LLM.  
* **Conclusion:** This marked the system's architectural maturation. By addressing the factual accuracy problem at the **retrieval stage** rather than the generation stage, the root cause of the most severe hallucinations was eliminated.

### **exp6 \- exp8: Refinement, Optimization, and Final Architecture**

#### **2.1 Experiments in Optimization**

* **Failed Few-Shot Attempt (exp6):** An experiment to replace directive prompting with few-shot examples failed completely. The gemma:2b model was unable to generalize from the examples and regressed to a state of constant refusal, confirming that directive prompting was the correct strategy.  
* **Resource Optimization (exp7):** The successful architecture from exp5 was reinstated and optimized for a low-resource (8GB RAM) environment. This involved tuning call\_llm parameters (num\_ctx) and reducing the number of documents retrieved to manage the context window effectively.

#### **2.2 Final Architecture (exp8) \- Hybrid "Filter-then-Fallback" Retrieval**

* **Implemented Improvements:** The final version refines the retrieval logic into a hybrid strategy. It first attempts a precise metadata filter. If, and only if, that search returns no results, it "falls back" to performing a broad semantic search.  
* **Test Results and Evaluation:** This architecture provides the best of both worlds:  
  * **Precision:** For specific queries (e.g., "Tell me about patient 1002379034"), it uses the fast and accurate metadata filter.  
  * **Flexibility:** For broad queries (e.g., "Show high-confidence seizure segments"), it leverages the power of semantic search.  
  * **Robustness:** The system is now resilient, factually grounded, and capable of handling a wide range of query types efficiently.

## **3\. Comparative Analysis**

| Version | Core Strategy | Factual Accuracy | Hallucination Control | Key Outcome |
| :---- | :---- | :---- | :---- | :---- |
| **exp0** | Naive RAG | Very Poor | None | **Failure:** High rate of hallucination and misattribution. |
| **exp1** | Single-Pass | N/A | Refusal | **Failure:** System became non-responsive. |
| **exp2** | Two-Step RAG | Poor | Poor | **Failure:** Unreliable summarization step. |
| **exp3** | Directive Prompting | Moderate | Moderate | **Partial Success:** Improved grounding, but retriever error persisted. |
| **exp4** | Stricter Directives | Good | Good | **Success:** Model learned to refuse when context was inconsistent. |
| **exp5** | **Metadata Filtering** | **Excellent** | **Excellent** | **Breakthrough:** Factual errors for ID queries eliminated at the source. |
| **exp6** | Few-Shot Prompting | N/A | Refusal | **Failure:** Incorrect prompting strategy for the model. |
| **exp7** | Optimized Filtering | Excellent | Excellent | **Success:** Production-ready architecture for low-resource environments. |
| **exp8** | **Hybrid Retrieval** | **Excellent** | **Excellent** | **Success:** Most robust and flexible architecture, balancing precision and recall. |

## **4\. Conclusion**

The evolution from exp0 to exp8 demonstrates a clear path to building a reliable RAG system. The initial assumption that a sophisticated, multi-step LLM pipeline could fix underlying data issues proved incorrect. Instead, success was achieved by focusing on two fundamental principles:

1. **Intelligent Retrieval is Non-Negotiable:** The most significant improvements came from enhancing the retriever. By implementing a pre-retrieval layer that can parse user intent and apply database-level filters, the system was able to guarantee factual accuracy for entity-specific queries, effectively solving the core hallucination problem.  
2. **Prompts Must be Tailored to the Model:** Small, instruction-tuned models like gemma:2b do not benefit from complex reasoning chains or few-shot examples. They perform best when given clear, explicit, and rule-based instructions within a well-structured directive prompt.

The final exp8 architecture, with its hybrid retrieval strategy and optimized directive prompt, stands as a robust and effective solution that successfully meets the project's initial objectives.