# EEG RAG System - Technical Experimental Report

**Project:** RAG-Based Question-Answering System for Medical EEG Data  
**Date:** October 2025  
**Tested Versions:** 4 different implementations  
**Test Set:** 5 standard questions

---

## 1. MAIN1 - Baseline Vanilla RAG

### 1.1 System Architecture

The first version implements a classical RAG approach consisting of three steps:

1. Semantic search retrieves relevant EEG segments from ChromaDB
2. These segments are provided to the LLM as context
3. The model generates answers using both its internal knowledge and the context

The prompt strategy was designed as follows:
- First provide a general medical definition (model's internal knowledge)
- Analyze examples in the context
- Synthesize both to create the explanation

### 1.2 Parameter Selection

Temperature was set to 0.6. The rationale was to balance creativity with context adherence. In retrospect, this proved slightly too high.

Context window was set to 8192 tokens, maximum output to 2048 tokens. EEG segments are quite lengthy, requiring a large context window.

### 1.3 Test Results

**"What is LPD?" query:**
The model defined LPD as "very high amplitude activity". Actually, LPD (Lateralized Periodic Discharges) is something different, but the model attributed characteristics of GPD (Generalized Periodic Discharges) segments it saw in the context to LPD.

The context contains LPD-labeled segments but no definition. The model made an error while trying to fill this gap.

**"Patient 9999999999" query:**
A non-existent patient ID was queried. The model started with "Patient 9999999999 may be experiencing..." and continued discussing patient 1002136740. It used the ID from the query but actually described a different patient's data.

**"LPD = low-power discharge" query:**
The user deliberately provided an incorrect definition. The model did not correct this, instead beginning to explain "Low-power discharge (LPD) refers to the absence of detectable electrical activity".

Interestingly, it interpreted segments showing "high amplitude" in the context as "low-power". There is an obvious contradiction here, but the model did not notice.

**Comparison query:**
"EEG 1001717358 vs 1002197945" comparison was requested. Only 1002136740 exists in the context. The model still started with "EEG segment from patient 1001717358" and showed data from a different patient.

### 1.4 Strengths

- Used technical details well (power, SNR, SEF values reported correctly)
- Captured differences between segments
- Medical terminology usage was appropriate
- Fast response time (single LLM call)

### 1.5 Weaknesses

- No hallucination control
- No patient ID validation
- Does not correct misinformation
- Fabricates information not in context
- Temperature slightly too high

---

## 2. MAIN2 - Strict Grounding RAG

### 2.1 Implemented Improvements

Several changes were made to address the hallucination problem in Main1:

**Prompt modifications:**
```
CRITICAL RULES:
1. Answer ONLY using information from segments
2. ALWAYS cite segment numbers
3. If segments don't contain answer: "insufficient information"
4. DO NOT make up medical definitions
```

Context was converted to numbered segments. Each segment was marked as [SEGMENT 1], [SEGMENT 2]. This way the model could provide references like "According to Segment 2...".

**Parameter tuning:**
- Temperature: 0.6 → 0.1 (aggressive reduction)
- top_k: 40 → 20 (limit response diversity)
- repeat_penalty: 1.2 added
- Stop tokens: ["USER QUESTION:", "CRITICAL RULES:"] (prevent prompt leak)

**Post-processing added:**
A `_contains_hallucination()` function was written. It searches for suspicious expressions using pattern matching:
- Uncertainty expressions like "I believe", "I think"
- Non-existent abbreviations like "GEP", "HFE"
- Patient IDs not in context

### 2.2 Test Results

**"What is LPD?" query:**
Despite strict grounding rules, the model still made an incorrect definition: "LPD (low-power activity) refers to a reduced or absent electrical signal".

This is interesting because the prompt says "DO NOT make up medical definitions". Apparently the model does not see this as a definition, but as an inference from context.

Hallucination detection was not triggered. "low-power activity" is not in the suspicious_patterns list, so it passed.

**"Patient 9999999999" query:**
Same error again. It said "Patient 9999999999 has multiple EEG segments..." and described 1002136740.

Patient ID validation did not work because it only checks IDs in the response, not in the query:

```python
# Current code:
mentioned_eeg_ids = re.findall(r'\b\d{10}\b', llm_response)
context_eeg_ids = re.findall(r'patient (\d{10})', doc)
if mentioned_eeg_ids - context_eeg_ids:
    return True
```

The 9999999999 in the query was never checked.

**Segment referencing:**
Segment numbers were used in some answers but inconsistently. The model sometimes says "Segment 2", sometimes forgets.

### 2.3 What Worked

- Temperature reduction was effective, model became more conservative
- Numbered segments good idea, worked in some places
- Stop tokens prevented prompt leak
- Repeat penalty reduced repetitions

### 2.4 What Did Not Work

- Hallucination detection too primitive
- Pattern matching insufficient (only 3-4 hard-coded patterns)
- Patient ID validation does not check query
- Grounding rules bypassed by model

Minimal improvement over Main1. Code more structured but results similar.

---

## 3. MAIN3 - Self-Validating RAG

### 3.1 Motivation

After Main2's failure, a more radical approach was attempted. Idea: have the model validate its own answer and correct if necessary.

### 3.2 4-Step Pipeline

**Step 1 - Summarization (temp=0.0):**
Summarize context and extract only relevant information. Max 300 words. Purpose is to filter noise.

**Step 2 - Answer Generation (temp=0.2):**
Generate answer using both medical knowledge and summary. Structure:
- Medical Background (general knowledge)
- Evidence from EEG Data (examples from context)

**Step 3 - Self-Validation (temp=0.0):**
Model checks its own answer:
- Is medical terminology correct?
- Are claims supported?
- Any hallucinations?

Output format:
```
VALIDATION: [PASS/FAIL]
CONFIDENCE: [HIGH/MEDIUM/LOW]
ISSUES: [problems or "None"]
```

**Step 4 - Self-Correction (temp=0.1):**
If validation fails or confidence is LOW, correction is performed.

### 3.3 Test Results

**"What is LPD?" query:**
Disaster. The model said "LPD stands for Low-Power Dopaminergic Activity".

This term does not exist in neurology. The worst answer from all versions came from here.

What did validation say? Probably PASS because no correction was made. The model found its own fabricated term correct in the validation step.

**"Patient 9999999999" query:**
"Patient 9999999999 appears to be experiencing a GPD-related condition". Hallucination again, validation did not catch it.

### 3.4 Why Self-Validation Failed

**Same Model Paradox:**
The same model (gemma:2b) both generates answers and validates them. It is not surprising that the mind making errors finds the same errors normal in validation.

Analogy: You ask a student to both do homework and check their own homework. They will misunderstand the same concept in validation that they misunderstood initially.

**Ground Truth Absence:**
There is no external source to check medical terminology. When the model says "Low-Power Dopaminergic Activity", it does not know this is wrong because there is no knowledge base to compare against.

**Validation Parsing Risk:**
```python
passed = "VALIDATION: PASS" in validation_text.upper()
```

The LLM may not produce this exact format. It might write "The validation passes" or "Status: Passed". If parsing fails, wrong result.

### 3.5 Performance

- 4 LLM calls → ~20 seconds latency
- 4x slower than Main1
- 4x cost
- Worse results

Self-validation created a false sense of security. You think the system validates hallucinations but actually it catches nothing.

---

## 4. MAIN4 - Hybrid Simplified RAG

### 4.1 Approach

Refactored version of Main3. Same 4-step pipeline but cleaner code.

**Centralized LLM Function:**
```python
def call_llm(prompt, temperature=0.1, max_tokens=1024):
    # Single function for all LLM calls
    # DRY principle
```

Main3 had separate functions for each step with much code duplication. This was cleaned up.

**Token Efficiency:**
- Summary: 512 tokens max
- Validation: 256 tokens max  
- Answer: 1024 tokens max

Main3 had higher limits but they were unnecessary.

### 4.2 Test Results

Almost identical results to Main3. Makes sense because underlying logic is the same.

**"What is LPD?" query:**
"LPD refers to High Amplitude, Clean Signal" - same error as Main1. At least it did not say "Low-Power Dopaminergic Activity".

**Other questions:**
Somewhere between Main1 and Main3. Sometimes like Main1, sometimes like Main3.

### 4.3 Code Quality

Cleanest code here. Best practices applied:
- DRY principle
- Good error handling
- Type hints present
- Readable

But clean code does not solve hallucination.

### 4.4 Evaluation

Better version of Main3 from engineering perspective but does not solve the fundamental problem. Same model validation still present.

---

## COMPARATIVE ANALYSIS

### Hallucination Rates
```
Main1: 4/5 questions wrong (80%)
Main2: 4/5 questions wrong (80%)
Main3: 5/5 questions wrong (100%)
Main4: 4/5 questions wrong (80%)
```

Main3 worst. Self-validation fixed nothing, added new hallucinations on top.

### Latency
```
Main1: ~5s  (1 call)
Main2: ~6s  (1 call + post-processing)
Main3: ~20s (4 calls)
Main4: ~18s (3-4 calls)
```

Multi-step pipelines 4x slower but results not better.

### Context Grounding
```
Main1: 3/10
Main2: 4/10 (segment referencing partially worked)
Main3: 2/10 (worst)
Main4: 4/10
```

Numbered segments (Main2, Main4) helped a bit but not enough.

### Code Maintainability
```
Main1: 6/10 (simple but naive)
Main2: 7/10 (post-processing added)
Main3: 5/10 (over-engineered)
Main4: 8/10 (clean refactor)
```

---

## CORE PROBLEMS

### Problem 1: Information Not in Context

All 4 systems answered "What is LPD?" incorrectly. Because:
- Context has LPD-labeled segments
- But no definition of LPD
- Model tries to fill the gap

Correct approach:
> "The segments are labeled as LPD but don't include a definition. In neurophysiology, LPD stands for Lateralized Periodic Discharges. Would you like me to search for more details?"

No system behaved this way.

### Problem 2: Patient ID Validation

All 4 systems hallucinated on "Patient 9999999999" question.

Main2's ID validation only checks the response:
```python
mentioned_eeg_ids = re.findall(r'\b\d{10}\b', llm_response)
context_eeg_ids = re.findall(r'patient (\d{10})', context)
```

The ID in the query is never checked. Should have been pre-LLM validation:

```python
query_ids = extract_ids(user_query)
available_ids = {m['eeg_id'] for m in metadata}
missing = query_ids - available_ids

if missing:
    return f"Patient ID {missing} not found in database"
```

### Problem 3: Misconception Correction

Wrong information "LPD = low-power discharge" was given. All 4 systems accepted it.

Main1 most interesting: tried to explain segments saying "high amplitude" with "low-power". Obvious contradiction but model did not notice.

Correct approach:
> "Actually, that's incorrect. LPD stands for Lateralized Periodic Discharges, not low-power discharge. The segments show high amplitude activity, which is opposite of low-power."

### Problem 4: Same Model Self-Validation

Critical error of Main3 and Main4. Same model both generates and validates.

Solution options:
1. Use different model (gemma:2b generate, llama3:70b validate)
2. Use external medical KB
3. Structured output + schema validation

---

## RECOMMENDATIONS

### Recommendation 1: Pre-LLM Validation Layer

Before calling LLM:
- Check patient IDs
- Determine query type (definition vs data query)
- Perform context sufficiency check

### Recommendation 2: Medical Knowledge Base Integration

Use external medical terminology database:
- UMLS (Unified Medical Language System)
- SNOMED CT
- Custom EEG terminology KB

Validate terms generated by LLM against KB.

### Recommendation 3: Hybrid Validation

```python
# Step 1: Rule-based checks
if patient_id not in metadata: reject
if definition_query and no_definition_in_context: explain

# Step 2: LLM validation (different model)
validation = stronger_model.validate(answer)

# Step 3: KB validation
medical_terms = extract_terms(answer)
for term in medical_terms:
    if not medical_kb.exists(term): flag
```

### Recommendation 4: Confidence Scores

Add confidence score to each answer:
- Definition in context → high confidence
- Inference from context → medium confidence
- Model knowledge → low confidence (not recommended for medical)

Show to user: "This answer is based on inference (medium confidence)"

### Recommendation 5: Prompt Engineering

Saying "Use your medical knowledge" is risky. Instead:
- Context-only mode
- Clear escalation: "Context insufficient, would you like me to search?"

---

## CONCLUSION

Four different RAG implementations were tested. Results:

**Main1 (Baseline):**
- Simplest, fastest
- Hallucination rate 80%
- Good starting point but not production-ready

**Main2 (Strict Grounding):**
- Post-processing added
- Hallucination detection with pattern matching
- Minimal improvement over Main1
- Hallucination rate still 80%

**Main3 (Self-Validating):**
- 4-step pipeline
- Self-validation approach
- Slowest (4x), most expensive
- Hallucination rate 100% (!!)
- False sense of security
- **Do not use**

**Main4 (Hybrid):**
- Clean version of Main3
- Best code quality
- But same fundamental problem
- Hallucination rate 80%

**General Observation:**
Sophisticated multi-step pipelines did not improve performance. In some cases it got worse (Main3).

Core problem: Model is not fabricating information not in context, it is making "inferences". Completely preventing this with prompt engineering appears difficult.

**Going Forward:**
- External validation (KB, different model) essential
- Add pre-LLM checks
- Implement confidence scoring
- Test specialized model for medical domain