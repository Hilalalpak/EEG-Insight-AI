JUDGE_PROMPT = """
[Your Task]
Your role is to act as an expert evaluator for a RAG (Retrieval-Augmented Generation) system.
Your goal is to score the "Model's Response" on a scale of 1 to 5, based on its accuracy and completeness
compared to the "Ideal Answer".
Please ensure your response is ONLY in the required JSON format.

[Scoring Guide]
1: Completely incorrect, irrelevant, or hallucinatory.
2: Partially correct, but misses significant details or context.
3. Correct, but superficial or not as clear/concise as the ideal answer.
4: Very good. Correct and answers the question fully.
5: Excellent. The answer is as comprehensive, accurate, and clear as the ideal answer.

[Inputs for Review]
Question: {question}
Ideal Answer (Reference): {ideal_answer}
Model's Response (To be scored): {model_answer}

[Required Output Format]
Provide your evaluation *only* as a single JSON object with two keys: "score" and "reasoning".
Example:
{{"score": 1, "reasoning": "The model's answer is a hallucination and factually incorrect."}}
"""