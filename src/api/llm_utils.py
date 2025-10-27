"""
Helpers for building prompts and calling the Ollama LLM.
"""
import requests
import logging
from infrastructure.conf.interfaces import LLMConfigInterface

logger = logging.getLogger(__name__)

def call_llm(llm_config: LLMConfigInterface, prompt: str) -> str:
    """Wrapper for the Ollama API call."""

    model_name = llm_config.get_llm_model_name()
    endpoint = llm_config.get_ollama_endpoint()
    options = llm_config.get_llm_options().copy()

    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "options": options}

    try:
        response = requests.post(f"{endpoint.rstrip('/')}/api/generate", json=payload, timeout=120)
        response.raise_for_status()
        response_json = response.json()
        answer = response_json.get("response", "").strip()

        if not answer:
            logger.warning("Empty LLM response")
            raise RuntimeError("LLM returned empty response")

        return answer

    except Exception as e:
        logger.error(f"LLM call failed: {e}", exc_info=True)
        raise RuntimeError(f"LLM error: {e}")

def build_prompt(query: str, query_type: str, signal_ctx: str, doc_ctx: str, transc_ctx: str) -> str:
    """Formats the RAG prompt"""

    # TODO: A chunking based on token would be better
    if query_type == 'reasoning':
        return f"""Expert reasoning from Dr. Hirsch:
{transc_ctx}

Reference definition:
{doc_ctx[:300]}

Patient example:
{signal_ctx[:300]}

Question: {query}

Answer (use expert's reasoning steps):"""

    elif query_type == 'definition':
        return f"""Medical definition (ACNS 2021):
{doc_ctx}

Clinical context from expert:
{transc_ctx[:300]}

Question: {query}

Answer (define term clearly):"""

    elif query_type == 'patient_data':
        return f"""Patient data:
{signal_ctx}

Reference definition:
{doc_ctx[:300]}

Question: {query}

Answer (describe findings):"""

    else:  # general
        return f"""Medical reference: {doc_ctx[:400]}

Expert guidance: {transc_ctx[:400]}

Patient data: {signal_ctx[:400]}

Q: {query}

A:"""

def run_rag_chain(llm_config: LLMConfigInterface,
                  query: str,
                  query_type: str,
                  signal_docs: list[str],
                  document_docs: list[str],
                  transcript_docs: list[str]) -> tuple[str, dict]:
    """Main RAG logic: build prompt, get answer, validate"""

    doc_ctx = "\n".join(document_docs[:1])[:800] if document_docs else ""
    signal_ctx = "\n".join(signal_docs[:2])[:500] if signal_docs else ""
    transc_ctx = "\n".join(transcript_docs[:2])[:700] if transcript_docs else ""

    prompt = build_prompt(query, query_type, signal_ctx, doc_ctx, transc_ctx)

    answer = call_llm(llm_config, prompt)

    validation = {
        "passed": len(answer) > 50,
        "confidence": "HIGH" if len(answer) > 150 else "MEDIUM",
        "query_type": query_type,
        "sources_used": {
            "video": len(transc_ctx) > 0,
            "medical": len(doc_ctx) > 0,
            "eeg": len(signal_ctx) > 0}}

    return answer, validation