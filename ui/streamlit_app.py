import streamlit as st
import requests

st.set_page_config(page_title="EEG RAG System", page_icon="🧠", layout="wide")

API_URL = "http://api-backend:8001/query"

def main():
    st.title("EEG RAG System")
    st.markdown("Query EEG recordings and receive AI-generated clinical insights.")

    st.sidebar.header("Settings")
    n_results = st.sidebar.slider("Segments to retrieve", 1, 10, 5)

    st.sidebar.subheader("Filters (Optional)")
    use_label_filter = st.sidebar.checkbox("Filter by label")
    label_filter = None
    if use_label_filter:
        label_options = ["Seizure", "GPD", "LPD", "LRDA", "GRDA", "Other", "unknown"]
        selected_label = st.sidebar.selectbox("Label", label_options)
        label_filter = {"expert_consensus": selected_label}

    filters = label_filter

    col1, col2 = st.columns([2, 3])
    with col1:
        st.subheader("Query")
        user_input = st.text_area("Enter your question:", placeholder="Example: Show EEG segments with seizure activity and high amplitude...", height=150)
        search_button = st.button("Analyze", type="primary", use_container_width=True)

    if search_button and user_input:
        with st.spinner("Connecting to backend... searching and analyzing..."):
            try:
                payload = {
                    "query": user_input,
                    "n_results": n_results,
                    "n_definitions": 3,
                    "n_videos": 2,
                    "filters": filters}

                response = requests.post(API_URL, json=payload, timeout=180)
                response.raise_for_status()

                data = response.json()

                if "error" in data:
                    st.error(f"Backend Error: {data['error']}")
                    st.stop()

                signal_results = data.get("retrieved_signal_segments", {}).get("documents", [[]])[0]
                document_results = data.get("retrieved_document_chunks", {}).get("documents", [[]])[0]
                transcript_results = data.get("retrieved_transcript_chunks", {}).get("documents", [[]])[0]

                all_results = signal_results + document_results + transcript_results

                llm_response = data.get("llm_response")

                if not all_results:
                    st.warning("No results found. Try adjusting your query or filters.")
                    st.stop()

                with col2:
                    st.success(f"Found {len(all_results)} relevant segments.")
                    st.subheader("AI Analysis")
                    st.markdown(llm_response)

                st.subheader("Retrieved EEG Segments for context")

                # Signal segments
                if signal_results:
                    st.markdown("**🔴 EEG Signal Segments**")
                    for i, doc in enumerate(signal_results):
                        with st.expander(f"Signal {i + 1}", expanded=(i < 2)):
                            st.write(doc)

                # Document chunks
                if document_results:
                    st.markdown("**📄 Medical Definitions**")
                    for i, doc in enumerate(document_results):
                        with st.expander(f"Definition {i + 1}", expanded=(i < 1)):
                            st.write(doc)

                # Transcript chunks
                if transcript_results:
                    st.markdown("**🎥 Expert Video Transcripts**")
                    for i, doc in enumerate(transcript_results):
                        with st.expander(f"Transcript {i + 1}", expanded=(i < 1)):
                            st.write(doc)

            except requests.exceptions.Timeout:
                st.error(
                    f"Request timed out (180s). The backend is likely overloaded or the query is too complex. Try again.")
            except requests.exceptions.RequestException as e:
                st.error(f"Connection error: Could not connect to API at {API_URL}.")
                st.error(f"Details: {e}")
            except Exception as e:
                st.error(f"An unexpected UI error occurred: {e}")

if __name__ == "__main__":
    main()