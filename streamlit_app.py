import streamlit as st
import requests
import json
from typing import List

# =====================================================
# CONFIGURATION
# =====================================================
API_URL = "http://localhost:8000/search"  # Your FastAPI endpoint

st.set_page_config(
    page_title="Hybrid Search Demo",
    page_icon="🔍",
    layout="wide"
)

# =====================================================
# UI HEADER
# =====================================================
st.title("🔍 Hybrid Search Demo (Elasticsearch + SentenceTransformer)")
st.markdown(
    """
    This Streamlit app interacts with your **FastAPI hybrid search API**.  
    It performs BM25 or vector-based hybrid searches over your HS code index.
    """
)

# =====================================================
# INPUT FORM
# =====================================================
with st.form("search_form"):
    query = st.text_input("Enter your search query:", placeholder="e.g. aluminium sheets")
    size = st.slider("Number of results", min_value=1, max_value=50, value=10)
    use_vector = st.checkbox("Use Vector (Hybrid) Search", value=True)
    alpha = st.slider("Vector weight (alpha)", 0.0, 1.0, 0.5, step=0.1)
    submitted = st.form_submit_button("Run Search")

# =====================================================
# HELPER FUNCTIONS
# =====================================================
def search_api(query: str, size: int, use_vector: bool, alpha: float):
    """Send a search request to the FastAPI backend."""
    payload = {
        "query": query,
        "size": size,
        "use_vector": use_vector,
        "alpha": alpha,
        "filter": {
            "trading_types": [],
            "in_vehicle_ids": [],
            "out_vehicle_ids": []
        }
    }

    try:
        response = requests.post(API_URL, json=payload)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Request failed: {e}")
        return None

# =====================================================
# RUN SEARCH AND DISPLAY RESULTS
# =====================================================
if submitted:
    if not query.strip():
        st.warning("Please enter a query.")
    else:
        with st.spinner("Searching..."):
            result = search_api(query, size, use_vector, alpha)

        if result:
            st.success(f"✅ Found {result.get('total-hits', 0)} hits")

            hits: List[dict] = result.get("Ranked-objects", [])
            if not hits:
                st.info("No results found.")
            else:
                # Create expandable cards for each hit
                for i, hit in enumerate(hits, start=1):
                    with st.expander(f"#{i} — {hit.get('name_az_d4', hit.get('code', 'No Name'))}"):
                        st.markdown(f"**Code:** `{hit.get('code')}`")
                        st.markdown(f"**Score:** {hit.get('score'):.4f}" if hit.get('score') else "")
                        st.markdown(f"**Path:** {hit.get('Path') or '—'}")

                        tradings = hit.get("tradings", [])
                        if tradings:
                            st.markdown("**Tradings:**")
                            for t in tradings:
                                st.write(f"- {t.get('tradeName')} ({t.get('tradeType')})")
                        else:
                            st.markdown("*No trading info available.*")
        else:
            st.error("❌ No valid response received from the API.")

# =====================================================
# FOOTER
# =====================================================
st.markdown("---")
st.caption("© 2025 Hybrid Search Demo — FastAPI + Streamlit + Elasticsearch")
