import streamlit as st
from transformers import pipeline
import requests
import os

# --- Configuration ---
MODEL_PATH = "models/misinfo_llm"
API_KEY = "AIzaSyDuZlBM5m6pBNxFzbLXw_HSxecSFeZ1-HY"  # Replace with your actual key

# --- Load Custom Model ---
@st.cache_resource
def load_custom_llm():
    if os.path.exists(MODEL_PATH):
        # We use a pipeline for easy 'text-classification'
        return pipeline("text-classification", model=MODEL_PATH, tokenizer=MODEL_PATH)
    return None

classifier = load_custom_llm()

# --- Google Fact Check Logic ---
def check_google_fact(claim):
    # Try the full claim first
    url = f"https://factchecktools.googleapis.com/v1alpha1/claims:search?query={claim}&key={API_KEY}"
    response = requests.get(url).json()
    
    # If no results, try a "Simplified" version (the first 3 words)
    if not response.get('claims'):
        simple_claim = " ".join(claim.split()[:3])
        url = f"https://factchecktools.googleapis.com/v1alpha1/claims:search?query={simple_claim}&key={API_KEY}"
        response = requests.get(url).json()
        
    return response.get('claims', [])

# --- UI Setup ---
st.set_page_config(page_title="Guardian AI", page_icon="🛡️")
st.title("🛡️ Guardian AI: Hybrid Fact-Checker")
st.markdown("Analyzing claims using **Google Knowledge** + **Custom Trained LLM (WELFake)**")

claim = st.text_input("Enter a news claim to verify:", placeholder="e.g., NASA found life on Mars")

if st.button("Verify Claim", type="primary"):
    if claim:
        with st.spinner("Analyzing..."):
            # 1. Check Google Knowledge
            facts = check_google_fact(claim)
            
            # 2. Check Trained LLM Intelligence
            llm_result = classifier(claim)[0] if classifier else {"label": "N/A", "score": 0}
            label = "REAL" if llm_result['label'] == 'LABEL_1' else "FAKE"
            confidence = llm_result['score']

            # --- Display Results ---
            st.divider()
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🤖 Custom LLM Analysis")
                color = "green" if label == "REAL" else "red"
                st.markdown(f"Verdict: :{color}[**{label}**]")
                st.metric("Model Confidence", f"{confidence:.2%}")
                st.caption("Based on patterns learned from 72,000 news articles.")

            with col2:
                st.subheader("🔍 Google Fact Check")
                if facts:
                    for f in facts[:2]:
                        st.write(f"**Claim:** {f['text']}")
                        st.info(f"**Rating:** {f['claimReview'][0]['textualRating']}")
                else:
                    st.warning("No direct match found in Google's database.")

            # Summary Logic
            if not facts and label == "FAKE":
                st.error("⚠️ HIGH RISK: No official debunking found, but language patterns suggest this is MISINFORMATION.")
    else:
        st.warning("Please enter a claim first!")
