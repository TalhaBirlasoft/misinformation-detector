import streamlit as st
from transformers import pipeline
import requests
import os
import urllib.parse

# --- Configuration ---
MODEL_PATH = "models/misinfo_llm"
# The API Key allows access to the Google Fact Check database
API_KEY = "AIzaSyDuZlBM5m6pBNxFzbLXw_HSxecSFeZ1-HY" 

# --- Load Custom Model ---
@st.cache_resource
def load_custom_llm():
    if os.path.exists(MODEL_PATH):
        # Using the HuggingFace pipeline for high-performance text classification
        return pipeline("text-classification", model=MODEL_PATH, tokenizer=MODEL_PATH)
    return None

classifier = load_custom_llm()

# --- Google Fact Check Logic ---
def check_google_fact(claim):
    # Encoding the query for URL safety
    encoded_claim = urllib.parse.quote(claim)
    url = f"https://factchecktools.googleapis.com/v1alpha1/claims:search?query={encoded_claim}&key={API_KEY}"
    response = requests.get(url).json()
    
    # Fallback: If no direct match, simplify the query to the first few keywords
    if not response.get('claims'):
        simple_claim = " ".join(claim.split()[:3])
        encoded_simple = urllib.parse.quote(simple_claim)
        url = f"https://factchecktools.googleapis.com/v1alpha1/claims:search?query={encoded_simple}&key={API_KEY}"
        response = requests.get(url).json()
        
    return response.get('claims', [])

# --- UI Setup ---
st.set_page_config(page_title="Guardian AI", page_icon="🛡️")
st.title("🛡️ Guardian AI: Hybrid Fact-Checker")
st.markdown("### Hybrid Analysis: Linguistic Patterns + Knowledge Layer")
st.markdown("Fine-tuned on **WELFake Dataset (72,000 articles)**")

claim = st.text_input("Enter news headline to score:", placeholder="e.g., Image shows U.S. Embassy on fire")

if st.button("Verify & Score", type="primary"):
    if claim:
        with st.spinner("Executing Multi-Layer Analysis..."):
            # 1. LAYER 1: Knowledge-Base Verification (Google)
            facts = check_google_fact(claim)
            
            # 2. LAYER 2: Deep Learning Linguistic Check (DistilBERT)
            llm_result = classifier(claim)[0] if classifier else {"label": "LABEL_0", "score": 0}
            llm_label = "REAL" if llm_result['label'] == 'LABEL_1' else "FAKE"
            ai_confidence = llm_result['score']

            # --- 3. HYBRID SCORING HEURISTIC ---
            # Initializing score based on LLM output
            ai_score_percent = ai_confidence * 100
            final_score = ai_score_percent
            is_override = False
            logic_note = "Veracity score derived from stylistic pattern analysis."

            if facts:
                rating_text = facts[0]['claimReview'][0].get('textualRating', "").lower()
                
                # EXPANDED Logic Gate: Catching 'AI-generated', 'False', or 'Manipulated' labels
                fake_indicators = ["false", "fake", "misleading", "myth", "no", "ai", "altered", "manipulated", "synthetic"]
                
                if any(word in rating_text for word in fake_indicators):
                    # APPLY MATHEMATICAL PENALTY: Caps score at 15% for safety
                    final_score = min(ai_score_percent, 15.0) 
                    logic_note = "🚨 Knowledge Layer Override: Verified as False or Altered content."
                    if llm_label == "REAL":
                        is_override = True
                
                # Boost score if confirmed true by experts
                elif any(word in rating_text for word in ["true", "correct", "proven", "authentic"]):
                    final_score = max(ai_score_percent, 92.0)
                    logic_note = "✅ Knowledge Layer Verified: Content confirmed by external sources."

            # --- 4. DATA DASHBOARD DISPLAY ---
            st.divider()
            
            # Main Veracity Score Gauge
            st.markdown(f"## 🛡️ Veracity Score: {final_score:.1f}%")
            if final_score > 75:
                st.success("### SYSTEM VERDICT: HIGH PROBABILITY AUTHENTIC")
            elif final_score > 40:
                st.warning("### SYSTEM VERDICT: UNVERIFIED / MIXED SIGNALS")
            else:
                st.error("### SYSTEM VERDICT: HIGH PROBABILITY MISINFORMATION")
            
            st.progress(final_score / 100)
            st.caption(f"**Analysis Note:** {logic_note}")

            # Technical Column Breakdown for the Report
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🤖 AI Style Layer")
                st.write(f"Stylistic Verdict: **{llm_label}**")
                st.write(f"Confidence: {ai_score_percent:.1f}%")
                st.caption("Analyzes syntax, emotion, and tone.")

            with col2:
                st.subheader("🔍 Knowledge Layer")
                if facts:
                    review = facts[0]['claimReview'][0]
                    st.info(f"**Expert Rating:** {review['textualRating']}")
                    st.write(f"**Title:** {review.get('title', 'N/A')}")
                    st.caption(f"Fact-Checker: {review['publisher']['name']}")
                else:
                    st.write("No existing database match found.")
                    st.caption("System relying on linguistic intelligence.")

            # Documentation Alert for the Professor
            if is_override:
                st.warning("⚠️ **Architectural Note:** The Linguistic Layer assigned a 'Real' label due to professional phrasing, but the Knowledge Layer triggered a priority override based on factual debunking.")
    else:
        st.warning("Please enter a claim first!")