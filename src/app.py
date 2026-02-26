import streamlit as st
from transformers import pipeline
import requests
import os
import urllib.parse

# --- Configuration ---
MODEL_PATH = "models/misinfo_llm"
# Make sure this is your valid key
API_KEY = "AIzaSyDuZlBM5m6pBNxFzbLXw_HSxecSFeZ1-HY" 

# --- Load Custom Model ---
@st.cache_resource
def load_custom_llm():
    if os.path.exists(MODEL_PATH):
        return pipeline("text-classification", model=MODEL_PATH, tokenizer=MODEL_PATH)
    return None

classifier = load_custom_llm()

# --- Google Fact Check Logic ---
def check_google_fact(claim):
    encoded_claim = urllib.parse.quote(claim)
    url = f"https://factchecktools.googleapis.com/v1alpha1/claims:search?query={encoded_claim}&key={API_KEY}"
    response = requests.get(url).json()
    
    if not response.get('claims'):
        simple_claim = " ".join(claim.split()[:3])
        encoded_simple = urllib.parse.quote(simple_claim)
        url = f"https://factchecktools.googleapis.com/v1alpha1/claims:search?query={encoded_simple}&key={API_KEY}"
        response = requests.get(url).json()
        
    return response.get('claims', [])

# --- UI Setup ---
st.set_page_config(page_title="Guardian AI", page_icon="🛡️")
st.title("🛡️ Guardian AI: Hybrid Fact-Checker")
st.markdown("Analyzing claims using **Google Knowledge** + **Custom Trained LLM (WELFake)**")

claim = st.text_input("Enter a news claim to verify:", placeholder="e.g., Dandelion tea cures cancer")

if st.button("Verify Claim", type="primary"):
    if claim:
        with st.spinner("Analyzing Layers..."):
            # 1. Check Google Knowledge Layer
            facts = check_google_fact(claim)
            
            # 2. Check Trained LLM Layer
            llm_result = classifier(claim)[0] if classifier else {"label": "LABEL_0", "score": 0}
            llm_label = "REAL" if llm_result['label'] == 'LABEL_1' else "FAKE"
            confidence = llm_result['score']

            # --- 3. HYBRID LOGIC (The Override) ---
            final_verdict = llm_label
            is_override = False
            
            if facts:
                rating_text = facts[0]['claimReview'][0].get('textualRating', "").lower()
                # Indicators that the claim is actually false
                if any(word in rating_text for word in ["false", "fake", "misleading", "myth", "no"]):
                    if llm_label == "REAL":
                        final_verdict = "FAKE"
                        is_override = True

            # --- Display Results ---
            st.divider()
            
            if final_verdict == "FAKE":
                st.error(f"### FINAL SYSTEM VERDICT: {final_verdict}")
            else:
                st.success(f"### FINAL SYSTEM VERDICT: {final_verdict}")

            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🤖 AI Style Analysis")
                st.write(f"Tone Analysis: **{llm_label}**")
                st.progress(confidence)
                st.caption("LLM analyzes linguistic patterns.")

            with col2:
                st.subheader("🔍 Google Knowledge")
                if facts:
                    # facts[0] is the main result
                    # 'text' is the claim found, 'title' is the expert explanation
                    f = facts[0]
                    review = f['claimReview'][0]
                    
                    st.warning(f"**Expert Rating: {review['textualRating']}**")
                    # THIS IS THE LINE THAT WAS MISSING:
                    st.info(f"**Explanation:** {review.get('title', 'Check source for details.')}")
                    st.write(f"**Source:** {review['publisher']['name']}")
                else:
                    st.warning("No direct match in Knowledge Base.")

            # Final Explanation for the Professor
            if is_override:
                st.warning("⚠️ **Hybrid Logic Note:** The AI model was fooled by the professional tone, but the Google Knowledge Layer provided a factual override based on the evidence shown above.")
    else:
        st.warning("Please enter a claim first!")