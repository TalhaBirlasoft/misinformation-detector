# 🛡️ Guardian AI: Hybrid Misinformation Detection System

Guardian AI is an advanced, hybrid fact-checking platform designed to bridge the gap between **Linguistic Intelligence** and **Factual Knowledge**. 

## 🚀 The Core Innovation: Hybrid Verification
Most AI models for fake news detection are easily fooled by "Sophisticated Misinformation"—lies written in a formal, academic, or scientific tone. This system implements a **Dual-Layer Defense** to ensure reliability:

1.  **Linguistic Layer (The "Brain"):** A custom-trained **DistilBERT** Transformer model (fine-tuned on the 72,000-article WELFake dataset) that identifies the stylistic patterns of misinformation.
2.  **Knowledge Layer (The "Truth"):** Integration with the **Google Fact Check API** to provide external validation against established human-verified facts.
3.  **The Override Logic:** If the LLM is misled by a professional tone (e.g., medical misinformation), the system uses the Google API as the "Ground Truth" to override the verdict.

## 📊 Model Performance & Rigor
- **Base Model:** DistilBERT (Fine-tuned locally)
- **Dataset:** WELFake (72,134 labeled records)
- **Accuracy:** ~98.40% (Test Set)
- **Hardware Optimization:** Built for local execution on Apple Silicon (MPS/CPU) to ensure data privacy and speed.

### Addressing High-Confidence Anomalies
During testing, we observed that LLMs can exhibit **High Confidence Overgeneralization**. For example, a claim like *"Dandelion tea cures cancer"* might be flagged as REAL by the LLM because it uses a professional, non-sensationalist structure. Our **Hybrid Logic** successfully corrects these edge cases by prioritizing API data over stylistic analysis.

## 🛠️ Project Structure
- `src/app.py`: The Streamlit dashboard and hybrid logic engine.
- `src/train_real_data.py`: The pipeline used for model training and fine-tuning.
- `data/`: (Local Only) Contains the WELFake dataset.
- `models/`: (Local Only) Contains the saved DistilBERT model weights.
- `requirements.txt`: Necessary Python dependencies.

## 💻 Installation & Setup

### 1. Environment Setup
```bash
# Clone the repository
git clone <your-repo-link>
cd <your-repo-folder>

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
Base Model: DistilBERT (Transformer-based LLM)

##model specification 

Dataset: WELFake (72,134 labeled articles)

Training Split: 80% Train / 20% Test

Final Test Accuracy: 98.40%

Hybrid Logic: Includes a Google Fact Check API override to correct "Sophisticated Misinformation."


