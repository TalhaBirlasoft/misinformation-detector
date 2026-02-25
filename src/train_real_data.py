import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset

# 1. Load the REAL WELFake dataset
if not os.path.exists('data/WELFake.csv'):
    print("Error: data/WELFake.csv not found. Please download it first.")
    exit()

print("Loading WELFake dataset...")
df = pd.read_csv('data/WELFake.csv')
df = df.dropna(subset=['title', 'text', 'label'])
df['content'] = df['title'] + " " + df['text']

# IMPORTANT: 1000 samples for a fast MacBook demo. 
# Use 5000+ if you have a few hours for better accuracy.
df_sample = df.sample(10000, random_state=42) 

# 2. THE GOLDEN RULE: Train/Test Split (80% Train, 20% Test)
train_df, test_df = train_test_split(df_sample, test_size=0.2, random_state=42)

train_dataset = Dataset.from_pandas(train_df[['content', 'label']])
test_dataset = Dataset.from_pandas(test_df[['content', 'label']])

# 3. Tokenization (DistilBERT LLM)
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_func(examples):
    return tokenizer(examples["content"], padding="max_length", truncation=True)

print("Tokenizing data...")
train_tokenized = train_dataset.map(tokenize_func, batched=True)
test_tokenized = test_dataset.map(tokenize_func, batched=True)

# 4. Metrics Function (For the Accuracy Assessment)
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(labels, predictions)}

# 5. Model & Training Configuration
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

training_args = TrainingArguments(
    output_dir="./models/results",
    eval_strategy="epoch",  # Run accuracy test after each epoch
    save_strategy="epoch",
    num_train_epochs=1,           # 1 Epoch is enough for a demo
    per_device_train_batch_size=4, # Keep low for MacBook Air RAM
    weight_decay=0.01,
    logging_dir='./logs',
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized,
    eval_dataset=test_tokenized,  # The "Final Exam"
    compute_metrics=compute_metrics
)

# 6. Start Training
print("Starting Fine-tuning...")
trainer.train()

# 7. Final Evaluation
print("\n" + "="*30)
print("FINAL ML ACCURACY ASSESSMENT")
print("="*30)
eval_results = trainer.evaluate()
print(f"Accuracy on unseen data: {eval_results['eval_accuracy']:.2%}")
print("="*30)

# 8. Save the Trained Brain
os.makedirs("models/misinfo_llm", exist_ok=True)
model.save_pretrained("models/misinfo_llm")
tokenizer.save_pretrained("models/misinfo_llm")
print("\nSuccess! Custom LLM trained and saved to models/misinfo_llm")
