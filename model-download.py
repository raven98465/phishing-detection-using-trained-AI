from transformers import AutoTokenizer, AutoModelForSequenceClassification
import requests
import os

# Model Name List
model_names = [
    'ealvaradob/bert-finetuned-phishing',
    'dima806/phishing-email-detection',
    'kamikaze20/phishing-email-detection_final_2',
    'huynq3Cyradar/bert-large-finetuned-phishing-url-version',
    'aibot123/phishing-detection-body-only'
]

# Increase timeout and retry times
session = requests.Session()
adapter = requests.adapters.HTTPAdapter(max_retries=5)
session.mount("https://", adapter)
timeout = (10, 30)  # Connection and read timeout

# Retrieve the directory where the current script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Download and save models
for model_name in model_names:
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=False)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, use_auth_token=False)

    # Dynamically save the path
    model_save_path = os.path.join(script_dir, 'AI model', model_name.split("/")[-1])

    # Create a save path (if it does not exist)
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)

    # Save tokenizer and model
    tokenizer.save_pretrained(model_save_path)
    model.save_pretrained(model_save_path)

print("All models have been downloaded and saved to the local directory E:/PG-dissertation/AI model")