import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch
import time
import os

def check_cuda_device():
    if torch.cuda.is_available():
        device = torch.device("cuda:0")  # Ensure that it uses the first CUDA device
        device_name = torch.cuda.get_device_name(0)
        print(f"Using device: {device} - {device_name}")
    else:
        print("CUDA is not available. Using CPU.")
        device = torch.device("cpu")

    return device
def test_model():
    device = check_cuda_device()

    # Load preprocessed data
    df = pd.read_csv("E:/PG-dissertation/dataset/combined_preprocessed_test.csv")
    print(f"Loaded preprocessed dataset shape: {df.shape}")

    # Ensure results directory exists
    results_dir = "E:/PG-dissertation/dataset/test-result"
    os.makedirs(results_dir, exist_ok=True)
    print("directory set")

    # Define local model path
    local_model_paths = {
        'dima806/phishing-email-detection': 'E:/PG-dissertation/AI model/phishing-email-detection',
        'ealvaradob/bert-finetuned-phishing': 'E:/PG-dissertation/AI model/bert-finetuned-phishing',
        'kamikaze20/phishing-email-detection_final_2': 'E:/PG-dissertation/AI model/phishing-email-detection_final_2'
    }

    # Test each model
    for model_name, local_path in local_model_paths.items():
        # Load model and tokeniser from local path
        tokenizer = AutoTokenizer.from_pretrained(local_path)
        model = AutoModelForSequenceClassification.from_pretrained(local_path).to(device)

        # Store current model's test results
        correct_predictions = 0
        total_predictions = 0
        all_labels = []
        all_predictions = []

        # Start timer
        start_time = time.time()

        # Classify each email in batches
        batch_size = 32  # Increase batch size if GPU memory allows
        for i in range(0, len(df), batch_size):
            batch = df['text'][i:i+batch_size].tolist()
            labels = df['label'][i:i+batch_size].tolist()
            inputs = tokenizer(batch, truncation=True, padding=True, max_length=512, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(**inputs)
                scores = torch.nn.functional.softmax(outputs.logits, dim=-1)
                predicted_labels = scores.argmax(dim=1).cpu().numpy()

                correct_predictions += (predicted_labels == labels).sum()
                total_predictions += len(labels)

                all_labels.extend(labels)
                all_predictions.extend(predicted_labels)

            # Print progress for every 100 emails processed
            if (i + batch_size) % 100 == 0 or (i + batch_size) >= len(df):
                accuracy = correct_predictions / total_predictions
                elapsed_time = time.time() - start_time
                print(f"Processed {min(i + batch_size, len(df))} emails, current model: {model_name}")
                print(f"Current accuracy: {accuracy:.4f}, Elapsed time: {elapsed_time:.2f} seconds")

        # Print final results for the current model
        final_accuracy = correct_predictions / total_predictions
        print(f"Model {model_name} testing completed")
        print(f"Final accuracy for {model_name}: {final_accuracy:.4f}")

        # Save labels and predictions for evaluation
        safe_model_name = model_name.replace('/', '_')  # Replace slashes with underscores
        labels_predictions_df = pd.DataFrame({'label': all_labels, 'prediction': all_predictions})
        labels_predictions_df.to_csv(f"{results_dir}/{safe_model_name}_labels_predictions.csv", index=False)
        print(f"Model {model_name} labels and predictions saved to '{results_dir}/{safe_model_name}_labels_predictions.csv'")

if __name__ == "__main__":
    test_model()