import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def evaluate_result():
    # Path to results directory
    results_dir = "E:/PG-dissertation/dataset/test-result/"

    # List of model names
    model_names = [
        'dima806_phishing-email-detection',
        'ealvaradob_bert-finetuned-phishing',
        'kamikaze20_phishing-email-detection_final_2'
    ]

    # Dictionary to store evaluation metrics
    evaluation_metrics = {}

    # Evaluate each model
    for model_name in model_names:
        # Load labels and predictions
        labels_predictions_df = pd.read_csv(f"{results_dir}/{model_name}_labels_predictions.csv")
        labels = labels_predictions_df['label']
        predictions = labels_predictions_df['prediction']

        # Calculate metrics
        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')

        # Store metrics
        evaluation_metrics[model_name] = {
            'eval_accuracy': accuracy,
            'eval_precision': precision,
            'eval_recall': recall,
            'eval_f1': f1,
        }

        # Print metrics
        print(f"Evaluation metrics for {model_name}:")
        for metric, value in evaluation_metrics[model_name].items():
            print(f"{metric}: {value}")

    # Save evaluation metrics to CSV
    evaluation_metrics_df = pd.DataFrame.from_dict(evaluation_metrics, orient='index')
    evaluation_metrics_df.to_csv(f"{results_dir}/evaluation_metrics.csv")
    print(f"Evaluation metrics saved to '{results_dir}/evaluation_metrics.csv'")

if __name__ == "__main__":
    evaluate_result()