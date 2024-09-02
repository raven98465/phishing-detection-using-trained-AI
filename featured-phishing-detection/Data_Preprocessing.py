import pandas as pd
from transformers import BertTokenizerFast
import json
from sklearn.model_selection import train_test_split

def preprocessing():
    # Define dataset file paths
    dataset_files = [
        "E:/PG-dissertation/dataset/TREC_05.csv",
        "E:/PG-dissertation/dataset/TREC_06.csv",
        "E:/PG-dissertation/dataset/TREC_07.csv"
    ]

    dfs = []
    for file in dataset_files:
        try:
            df = pd.read_csv(file, engine='python', on_bad_lines='skip').drop_duplicates()
            print(f"Dataset {file} shape: {df.shape}")
            dfs.append(df)
        except Exception as e:
            print(f"Error reading {file}: {e}")

    if not dfs:
        raise ValueError("No valid datasets found to concatenate.")

    # Combine datasets
    df = pd.concat(dfs, ignore_index=True)
    print(f"Combined dataset shape: {df.shape}")

    # Clean label column
    print("Unique values in label column before cleaning:", df['label'].unique())
    df.dropna(subset=['label', 'body', 'subject', 'sender', 'receiver'], inplace=True)
    df['label'] = pd.to_numeric(df['label'], errors='coerce')
    df.dropna(subset=['label'], inplace=True)
    df['label'] = df['label'].astype(int)
    print("Unique values in label column after cleaning:", df['label'].unique())
    print("Proportion of phishing emails:", df['label'].mean())
    print("Data size is", df.shape)
    print("Data preprocessing completed.")

    # Initialise tokeniser
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased', use_fast=True, low_cpu_mem_usage=False)
    print("Tokeniser initialised.")

    # Ensure all values in 'body' and 'subject' columns are strings and handle missing values
    df['body'] = df['body'].fillna("").astype(str)
    df['subject'] = df['subject'].fillna("").astype(str)

    # Tokenise 'body' and 'subject'
    print("Tokenising body and subject...")
    df['tokenized_body'] = df['body'].apply(
        lambda x: tokenizer(x, truncation=True, padding='max_length', max_length=512))
    df['tokenized_subject'] = df['subject'].apply(
        lambda x: tokenizer(x, truncation=True, padding='max_length', max_length=64))

    # Encode categorical features like sender and receiver
    print("Encoding categorical features...")
    df['encoded_sender'] = df['sender'].astype('category').cat.codes
    df['encoded_receiver'] = df['receiver'].astype('category').cat.codes

    # Split the data into training and testing sets while maintaining the label distribution
    _, test_df = train_test_split(df, test_size=0.1, stratify=df['label'], random_state=42)
    train_df, _ = train_test_split(_, test_size=0.5, stratify=_['label'], random_state=42)

    print(f"Training set shape: {train_df.shape}")
    print(f"Test set shape: {test_df.shape}")

    # 计算钓鱼邮件的百分比
    train_phishing_percent = train_df['label'].mean() * 100
    test_phishing_percent = test_df['label'].mean() * 100

    print(f"Training set phishing email percentage: {train_phishing_percent:.2f}%")
    print(f"Test set phishing email percentage: {test_phishing_percent:.2f}%")

    # Create a list of dictionaries to save as JSON
    def create_data_list(df):
        data = []
        for _, row in df.iterrows():
            item = {
                'input_ids': row['tokenized_body']['input_ids'],
                'attention_mask': row['tokenized_body']['attention_mask'],
                'tokenized_subject': row['tokenized_subject']['input_ids'],
                'encoded_sender': row['encoded_sender'],
                'encoded_receiver': row['encoded_receiver'],
                'label': row['label']
            }
            data.append(item)
        return data

    train_data = create_data_list(train_df)
    test_data = create_data_list(test_df)

    # Save preprocessed data
    with open("E:/PG-dissertation/dataset/train_data.json", 'w') as f:
        json.dump(train_data, f)
    with open("E:/PG-dissertation/dataset/test_data.json", 'w') as f:
        json.dump(test_data, f)

    print("Feature engineering completed and data saved.")
    return df

if __name__ == "__main__":
    preprocessing()