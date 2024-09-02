import pandas as pd

def preprocess_data():
    # List of dataset files to merge
    dataset_files = [
        "E:/PG-dissertation/dataset/07email_text.csv",
        "E:/PG-dissertation/dataset/05email_text.csv",
        "E:/PG-dissertation/dataset/06email_text.csv"
    ]

    # Load each dataset, print its shape, and collect them into a list
    dfs = []
    for file in dataset_files:
        df = pd.read_csv(file).drop_duplicates()
        print(f"Dataset {file} shape: {df.shape}")
        dfs.append(df)

    # Concatenate all datasets
    df = pd.concat(dfs, ignore_index=True)

    # Ensure the dataset has the necessary columns
    if 'label' not in df.columns or 'text' not in df.columns:
        raise ValueError("One of the datasets does not have the required 'label' and 'text' columns.")

    # Print the shape of the combined dataframe to check the size of the data
    print(f"Combined dataset shape: {df.shape}")

    # Remove rows with missing values in 'label' and 'text' columns
    df.dropna(subset=['label', 'text'], inplace=True)

    # Remove duplicate entries
    df.drop_duplicates(inplace=True)

    # Print the shape of the dataframe after cleaning to verify changes
    print(f"Cleaned dataset shape: {df.shape}")


    print("Proportion of phishing emails:", df['label'].mean())
    print("Data size is", df.shape)
    print("Data preprocessing completed.")

    # Save preprocessed data
    df.to_csv("E:/PG-dissertation/dataset/combined_preprocessed_test.csv", index=False)
    print("Preprocessed data saved to 'E:/PG-dissertation/dataset/combined_preprocessed_test.csv'")

if __name__ == "__main__":
    preprocess_data()