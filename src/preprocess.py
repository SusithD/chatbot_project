import pandas as pd
import os
from sklearn.model_selection import train_test_split
from datasets import Dataset

def preprocess_data(file_path):
    """
    Load and preprocess the dataset from a CSV file.
    The dataset should have `Instruction`, `Input`, and `Output` columns.
    """
    # Load the CSV file
    data = pd.read_csv(file_path)

    # Combine Instruction and Input into a single text prompt for the chatbot
    data['dialogue'] = data.apply(
        lambda row: f"User: {row['Instruction']} {row['Input']}\nChatbot: {row['Output']}", axis=1
    )

    # Split into training and evaluation datasets
    train_data, eval_data = train_test_split(data, test_size=0.2, random_state=42)

    # Convert to Hugging Face Dataset format
    train_dataset = Dataset.from_pandas(train_data[['dialogue']])
    eval_dataset = Dataset.from_pandas(eval_data[['dialogue']])

    return train_dataset, eval_dataset

if __name__ == "__main__":
    # File path to the dataset
    dataset_path = "data/Python Programming Questions Dataset.csv"

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")

    # Preprocess the data
    train_dataset, eval_dataset = preprocess_data(dataset_path)

    # Save preprocessed datasets
    train_dataset.save_to_disk("data/prepared_train")
    eval_dataset.save_to_disk("data/prepared_dev")

    print(f"Preprocessed and saved train and eval datasets.")
