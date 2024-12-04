from transformers import AutoModelForQuestionAnswering, AutoTokenizer
from datasets import load_from_disk
from transformers import Trainer, TrainingArguments

def evaluate_model():
    model_name = "distilbert-base-uncased"
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load preprocessed datasets
    eval_data = load_from_disk('data/prepared_dev')  # Load dev data

    def tokenize(batch):
        return tokenizer(batch['question'], padding=True, truncation=True, max_length=512)

    # Tokenize the evaluation dataset
    eval_data = eval_data.map(tokenize, batched=True)

    # Training arguments for evaluation
    training_args = TrainingArguments(
        output_dir="models/evaluation_results",  # Save evaluation results
        per_device_eval_batch_size=16,
    )

    # Initialize the Trainer for evaluation
    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=eval_data,  # Use eval_data
        tokenizer=tokenizer,
    )

    # Evaluate the model
    results = trainer.evaluate()
    print(f"Evaluation results: {results}")

if __name__ == "__main__":
    evaluate_model()
