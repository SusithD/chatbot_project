from transformers import AutoModelForQuestionAnswering, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_from_disk

def train_model():
    model_name = "distilbert-base-uncased"
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Correct the path to the preprocessed dataset
    train_data = load_from_disk("data/prepared_train")  # Load from 'data/prepared_train'
    eval_data = load_from_disk('data/prepared_dev')  # Load from 'data/prepared_dev'

    def tokenize(batch):
        # Tokenize questions and answers, including start and end positions
        tokenized_batch = tokenizer(batch['question'], padding=True, truncation=True, max_length=512)
        
        # Handle start and end positions during tokenization (mapping answers to tokenized indices)
        start_positions = []
        end_positions = []
        for i, answer in enumerate(batch['answer']):
            start_pos = batch['start_positions'][i]  # Get the start position for the answer
            end_pos = batch['end_positions'][i]  # Get the end position for the answer

            # Adjust the positions based on tokenized input (if answer spans multiple tokens)
            start_positions.append(start_pos)
            end_positions.append(end_pos)
        
        tokenized_batch['start_positions'] = start_positions
        tokenized_batch['end_positions'] = end_positions
        return tokenized_batch

    # Tokenize both training and evaluation datasets
    train_data = train_data.map(tokenize, batched=True)
    eval_data = eval_data.map(tokenize, batched=True)

    # Training arguments
    training_args = TrainingArguments(
        output_dir="models/finetuned_model",  # Save the model here
        evaluation_strategy="epoch",  # Use evaluation at the end of each epoch
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
    )

    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,  # Use 'train_data' here
        eval_dataset=eval_data,  # Use 'eval_data' here
        tokenizer=tokenizer,
    )

    # Train the model
    trainer.train()

    # Save the fine-tuned model and tokenizer
    model.save_pretrained("models/finetuned_model")  # Save the fine-tuned model
    tokenizer.save_pretrained("models/finetuned_model")  # Save the tokenizer

if __name__ == "__main__":
    train_model()
