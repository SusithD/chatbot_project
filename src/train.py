from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from datasets import load_from_disk

def train_model(train_dataset_path, eval_dataset_path, output_dir="models/finetuned_gemma"):
    """
    Fine-tune a chatbot model on the provided dataset.
    """
    model_name = "Salesforce/codegen-350M-multi"  # Example

    # Authenticate and use model from Hugging Face
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Set pad_token to eos_token
    tokenizer.pad_token = tokenizer.eos_token

    def tokenize(batch):
        """
        Tokenizes the dialogues with padding and truncation, max length 512.
        """
        inputs = tokenizer(batch['dialogue'], padding=True, truncation=True, max_length=512)
        inputs['labels'] = inputs['input_ids'].copy()
        return inputs

    # Load preprocessed datasets
    train_dataset = load_from_disk(train_dataset_path)
    eval_dataset = load_from_disk(eval_dataset_path)

    # Apply tokenization
    train_dataset = train_dataset.map(tokenize, batched=True)
    eval_dataset = eval_dataset.map(tokenize, batched=True)

    # Use DataCollatorForSeq2Seq for dynamic padding
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=4,
        num_train_epochs=3,
        weight_decay=0.01,
        fp16=True,
        gradient_accumulation_steps=8,
        save_total_limit=2,
        dataloader_num_workers=4,
        logging_dir=f"{output_dir}/logs",
        logging_steps=500,
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Train the model
    trainer.train()

    # Save the fine-tuned model and tokenizer
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

if __name__ == "__main__":
    # Paths to preprocessed datasets
    train_dataset_path = "data/prepared_train"
    eval_dataset_path = "data/prepared_dev"

    # Train the model
    train_model(train_dataset_path, eval_dataset_path)
