from transformers import AutoModelForQuestionAnswering, AutoTokenizer
import torch

class Chatbot:
    def __init__(self):
        model_name = "models/finetuned_model"  # Path to the fine-tuned model
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def get_answer(self, question, context):
        # Tokenize input question and context
        inputs = self.tokenizer(question, context, return_tensors='pt', truncation=True, padding=True, max_length=512)
        
        # Get model predictions
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Extract answer start and end positions
        start_scores, end_scores = outputs.start_logits, outputs.end_logits

        # Find the most likely start and end of the answer
        start_index = torch.argmax(start_scores)
        end_index = torch.argmax(end_scores)

        # Convert tokens to answer string
        answer_tokens = inputs['input_ids'][0][start_index:end_index + 1]
        answer = self.tokenizer.decode(answer_tokens, skip_special_tokens=True)

        return answer
