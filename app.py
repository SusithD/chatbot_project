from flask import Flask, request, render_template
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline

app = Flask(__name__)

# Load the fine-tuned model
model_name = "models/finetuned_model"
model = AutoModelForQuestionAnswering.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    question = request.form['question']
    context = "Provide a context from your data or user input."
    result = qa_pipeline(question=question, context=context)
    return {"answer": result['answer']}

if __name__ == '__main__':
    app.run(debug=True)
