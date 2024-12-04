from flask import Flask, request, render_template, jsonify
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
    data = request.get_json()  # Get the JSON data from the request
    question = data.get('question')  # Get the question from the JSON
    context = data.get('context')  # Get the context from the JSON
    
    if not question or not context:
        return jsonify({"error": "Question and context are required."}), 400

    result = qa_pipeline(question=question, context=context)
    return jsonify({"answer": result['answer']})

if __name__ == '__main__':
    app.run(debug=True)
