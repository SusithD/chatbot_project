from flask import Flask, request, render_template, jsonify
from transformers import BlenderbotForConditionalGeneration, BlenderbotTokenizer
from pymongo import MongoClient
import logging
import os
from time import time
from dotenv import load_dotenv

app = Flask(__name__)

# Initialize logger
logging.basicConfig(level=logging.INFO)

# Load environment variables from .env file
load_dotenv()

# MongoDB connection using the URI from the .env file
mongo_uri = os.getenv("MONGO_URI")
if not mongo_uri:
    raise ValueError("MongoDB URI not found in .env file")

client = MongoClient(mongo_uri)
db = client.chatbotDB
chat_collection = db.chats

# Load the BlenderBot model and tokenizer
model_name = "facebook/blenderbot-3B"
logging.info(f"Loading model: {model_name}")
model = BlenderbotForConditionalGeneration.from_pretrained(model_name)
tokenizer = BlenderbotTokenizer.from_pretrained(model_name)

def generate_response(input_text):
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
    outputs = model.generate(
        inputs["input_ids"],
        max_length=200,
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    return response

# Rate-limiting to prevent spam
last_request_time = {}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    client_ip = request.remote_addr
    current_time = time()

    if client_ip in last_request_time and current_time - last_request_time[client_ip] < 1:
        return jsonify({"error": "Too many requests. Please wait a moment."}), 429
    last_request_time[client_ip] = current_time

    data = request.get_json()
    session_id = data.get("session_id", "")
    question = data.get("question", "").strip()

    if not session_id or not question:
        return jsonify({"error": "Invalid input. Provide session_id and question."}), 400

    try:
        logging.info(f"Received question: {question}")
        answer = generate_response(question)

        # Fallback for empty response
        if not answer:
            answer = "I'm not sure how to respond to that. Could you ask something else?"

        # Save chat to DB
        chat_collection.update_one(
    {"session_id": session_id},
    {
        "$push": {
            "messages": {"role": "user", "text": question, "time": time()},
            "messages": {"role": "bot", "text": answer, "time": time()}
        }
    },
    upsert=True
)
    except Exception as e:
        logging.error(f"Error during response generation: {str(e)}")
        answer = "An error occurred while processing your request. Please try again."

    return jsonify({"answer": answer})

@app.route('/new_chat', methods=['POST'])
def new_chat():
    session_id = str(time())  # Generate a unique session ID based on timestamp
    chat_collection.insert_one({"session_id": session_id, "messages": []})
    return jsonify({"session_id": session_id})

@app.route('/get_chats', methods=['GET'])
def get_chats():
    session_id = request.args.get("session_id")
    if not session_id:
        return jsonify({"error": "Session ID is required."}), 400

    chat = chat_collection.find_one({"session_id": session_id})
    if not chat:
        return jsonify({"error": "Chat not found."}), 404

    return jsonify({"session_id": session_id, "messages": chat["messages"]})

@app.route('/get_all_chats', methods=['GET'])
def get_all_chats():
    chats = chat_collection.find()
    chat_list = [{"session_id": chat["session_id"], "messages": chat["messages"]} for chat in chats]
    return jsonify(chat_list)

@app.route('/delete_chat_by_id', methods=['DELETE'])
def delete_chat_by_id():
    session_id = request.args.get("session_id")
    if not session_id:
        return jsonify({"error": "Session ID is required."}), 400

    chat_collection.delete_one({"session_id": session_id})
    return jsonify({"message": "Chat deleted successfully."})

if __name__ == '__main__':
    app.run(debug=True)
