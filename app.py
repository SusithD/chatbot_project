from flask import Flask, request, render_template, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM
from pymongo import MongoClient
import logging
import os
from time import time
from dotenv import load_dotenv

app = Flask(__name__)

# Initialize logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Load environment variables from .env file
load_dotenv()

# MongoDB connection using the URI from the .env file
mongo_uri = os.getenv("MONGO_URI")
if not mongo_uri:
    raise ValueError("MongoDB URI not found in .env file")

client = MongoClient(mongo_uri)
db = client.chatbotDB
chat_collection = db.chats

# Load the fine-tuned model and tokenizer
model_path = "models/finetuned_gemma"  # Path to your fine-tuned model
logging.info(f"Loading fine-tuned model from: {model_path}")
try:
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    logging.info("Fine-tuned model and tokenizer loaded successfully.")
except Exception as e:
    logging.error(f"Failed to load model: {str(e)}")
    raise e


def generate_response(input_text):
    """
    Generate a response using the fine-tuned model.
    """
    try:
        inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
        outputs = model.generate(
            inputs["input_ids"],
            max_length=200,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id
        )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        
        # Ensure response does not contain the question
        if response.startswith(input_text):
            response = response[len(input_text):].strip()
        
        return response
    except Exception as e:
        logging.error(f"Error generating response: {str(e)}")
        return "Sorry, I encountered an error while generating a response."


# Rate-limiting to prevent spam
last_request_time = {}

@app.route('/')
def home():
    """
    Render the homepage.
    """
    return render_template('index.html')


@app.route('/ask', methods=['POST'])
def ask():
    """
    Process user queries and return chatbot responses.
    """
    client_ip = request.remote_addr
    current_time = time()

    # Rate-limiting logic
    if client_ip in last_request_time and current_time - last_request_time[client_ip] < 1:
        return jsonify({"error": "Too many requests. Please wait a moment."}), 429
    last_request_time[client_ip] = current_time

    data = request.get_json()
    session_id = data.get("session_id", "").strip()
    question = data.get("question", "").strip()

    # Input validation
    if not session_id:
        return jsonify({"error": "Missing session_id in request."}), 400
    if not question:
        return jsonify({"error": "Question must be a non-empty string."}), 400

    try:
        logging.info(f"Received question from session {session_id}: {question}")

        # Ensure the session exists
        chat_document = chat_collection.find_one({"session_id": session_id})
        if not chat_document:
            return jsonify({"error": "Session ID not found. Please start a new chat."}), 404

        answer = generate_response(question)

        # Fallback for empty response
        if not answer:
            answer = "I'm not sure how to respond to that. Could you ask something else?"

        # Save the user question and bot answer to the corresponding session
        chat_collection.update_one(
            {"session_id": session_id},
            {
                "$push": {
                    "messages": [
                        {"role": "user", "text": question, "time": time()},
                        {"role": "bot", "text": answer, "time": time()}
                    ]
                }
            }
        )
        logging.info(f"Response saved for session {session_id}")
    except Exception as e:
        logging.error(f"Error during response generation: {str(e)}")
        answer = "An error occurred while processing your request. Please try again."

    return jsonify({"answer": answer})


@app.route('/new_chat', methods=['POST'])
def new_chat():
    """
    Start a new chat session.
    """
    # Generate a unique session ID based on timestamp
    session_id = str(int(time()))
    # Insert a new document for the new chat
    chat_collection.insert_one({"session_id": session_id, "messages": []})
    logging.info(f"New chat session created: {session_id}")
    return jsonify({"session_id": session_id})


@app.route('/get_chats', methods=['GET'])
def get_chats():
    """
    Retrieve all messages for a specific chat session.
    """
    session_id = request.args.get("session_id")
    if not session_id:
        return jsonify({"error": "Session ID is required."}), 400

    chat = chat_collection.find_one({"session_id": session_id})
    if not chat:
        return jsonify({"error": "Chat not found."}), 404

    return jsonify({"session_id": session_id, "messages": chat["messages"]})


@app.route('/get_all_chats', methods=['GET'])
def get_all_chats():
    """
    Retrieve all chat sessions.
    """
    chats = chat_collection.find()
    chat_list = [{"session_id": chat["session_id"], "messages": chat["messages"]} for chat in chats]
    return jsonify(chat_list)


@app.route('/delete_chat_by_id', methods=['DELETE'])
def delete_chat_by_id():
    """
    Delete a specific chat session by ID.
    """
    session_id = request.args.get("session_id")
    if not session_id:
        return jsonify({"error": "Session ID is required."}), 400

    result = chat_collection.delete_one({"session_id": session_id})
    if result.deleted_count == 0:
        return jsonify({"error": "Chat not found."}), 404

    logging.info(f"Chat session {session_id} deleted successfully.")
    return jsonify({"message": "Chat deleted successfully."})


if __name__ == '__main__':
    app.run(debug=True)
