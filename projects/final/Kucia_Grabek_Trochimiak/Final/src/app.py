import json
import uuid
import os
from datetime import datetime

from flask import Flask, request, jsonify, render_template

from rag.database import RAGSystem
from chat.chat import ArtExpertWithRAG

# Initialize the Flask app
app = Flask(__name__)

# JSON file to store conversations and feedback
CHAT_FILE = "chat_history.json"


art_expert_chat = ArtExpertWithRAG(RAGSystem("rag_collection", "en"))


# Function to generate responses
def generate_response(messages: list[dict]) -> str:
    prompt = messages[-1]["content"]

    response = art_expert_chat.get_response(prompt, messages[:-1])

    return response


def load_chat_history() -> dict:
    try:
        with open(CHAT_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {"conversations": {}}


def save_chat_history(data: dict) -> None:
    # Ensure the directory exists
    directory = os.path.dirname(CHAT_FILE)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
        
    with open(CHAT_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        user_input = data.get("message")
        conv_id = data.get("conversationId")
        message_id = data.get("messageId")

        if not user_input:
            return jsonify({"error": "Message is required"}), 400

        # Ensure we have a conversation ID
        if not conv_id:
            conv_id = str(uuid.uuid4())

        chat_history = load_chat_history()

        # Initialize conversations dict if it doesn't exist
        if "conversations" not in chat_history:
            chat_history["conversations"] = {}

        # Initialize conversation if new
        if conv_id not in chat_history["conversations"]:
            chat_history["conversations"][conv_id] = {
                "messages": [],
                "created_at": datetime.now().isoformat()
            }

        if not message_id:
            message_id = str(uuid.uuid4())

        # Save user's message to history
        user_message = {
            "message_id": message_id,
            "timestamp": datetime.now().isoformat(),
            "role": "user",
            "content": user_input,
            "feedback": None
        }
        chat_history["conversations"][conv_id]["messages"].append(user_message)

        messages = []
        # Add conversation history
        for msg in chat_history["conversations"][conv_id]["messages"]:
            messages.append({
                "role": msg.get("role", "user"),
                "content": msg["content"]
            })

        assistant_response = generate_response(messages)

        # Save assistant's response in conversation history
        assistant_message = {
            "message_id": str(uuid.uuid4()),
            "timestamp": datetime.now().isoformat(),
            "role": "assistant",
            "content": assistant_response,
            "feedback": None
        }
        chat_history["conversations"][conv_id]["messages"].append(assistant_message)
        save_chat_history(chat_history)

        return jsonify({
            "response": assistant_response,
            "conversationId": conv_id,
            "messageId": message_id
        })

    except Exception as e:
        print(f"Error in chat endpoint: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route('/feedback', methods=['POST'])
def submit_feedback():
    try:
        data = request.json
        conv_id = data.get("conversationId")
        message_id = data.get("messageId")
        feedback_type = data.get("type")
        comment = data.get("comment", "")

        print(f"Received feedback - conv_id: {conv_id}, message_id: {message_id}")  # Debug log

        if not all([conv_id, message_id, feedback_type]):
            return jsonify({
                "error": "Missing required feedback data",
                "received_data": data
            }), 400

        chat_history = load_chat_history()

        # Initialize conversations dict if it doesn't exist
        if "conversations" not in chat_history:
            chat_history["conversations"] = {}

        # Find and update the specific message
        conversation = chat_history["conversations"].get(conv_id)
        if not conversation:
            print(f"Conversation not found: {conv_id}")  # Debug log
            print(f"Available conversations: {list(chat_history['conversations'].keys())}")  # Debug log
            return jsonify({"error": "Conversation not found"}), 404

        message_found = False
        for message in conversation["messages"]:
            if message.get("message_id") == message_id:
                message["feedback"] = {
                    "type": feedback_type,
                    "comment": comment,
                    "timestamp": datetime.now().isoformat()
                }
                message_found = True
                break

        if not message_found:
            print(f"Message not found: {message_id}")  # Debug log
            return jsonify({"error": "Message not found"}), 404

        save_chat_history(chat_history)
        print(f"Feedback saved successfully")  # Debug log
        return jsonify({"status": "Feedback recorded successfully"})

    except Exception as e:
        print(f"Feedback submission error: {str(e)}")  # Debug log
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
