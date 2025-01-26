import uuid
from datetime import datetime
from flask import Flask, request, jsonify, render_template
import os
import base64
import tempfile
from transformers import pipeline
import torch
import json
from redis import Redis

# Initialize the Flask app
app = Flask(__name__)

# Redis configuration
redis_host = 'localhost'
redis_port = 6379
redis_db = 0

# Initialize Redis connection
redis_client = Redis(host=redis_host, port=redis_port, db=redis_db, decode_responses=True)

# Path to your local model
model_id = "C:/Users/Filip/Desktop/PW/ASP/Llama-3.2-3B-Instruct"  # Replace with your actual model path

# Load the model using pipeline
pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
# Access the tokenizer if needed
tokenizer = pipe.tokenizer
tokenizer.pad_token = tokenizer.eos_token

# JSON file to store conversations and feedback
CHAT_FILE = "chat_history.json"

# Function to generate responses
def generate_response(messages, max_new_tokens=256):
    # Build the prompt from messages
    prompt = ''
    for message in messages:
        if message['role'] == 'system':
            prompt += f"{message['content']}\n"
        elif message['role'] == 'user':
            prompt += f"Użytkownik: {message['content']}\n"
        elif message['role'] == 'assistant':
            prompt += f"Asystent: {message['content']}\n"
    prompt += "Asystent:"

    # Generate response using the pipeline
    outputs = pipe(
        prompt,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        do_sample=True,
    )
    
    # Extract the assistant's response
    generated_text = outputs[0]['generated_text']
    # Remove the prompt from the generated text
    response = generated_text[len(prompt):].strip()
    return response

# Helper functions for Redis operations
def save_message(conv_id, message):
    # Key for the conversation
    conv_key = f"conversation:{conv_id}:messages"
    # Append the message to the conversation list
    redis_client.rpush(conv_key, json.dumps(message))

    # Set a timestamp for the conversation if it's new
    conv_timestamp_key = f"conversation:{conv_id}:created_at"
    if not redis_client.exists(conv_timestamp_key):
        redis_client.set(conv_timestamp_key, datetime.now().isoformat())

    # Update the JSON file
    update_chat_history_json(conv_id)

def get_conversation_messages(conv_id):
    conv_key = f"conversation:{conv_id}:messages"
    messages_data = redis_client.lrange(conv_key, 0, -1)
    messages = [json.loads(msg_data) for msg_data in messages_data]
    return messages

def conversation_exists(conv_id):
    conv_key = f"conversation:{conv_id}:messages"
    return redis_client.exists(conv_key)

def update_message_feedback(conv_id, message_id, feedback):
    conv_key = f"conversation:{conv_id}:messages"
    messages_data = redis_client.lrange(conv_key, 0, -1)
    for idx, msg_data in enumerate(messages_data):
        msg = json.loads(msg_data)
        if msg.get("message_id") == message_id:
            msg["feedback"] = feedback
            redis_client.lset(conv_key, idx, json.dumps(msg))
            # Update the JSON file
            update_chat_history_json(conv_id)
            return True
    return False

# Functions to handle JSON file operations
def update_chat_history_json(conv_id):
    chat_history = load_chat_history()
    if "conversations" not in chat_history:
        chat_history["conversations"] = {}

    # Get messages from Redis
    messages = get_conversation_messages(conv_id)
    created_at_key = f"conversation:{conv_id}:created_at"
    created_at = redis_client.get(created_at_key)

    # Update the chat history
    chat_history["conversations"][conv_id] = {
        "messages": messages,
        "created_at": created_at
    }

    save_chat_history(chat_history)

def load_chat_history():
    try:
        with open(CHAT_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {"conversations": {}}

def save_chat_history(data):
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

        # Use provided message ID or generate a new one
        if not message_id:
            message_id = str(uuid.uuid4())

        # Save user's message to Redis
        user_message = {
            "message_id": message_id,
            "timestamp": datetime.now().isoformat(),
            "role": "user",
            "content": user_input,
            "feedback": None
        }
        save_message(conv_id, user_message)

        # Build the messages list for the model
        messages = []
        # Add the system prompt
        messages.append({
            "role": "system",
            "content": "Jesteś ekspertem w dziedzinie sztuki, który zawsze odpowiada w języku polskim."
        })
        # Add conversation history
        conversation_messages = get_conversation_messages(conv_id)
        for msg in conversation_messages:
            messages.append({
                "role": msg.get("role", "user"),
                "content": msg["content"]
            })

        # Generate assistant response
        assistant_response = generate_response(messages)

        # Save assistant's response in Redis
        assistant_message = {
            "message_id": str(uuid.uuid4()),
            "timestamp": datetime.now().isoformat(),
            "role": "assistant",
            "content": assistant_response,
            "feedback": None
        }
        save_message(conv_id, assistant_message)

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

        if not conversation_exists(conv_id):
            print(f"Conversation not found: {conv_id}")  # Debug log
            return jsonify({"error": "Conversation not found"}), 404

        feedback = {
            "type": feedback_type,
            "comment": comment,
            "timestamp": datetime.now().isoformat()
        }
        updated = update_message_feedback(conv_id, message_id, feedback)

        if not updated:
            print(f"Message not found: {message_id}")  # Debug log
            return jsonify({"error": "Message not found"}), 404

        print(f"Feedback saved successfully")  # Debug log
        return jsonify({"status": "Feedback recorded successfully"})

    except Exception as e:
        print(f"Feedback submission error: {str(e)}")  # Debug log
        return jsonify({"error": str(e)}), 500

# Note: The text-to-speech endpoint remains unchanged
@app.route('/text-to-speech', methods=['POST'])
def generate_speech():
    try:
        data = request.json
        text = data.get('text')
        
        if not text:
            return jsonify({"error": "Text is required"}), 400

        # Assuming you have a local TTS system or another API to handle TTS
        # Here, we just return a placeholder since OpenAI's TTS is not used
        audio_base64 = "placeholder_for_audio_base64_string"

        return jsonify({
            "audio": audio_base64,
            "status": "success"
        })

    except Exception as e:
        print(f"Error in text-to-speech endpoint: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
