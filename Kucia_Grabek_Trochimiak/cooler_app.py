import json
import uuid
from datetime import datetime
from flask import Flask, request, jsonify, render_template
from chat.polish_art_expert import PolishArtExpertRAG
from rag.database import PolishRAGSystem
import os
import base64
import tempfile
from transformers import pipeline
import torch

# Initialize the Flask app
app = Flask(__name__)

# JSON file to store conversations and feedback
CHAT_FILE = "flask_frontend/chat_history.json"

# Path to your local model
# model_id = "C:/Users/Filip/Desktop/PW/ASP/Llama-3.2-3B-Instruct"  # Replace with your actual model path

# Load the model using pipeline
#pipe = pipeline(
#    "text-generation",
#    model=model_id,
#    torch_dtype=torch.bfloat16,
#    device_map="auto",
#)

# Access the tokenizer if needed
#tokenizer = pipe.tokenizer
#tokenizer.pad_token = tokenizer.eos_token

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("Error: OPENAI_API_KEY environment variable not set")
    exit(1)

art_expert_chat = PolishArtExpertRAG(PolishRAGSystem("polish_documents"), api_key)

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
    # print(messages[1:-1])
    prompt = messages[-1]["content"]#+= "Asystent:"

    response = art_expert_chat.get_response(prompt, messages[1:-1])

    # Generate response using the pipeline
    #outputs = pipe(
    #    prompt,
    #    max_new_tokens=max_new_tokens,
    #    pad_token_id=tokenizer.eos_token_id,
    #    eos_token_id=tokenizer.eos_token_id,
    #    do_sample=True,
    #)
    
    # Extract the assistant's response
    # generated_text = outputs[0]['generated_text']
    #print("test")
    # Remove the prompt from the generated text
    #response = generated_text[len(prompt):].strip()
    #return response
    return response


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

        # Load existing chat history
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

        # Use provided message ID or generate a new one
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

        # Build the messages list for the model
        messages = []
        # Add the system prompt
        messages.append({
            "role": "system",
            "content": "Jesteś ekspertem w dziedzinie sztuki, który zawsze odpowiada w języku polskim."
        })
        # Add conversation history
        for msg in chat_history["conversations"][conv_id]["messages"]:
            messages.append({
                "role": msg.get("role", "user"),
                "content": msg["content"]
            })

        # Generate assistant response
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

# Note: The text-to-speech endpoint remains unchanged, as it's not directly related to the LLM.
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
