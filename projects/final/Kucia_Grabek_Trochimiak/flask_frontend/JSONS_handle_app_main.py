import json
import uuid
from datetime import datetime
from flask import Flask, request, jsonify, render_template
import os
import base64
import tempfile
from transformers import pipeline
import torch

# Initialize the Flask app
app = Flask(__name__)

# Directory to store conversation files
CHAT_DIR = "conversations"

# Path to your local model
model_id = "C:/Users/Filip/Desktop/PW/ASP/Llama-3.2-3B-Instruct"  # Replace with your actual model path
# model_id = "C:/Users/barto/Documents/DS2024Z/NLP/UII/NLP_ASP/Llama-3.2-3B-Instruct"

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

def get_conversation_file_path(conv_id):
    return os.path.join(CHAT_DIR, f"{conv_id}.json")

def load_conversation(conv_id):
    file_path = get_conversation_file_path(conv_id)
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {
            "messages": [],
            "created_at": datetime.now().isoformat()
        }

def save_conversation(conv_id, conversation):
    # Ensure the directory exists
    if not os.path.exists(CHAT_DIR):
        os.makedirs(CHAT_DIR)
        
    file_path = get_conversation_file_path(conv_id)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(conversation, f, indent=2, ensure_ascii=False)

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
            conv_id = f"conv_{int(datetime.now().timestamp() * 1000)}_{uuid.uuid4().hex[:8]}"

        # Load existing conversation
        conversation = load_conversation(conv_id)

        # Use provided message ID or generate a new one
        if not message_id:
            message_id = f"msg_{int(datetime.now().timestamp() * 1000)}_{uuid.uuid4().hex[:8]}"

        # Save user's message
        user_message = {
            "message_id": message_id,
            "timestamp": datetime.now().isoformat(),
            "user_input": user_input,
            "assistant_response": None,
            "feedback": None
        }
        conversation["messages"].append(user_message)

        # Build the messages list for the model
        messages = []
        messages.append({
            "role": "system",
            "content": "Jesteś ekspertem w dziedzinie sztuki, który zawsze odpowiada w języku polskim."
        })
        for msg in conversation["messages"]:
            messages.append({
                "role": "user",
                "content": msg["user_input"]
            })
            if msg["assistant_response"]:
                messages.append({
                    "role": "assistant",
                    "content": msg["assistant_response"]
                })

        # Generate assistant response
        assistant_response = generate_response(messages)
        user_message["assistant_response"] = assistant_response

        save_conversation(conv_id, conversation)

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

        print(f"Received feedback - conv_id: {conv_id}, message_id: {message_id}")

        if not all([conv_id, message_id, feedback_type]):
            return jsonify({
                "error": "Missing required feedback data",
                "received_data": data
            }), 400

        conversation = load_conversation(conv_id)
        if not conversation:
            print(f"Conversation not found: {conv_id}")
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
            print(f"Message not found: {message_id}")
            return jsonify({"error": "Message not found"}), 404

        save_conversation(conv_id, conversation)
        print(f"Feedback saved successfully")
        return jsonify({"status": "Feedback recorded successfully"})

    except Exception as e:
        print(f"Feedback submission error: {str(e)}")
        return jsonify({"error": str(e)}), 500

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