from flask import Flask, jsonify, request
import pathlib
from mini.inference import MINI
import os

current_path = pathlib.Path(__file__).parent.absolute()
vector_store_path = os.path.join(current_path, 'vector_store')

app = Flask(__name__)

mini = MINI(model_name='meta-llama/Llama-3.2-3B-Instruct')


@app.route("/ask", methods=["POST"])
def get_reponse():
    if request.method == "POST":
        data = request.get_json()
        question = data.get("question")
        if question:
            answer, sources = mini.query(question)
            return jsonify({"answer": answer, "source": sources})

    return jsonify({"error": "Invalid request"}), 400


if __name__ == "__main__":
    app.run(debug=True, port=5003)
