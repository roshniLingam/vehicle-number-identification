from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from mlmodel import Model

app = Flask(__name__)
CORS(app)

# Create object of Model class
model = Model()

# Define a POST request mapping
@app.route("/upload", methods=["POST"])
def get_registration_number():
    return model.process_image()

if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=2040)