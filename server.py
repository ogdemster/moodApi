from flask import Flask, jsonify, request
from transformers import pipeline
import asyncio
from flask_cors import CORS

app = Flask(__name__)
# CORS(app, resources={r"/api/*": {"origins": "http://localhost:3000"}, "allow_headers": "*", "supports_credentials": True})
CORS(app)

# Load a pre-trained sentiment analysis model
model = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

@app.route('/')
def hello():
    return 'Hello, World!'

@app.route('/api/v1/hello')
def api_hello():
    message = {'message': 'Hello, API!'}
    return jsonify(message)

@app.route('/api/v1/data', methods=['POST'])
async def api_data():
    data = request.get_json()
    # Check if the input data is a valid JSON object
    if 'data' not in data or 'text' not in data['data']:
        return jsonify({'error': 'Missing required parameter: text'})
    # Run the model function in a separate thread
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(None, model, data['data']['text'])
    return jsonify({'received_data': result})

if __name__ == '__main__':
    app.run()
