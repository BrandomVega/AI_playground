from flask import Flask, request, jsonify
import requests

app = Flask(__name__)

@app.route('/send-data', methods=['POST'])
def send_data():
    input_data = request.get_json()
    url = 'http://localhost:5002/invocations'
    headers = {'Content-Type': 'application/json'}

    try:
        response = requests.post(url, json=input_data, headers=headers)
        response.raise_for_status()
        print(response.json())
        return jsonify(response.json())
    except requests.exceptions.RequestException as e:
        return jsonify({"error": str(e)}), 500
