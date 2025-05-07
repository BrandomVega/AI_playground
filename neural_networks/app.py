from flask import Flask, jsonify, request, send_file
import requests
import json

app = Flask(__name__)
path = ''   

@app.route('/')
def home():
    """ Load main page"""
    return open('templates/index.html').read()

@app.route('/temp-image')
def temp_image():
    """Returns the image from model artifacts"""
    return send_file(path, mimetype='image/jpeg')


@app.route('/predict', methods=['POST'])
def predict():
    """Requests a prediction from the model Docker container given x1, x2, and the model idx"""
    global path
    data = request.get_json()

    try:
        x1 = float(data.get('x1'))
        x2 = float(data.get('x2'))
        model = int(data.get('model'))
        print(model)
    except (TypeError, ValueError):
        return jsonify({'error': 'Entradas inválidas'}), 400

    print(f"Valores recibidos BV: x1={x1}, x2={x2}, model={model}")

    url = "http://127.0.0.1:8001/invocations"
    payload = {
        "instances": [{"x1": x1, "x2": x2}],
        "params": {"model": model}
    }

    headers = {"Content-Type": "application/json"}
    response = requests.post(url, data=json.dumps(payload), headers=headers)

    if response.status_code == 200:
        try:
            response_json = response.json()
            print(response_json)
            prediccion = response_json['predictions']['prediction'][0][0]
            path = response_json['predictions']['artifacts_path'][2]
            print(f"Imagen generada: {path}")
        except (KeyError, json.JSONDecodeError):
            print("Respuesta inválida del servidor.")
            return jsonify({'error': 'Respuesta inválida del modelo'}), 500
    else:
        print("Error: respuesta inesperada del servidor.")
        return jsonify({'error': 'Error en el modelo'}), 500

    return jsonify({'pred': round(prediccion, 10)})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
