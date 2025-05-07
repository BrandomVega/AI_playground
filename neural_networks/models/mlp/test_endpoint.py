import requests
import json
from PIL import Image

url = "http://127.0.0.1:8001/invocations"
data = {
    "instances": [
        {"x1": 0, "x2": 0}
    ],
    "params": {"model": 5}
}
headers = {"Content-Type": "application/json"}
response = requests.post(url, data=json.dumps(data), headers=headers)
print(response.text)
print("Status Code:", response.status_code)
if response.status_code == 200:
    try:
        response_json = response.json()
        #print("Response JSON:", response_json)
        prediccion = response_json['predictions']['prediction'][0][0]
        imagen = response_json['predictions']['artifacts_path'][2]
        print(imagen)

    except requests.exceptions.JSONDecodeError:
        print("Response is not valid JSON.")
else:
    print("Error: Received an unexpected response.")
    #print("Response Content:", response.text)

