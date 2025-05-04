from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import requests
import uvicorn

app = FastAPI()

@app.get("/", response_class=HTMLResponse)
def read_index():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

class InputData(BaseModel):
    x1: float
    x2: float

@app.post("/predict")
def predict(data: InputData):
    url = "http://localhost:5000/send-data"
    payload = {
        "dataframe_split": {
            "columns": ["x1", "x2"],
            "data": [[data.x1, data.x2]]
        }
    }
    headers = {'Content-Type': 'application/json'}
    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        prediction = response.json()['predictions'][0]['0']
        print(prediction)
        return {"prediction": prediction}
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run("fastp:app", port=8080, reload=True)
