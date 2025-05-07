# AI PLAYGROUND

Welcome to AI playground, where you can mess around with several models including artificial, convolutional and recurrent neuronal networks in an interactive and visual manner. 

| ![Challenge](model_5.png) | ![Moving](sigmoidPerceptron.gif) |
|---------------------------|----------------------------------|

This abroads from
-Train challenging datasets 
-Experiment with model architectures 
-Experiment with mathematical features inside the models
-Explore interpretability and visualization of models
-Manage ML lifecicle with tools as mlflow
-Develop and manage end to end services from web REST API's to server management and docker based deployments
 

## Usage
1. Clone repository:
    ```bash
    git clone https://github.com/BrandomVega/AI_playground.git
    cd AI_playground
    ```

3. Create python virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate
    ```
4. Install dependencies
    ```bash
    pip install -r requirements.txt
    ```
5. Track and register your models with mlflow
    ```bash
    cd models/mlp
    python models.py
    python build_model.py
    ```
6. Serve the model locally or build a container from your model
    ```bash
    mlflow models serve -m "models:/{MODEL}/1" --port {PORT}
    mlflow models build-docker --model-uri "models:/{MODEL}/1" --name "{NAME}"    
    ```
7. Choose the page to serve with flask
    ```bash
    cd neural-networks && python app.py
    ```
8. Finally
    ```bash
    have fun - bv
    ```
