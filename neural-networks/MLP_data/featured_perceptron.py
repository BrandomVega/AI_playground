import matplotlib.pyplot as plt
#plt.style.use('dark_background')
plt.style.use('bmh')

import os
os.environ["MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING"] = "true"
import time
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import torch
from torch import nn

import mlflow
mlflow.set_tracking_uri(uri="http://localhost:8080")
mlflow.set_experiment("features_perceptron")


from mlflow.models import infer_signature


def plot_data_results(X, y, line = []): 
    colors = ['orange' if label == 0 else 'black' for label in y]
    plt.scatter(X[:,0], X[:,1], c=colors)
    if line:
        plt.plot(line[0],line[1], color="gray")
    plt.xlim(-0.5,1.5)
    plt.ylim(-0.5,1.5)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("AND")
    plt.grid(True)
    plt.savefig(f"./plt_{'line' if line else 'data'}.jpg")
    plt.close()

# Tracking experiments 
with mlflow.start_run():
    start_time = time.time()

    # ==== Dataset ====
    X_df = pd.DataFrame({
        "x1": [0, 0, 1, 1],
        "x2": [0, 1, 0, 1]
    })
    Y_df = pd.DataFrame({
        "label": [0, 0, 0, 1]
    })

    # Para entrenar, conviértelo a tensores:
    X = torch.tensor(X_df.values, dtype=torch.float32)
    Y = torch.tensor(Y_df.values, dtype=torch.float32)
    # Importante: usa DataFrames para la firma
    signature = infer_signature(X_df, Y_df)

    # ==== Model/training ====
    class NeuralNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            self.fwd = nn.Linear(2,1)
        def forward(self,x):
            logit = self.fwd(x)
            logit = torch.sigmoid(logit)
            return logit
    device = "cpu"
    model = NeuralNetwork().to(device)

    epochs = 100
    lr = 0.3
    criterion = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(),lr=lr)

    mlflow.log_param("lr", lr)
    mlflow.log_param("epochs", epochs)
    mlflow.log_param("criterion", type(criterion).__name__)
    mlflow.log_param("optimizer", type(optimizer).__name__)
    

    for epoch in range(epochs):
        #time.sleep(1)
        preds = []
        targets = []
        for s in range(len(Y)):
            
            x = X[s]
            y = Y[s]

            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred,y)
            loss.backward()
            optimizer.step()

            preds.append(1 if pred.detach().item() >= 0.5 else 0)
            targets.append(y)

        #mlflow.log_metric("train_accuracy", acc, step=epoch)
        mlflow.log_metric(key="train_loss", value=loss, step=epoch, timestamp=int(time.time()*1000))    



    report_dict = classification_report(targets, preds, output_dict=True)
    mlflow.log_dict(report_dict, "classification_report.json")
    macro_f1 = report_dict['macro avg']['f1-score']
    accuracy = report_dict['accuracy']
    print("F1 Macro:", macro_f1)
    print("Accuracy:", accuracy)
    plt.clf()
    cm = confusion_matrix(targets, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='plasma')
    plt.savefig(f"./plt_confusionMatrix.jpg")
    print(classification_report(targets, preds))
    mlflow.log_artifact("plt_confusionMatrix.jpg")
    plt.clf()

    # ==== Plot weights ====
    modelWeights = model.parameters()
    weights = []
    for data in modelWeights:
        weights.append(data.tolist())
    w1 = weights[0][0][0]
    w2 = weights[0][0][1]
    b = weights[1][0]
    x1 = np.linspace(-2,2,100)
    x1 = torch.from_numpy(x1)
    y = (-1*b - w1*(x1))/w2

    plot_data_results(X,Y,[x1,y])

    mlflow.log_artifact("plt_data.jpg")
    mlflow.log_artifact("plt_line.jpg")
    mlflow.log_artifact("./featured_perceptron.py")
    mlflow.pytorch.log_model(model, artifact_path="perceptron_model", signature=signature)
    end_time = time.time()
    mlflow.log_metric("training_time_sec", end_time - start_time)