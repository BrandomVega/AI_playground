import os
import warnings; warnings.filterwarnings("ignore")
import mlflow.pytorch
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt;plt.style.use('bmh');plt.rcParams['figure.dpi'] = 100  
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.datasets import make_circles
import torch            # type: ignore
from torch import nn    # type: ignore
import mlflow 
from mlflow.models import infer_signature

os.environ["MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING"] = "true"
mlflow.set_tracking_uri(uri="http://localhost:8000")
mlflow.set_experiment("model_training")

NN_MODEL_NAME_PREFIX = "nn_model_"
MME_MODEL_NAME = "MME_nn_model"
ID = 0

def plot_data_results(X, y, ID=None,  weights=None, model=None): 
    plt.clf()
    colors = ['black' if label == 0 else 'orange' for label in y]
    plt.scatter(X[:,0], X[:,1], c=colors)
    if weights:
        w1 = weights[0][0]
        w2 = weights[0][0]
        b = weights[1]
        x = np.linspace(-2,2,100)
        x = torch.from_numpy(x)
        y = (-1*b - w1*(x))/w2
        plt.plot(x,y, color="gray")
    if model:
        definition = 300
        x1 = np.linspace(-2.0, 2.0, definition)
        x2 = np.linspace(-2.0, 2.0, definition)
        x1, x2 = np.meshgrid(x1, x2)
        array_Z = []
        print(f"Procesando mesh")
        for dim in range(definition):
            X_mesh = [[a,b] for a,b in zip(x1[dim],x2[dim])]
            X_mesh = torch.tensor(X_mesh, dtype = torch.float32, device = "cpu")
            Z = []
            for row in X_mesh:
                Z.append(model(row).detach().numpy())
            array_Z.append(Z)
        plt.imshow(array_Z, cmap='YlGnBu',extent=(-1,2,-1,2)) 
        plt.colorbar()
    if weights:
        name = "line"
    elif model:
        name = "model"
    else:
        name = "data"
    plt.xlim(-2,2)
    plt.ylim(-2,2)
    plt.xlabel("X")
    plt.ylabel("Y")
    if ID:
        plt.title(f"Model-{ID}")
    plt.grid(True)
    plt.savefig(f"./plt_{name}.jpg")
    plt.close()


for i in range(5):
    ID = i
    print(ID)
    if ID < 4:
        X_df = pd.DataFrame({
            "x1": [0, 0, 1, 1],
            "x2": [0, 1, 0, 1]
        })
        Y_df = pd.DataFrame({
            "label": [[0 if ID<2 else 1][0], 0, 0, 1]
        })
    elif ID==4:
        X, y = make_circles(n_samples=50, random_state=42, factor=0.2, noise=0.05)
        X_df = pd.DataFrame(X, columns=["x1", "x2"])
        Y_df = pd.DataFrame({"label": y})
    else: 
        points = 150
        datapoints = np.linspace(0, 4 * np.pi, points)

        r1 = np.linspace(0, 2, points)
        r2 = np.linspace(0.1, -2, points)
        x1 = r1 * np.cos(datapoints)
        y1 = r1 * np.sin(datapoints)
        x2 = r2 * np.cos(datapoints)
        y2 = r2 * np.sin(datapoints)

        x_total = np.concatenate([x1, x2])
        y_total = np.concatenate([y1, y2])
        X_df = pd.DataFrame({
            "x1": x_total,
            "x2": y_total
        })
        Y_df = pd.DataFrame({
            "label": [0]*points + [1]*points
        })

    X = torch.tensor(X_df.values, dtype=torch.float32)
    Y = torch.tensor(Y_df.values, dtype=torch.float32)
    plot_data_results(X,Y,ID=i, weights=None, model=None)

    signature = infer_signature(
        model_input=X_df,
        model_output=Y_df
    )

    class NeuralNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            if ID < 2:
                self.fwd = nn.Linear(2,1)
            elif ID < 4:
                self.fwd = nn.Sequential(
                    nn.Linear(2,3),
                    nn.Sigmoid(),
                    nn.Linear(3,1)
                )
            elif ID==4: 
                self.fwd = nn.Sequential(
                    nn.Linear(2,8),
                    nn.Tanh(),
                    nn.Linear(8,1),
                )
            else:
                raise NotImplemented
            
        def forward(self,x):
            if ID ==1:
                x = torch.cos(x)
            if ID ==3:
                x = torch.sin(x)
            if ID==5:
                raise NotImplemented
            x = self.fwd(x)
            x = torch.sigmoid(x)
            return x
        
    device = "cpu"
    model = NeuralNetwork().to(device)
    print(model)
    print(Y_df)

    with mlflow.start_run():
        epochs = 500
        lr = 0.01
        criterion = nn.BCELoss()
        optimizer = torch.optim.SGD(model.parameters(),lr=lr)
        mlflow.log_param("lr", lr)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("criterion", type(criterion).__name__)
        mlflow.log_param("optimizer", type(optimizer).__name__)

        for epoch in range(epochs):
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

            print(f"epoch: {epoch} - loss: {loss}")
            mlflow.log_metric(key="train_loss", value=loss, step=epoch, timestamp=int(time.time()*1000))    

        report_dict = classification_report(targets, preds, output_dict=True)
        print(classification_report(targets, preds))

        macro_f1 = report_dict['macro avg']['f1-score']
        accuracy = report_dict['accuracy']
        print("F1 Macro:", macro_f1)
        print("Accuracy:", accuracy)
        
        plt.clf()
        cm = confusion_matrix(targets, preds)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap='plasma')
        plt.savefig(f"./plt_bran_confusionMatrix.jpg")

        plot_data_results(X,Y, ID=i,model=model)

        artifact_path = f"{NN_MODEL_NAME_PREFIX}{ID}"

        mlflow.log_dict(report_dict, f"{NN_MODEL_NAME_PREFIX}{ID}/classification_report.json")
        mlflow.log_artifact("plt_confusionMatrix.jpg", artifact_path=artifact_path)
        mlflow.log_artifact("plt_data.jpg",artifact_path=artifact_path)
        mlflow.log_artifact("plt_model.jpg", artifact_path=artifact_path)
        mlflow.log_artifact("./models_beve_.py", artifact_path=artifact_path)
        mlflow.pytorch.log_model(
            model,
            artifact_path=f"{NN_MODEL_NAME_PREFIX}{ID}",
            registered_model_name=f"{NN_MODEL_NAME_PREFIX}{ID}",
            signature=signature
        )
