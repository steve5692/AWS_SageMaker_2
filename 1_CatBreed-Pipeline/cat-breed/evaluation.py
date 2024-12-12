import os
import json
import pathlib
import pickle
import tarfile
import joblib
import numpy as np
import pandas as pd
import torch
from torchvision import datasets, transforms

if __name__ == "__main__":

    model_path = f"/opt/ml/processing/model/model.tar.gz"
    with tarfile.open(model_path) as saved_model_tar:
        saved_model_tar.extractall(".")
    model_file_path = next(pathlib.Path(".").glob("model.pth")).stem + ".pth"
    model = torch.load(model_file_path)

    for param in model.parameters():
        param.requires_grad=False

    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    
    if torch.cuda.is_available():
        model.to(device)
    model.eval()

    test_path = "/opt/ml/processing/test/"

    test_dataset = datasets.ImageFolder(os.path.join(test_path))

    test_dataset.transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, shuffle=True, num_workers=4
    )

    correct = 0
    for i, l in test_dataloader:
        output = model(i)
        output = np.argmax(output)
        correct += (output==1).int().sum()

    accuracy = 100 * correct / len(test_dataloader)

    report_dict = {
        "classification_metrics": {
            "accuracy": {
                "value": float(accuracy.numpy())
            }
        }
    }

    output_dir = "/opt/ml/processing/evaluation"
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    evaluation_path = f"{output_dir}/evaluation.json"
    with open(evaluation_path, "w") as f:
        f.write(json.dumps(report_dict))











