'''
1. 필요한 라이브러리 import
'''
import argparse
import os
import subprocess, sys
import json

import logging
import logging.handlers

import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim

'''
2. Local <-> SageMaker Training instances 환경
'''
def train_sagemaker(args):
    if os.environ.get('SM_CURRENT_HOST') is not None:
        args.train_data_dir = os.environ.get('SM_CHANNEL_TRAIN')
        args.model_dir = os.environ.get('SM_MODEL_DIR')
        args.output_data_dir = os.environ.get('SM_OUTPUT_DATA_DIR')
    return args

'''
3. 디버깅을 위한 로거
'''
def get_logger():
    loglevel = logging.DEBUG
    l = logging.getLogger(__name__)
    if not l.hasHandlers():
        l.setLevel(loglevel)
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
        l.hander_set = True
    return l

logger = get_logger()

if __name__=='__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--train_data_dir', type=str, default='opt/ml/processing/output/train')
    parser.add_argument('--model-dir', type=str, default='opt/ml/model')
    parser.add_argument('--output-data-dir', type=str, default='opt/ml/output')

    args = parser.parse_args()

    args = train_sagemaker(args)
    
    train_data_dir = args.train_data_dir
    model_location = args.model_dir + '/CvD-model'
    metrics_location = args.output_data_dir + '/CvD-metrics.json'
    epochs = args.epochs

    os.makedirs(model_location, exist_ok=True)
    # os.makedirs(metrics_location, exist_ok=True)
    
    
    logger.info("############# New Training Script Argument Info #############")
    logger.info(f"train_data_path: {train_data_dir}")
    logger.info(f"model_location: {model_location}")
    logger.info(f"metrics_location: {metrics_location}")
    logger.info(f"Epoch: {epochs}")

    # 입력 데이터 dataloader
    transform = transforms.Compose(
        [
            transforms.Resize([256, 256]),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ]
    )

    train_dataset = torchvision.datasets.ImageFolder(
        train_data_dir,
        transform=transform
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=32,
        num_workers=8,
        shuffle=True
    )
    

    # 전이 학습, transfer learning
    model = models.resnet18(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False
    
    model.fc = nn.Linear(512, 2)
    for param in model.fc.parameters():
        param.requires_grad = True

    optimizer = torch.optim.Adam(model.fc.parameters())
    cost = nn.CrossEntropyLoss()

    # logger.info(f"Model Created! {str(model)}")

    # 학습

    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    acc_history = []
    loss_history = []
    best_acc = 0.0
    best_model_wts = model.state_dict()

    for epoch in range(epochs):
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            model.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = cost(outputs, labels)
            _, preds = torch.max(outputs, 1)
            loss.backward()
            optimizer.step()

            running_loss = loss.item() * inputs.size(0)
            running_corrects = torch.sum(preds==labels.data)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double()/len(train_loader.dataset)
            
        logger.info(f"Epoch {epoch+1}/{epochs}. Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
        acc_history.append(epoch_acc.item())
        loss_history.append(epoch_loss)

        if epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = model.state_dict()
            best_epoch = epoch+1

    torch.save(model.state_dict(), f"{model_location}/CvD_best_model.pth")
    logger.info(f"Model saved at best accuracy of {best_acc:.4f} at epoch {best_epoch}")

    history_dict = {
        'acc_history': acc_history,
        'loss_history': loss_history
    }

    with open(metrics_location, 'w') as f:
        json.dump(history_dict, f)

