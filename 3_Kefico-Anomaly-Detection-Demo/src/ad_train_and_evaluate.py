import os
import subprocess, sys
import json
import argparse

import logging
import logging.handlers

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

import pandas as pd
import numpy as np

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

    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--window_size', type=int, default=50)
    parser.add_argument('--latent_dim', type=int, default=20)
    parser.add_argument('--thres', type=float, default=1.0)
    
    parser.add_argument('--train_data_dir', type=str, default='opt/ml/processing/output/train')
    parser.add_argument('--model-dir', type=str, default='opt/ml/model')
    parser.add_argument('--output-data-dir', type=str, default='opt/ml/output')

    args = parser.parse_args()

    args = train_sagemaker(args)
    
    train_data_dir = args.train_data_dir
    model_location = args.model_dir + '/kefico-AE-model'
    metrics_location = args.output_data_dir + '/kefico-AE-results.txt'

    learning_rate = args.learning_rate
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    window_size = args.window_size
    latent_dim = args.latent_dim
    thres = args.thres

    os.makedirs(model_location, exist_ok=True)
    os.makedirs(args.output_data_dir, exist_ok=True)
    
    
    logger.info("############# New Training Script Argument Info #############")
    logger.info(f"train_data_path: {train_data_dir}")
    logger.info(f"model_location: {model_location}")
    logger.info(f"metrics_location: {metrics_location}")
    logger.info(f"num_epochs: {num_epochs}")
    logger.info(f"learning_rate: {learning_rate}")

    

    class MyDataLoader(Dataset):
        def __init__(self, df, window_size):
            self.original_index = df.index
            self.df = df.reset_index(drop=True)
            self.window_size = window_size
            # print(f'original df: ', self.df.shape)
            self.data = torch.tensor(self.df.values, dtype=torch.float32)
            # print(f'tensor size: ', self.data.size())
    
        def __len__(self):
            return self.df.shape[0] - self.window_size + 1
    
        def __getitem__(self, idx):
            """
            :param idx: 데이터셋에서 가져올 샘플의 인덱스
            :return: idx 번째부터 window_size만큼 잘라낸 데이터와 원본 인덱스 반환
            """
            if idx + self.window_size > len(self.df):
                raise IndexError("Index out of range for window size")
    
                # window_size만큼 데이터 슬라이싱
            sequence = self.data[idx: idx + self.window_size]
            return sequence, self.original_index[idx]


    class AutoEncoder(nn.Module):
        def __init__(self, input_dim, z_dim, win_size):
            super(AutoEncoder, self).__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim * win_size, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, z_dim),
                nn.ReLU())
            self.decoder = nn.Sequential(
                nn.Linear(z_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 128),
                nn.ReLU(),
                nn.Linear(128, input_dim * win_size),
                nn.Sigmoid())

        def forward(self, x):
            # x = x.view(x.size(0), -1)
            x = self.encoder(x)
            x = self.decoder(x)
            return x

    parquet_list = [file for file in os.listdir(train_data_dir) if file.endswith('.parquet')]
    if parquet_list:
        parquet_name = parquet_list[0]
    parquet_path = os.path.join(train_data_dir, parquet_name)
    df = pd.read_parquet(parquet_path)

    logger.info(f"df shape: {df.shape}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    var_num =len(df.columns)
    train_size = int(len(df) * 0.1)
    train_df = df.iloc[:train_size]
    
    # DataLoader 생성
    train_data = MyDataLoader(train_df, window_size)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False)

    # AutoEncoder 생성
    model = AutoEncoder(var_num, latent_dim, window_size).to(device)
    # logger.info(model)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    #학습
    model.train()
    best_loss = float('inf')  # 가장 낮은 loss 값을 저장할 변수
    best_model_wts = None


    for epoch in range(num_epochs):
        running_loss = 0.0
        for data, _ in train_loader:
            inputs = data.reshape(-1, var_num * window_size).to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        logger.info('Training AutoEncoder... Epoch: {}, Loss: {:.3f}'.format(epoch, epoch_loss))

        # 가장 낮은 epoch_loss일 때 모델 저장
        if epoch_loss < best_loss:
            best_loss = epoch_loss
        best_model_wts = model.state_dict()  # 모델의 state_dict 저장

    torch.save(best_model_wts, f'{model_location}/best_model.pth')
    logger.info("Best model saved with loss: {:.3f}".format(best_loss))

    eval_data = MyDataLoader(df, window_size)
    eval_loader = DataLoader(eval_data, batch_size=batch_size, shuffle=False )

    model.load_state_dict(best_model_wts)
    logger.info("Model Loaded!")

    
    with torch.no_grad():
        logger.info('Calculating Threshold..')
        all_mse = []
        for batch_data, batch_index in train_loader:
            batch_data = batch_data.reshape(-1, var_num * window_size).to(device)

            # 각 batch의 스코어를 계산해야 해!!
            output = model(batch_data)
            mse = nn.functional.mse_loss(output, batch_data, reduction='none')
            mse = mse.mean(dim=1).cpu().numpy()
            all_mse.extend(mse)

        threshold = np.quantile(all_mse, (100-thres)/100.0)
        logger.info(f'Threshold: {threshold}')

        all_mse = []
        all_index = []
        logger.info("Evaluating..")
        for batch_data, batch_index in eval_loader:
            batch_data = batch_data.reshape(-1, var_num * window_size).to(device)

            # 각 batch의 스코어를 계산해야 해!!
            output = model(batch_data)
            mse = nn.functional.mse_loss(output, batch_data, reduction='none')
            mse = mse.mean(dim=1).cpu().numpy()
            all_mse.extend(mse)
            all_index.extend(batch_index.cpu().numpy())

        pred = (all_mse > threshold).astype(int)

        # pred가 1인 지점의 all_indices 값만 가져오기
        filtered_indices = [index for index, p in zip(all_index, pred) if p == 1]

        logger.info(f"pred가 1인 지점의 all_indices 값 예시 : {filtered_indices[:3]}")
        logger.info(f"Anomaly found: {len(filtered_indices)}/{len(eval_loader.dataset)}")

    with open(metrics_location, 'w') as f:
        for item in filtered_indices:
            f.write(f"{item}\n")


















