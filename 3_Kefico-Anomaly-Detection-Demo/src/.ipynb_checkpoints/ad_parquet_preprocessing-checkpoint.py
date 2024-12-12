import argparse
import os
import time
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from urllib.parse import urlparse
from io import StringIO
import random
import shutil
import sys
import boto3

import logging
import logging.handlers

def train_sagemaker(args):
    if os.environ.get('SM_CURRENT_HOST') is not None:
        args.train_data_dir = os.environ.get('SM_CHANNEL_TRAIN')
        args.model_dir = os.environ.get('SM_MODEL_DIR')
        args.output_data_dir = os.environ.get('SM_OUTPUT_DATA_DIR')
    return args

def get_logger():
    '''
    로깅을 위한 파이썬 로거
    '''
    loglevel = logging.DEBUG
    l = logging.getLogger(__name__)
    if not l.hasHandlers():
        l.setLevel(loglevel)
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
        l.handler_set = True
    return l

logger = get_logger()

if __name__ == '__main__':
    '''
    커맨드 인자 파싱
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_preproc_input_dir', type=str, default='/opt/ml/processing/input/raw')
    parser.add_argument('--base_preproc_output_dir', type=str, default='/opt/ml/processing/output')
    parser.add_argument('--THRES_STD', type=float, default=1e-5)

    
    # parse arguments
    args = parser.parse_args()
    logger.info('#### Argument Info ####')
    logger.info(f"args.base_preproc_input_dir: {args.base_preproc_input_dir}")
    logger.info(f"args.base_preproc_output_dir: {args.base_preproc_output_dir}")
    logger.info(f"args.THRES_STD: {args.THRES_STD}")

    base_preproc_input_dir = args.base_preproc_input_dir
    base_preproc_output_dir = args.base_preproc_output_dir
    THRES_STD = args.THRES_STD
    
    chunk_parquet_list = os.listdir(base_preproc_input_dir)
    chunk_parquet_list.sort()
    
    logger.info(f"There are {len(chunk_parquet_list)} parquet files.")

    ex_file_name = chunk_parquet_list[0]
    controller_name, main_test_name, sub_test_name, _ = ex_file_name.split('-')
    new_file_name = "-".join([controller_name, main_test_name, sub_test_name, 'preprocessed_result.parquet'])

    df_list = []
    cnt = 0
    for chunk_parquet in chunk_parquet_list:
        # logger.info(f"Working on {chunk_parquet}")
        chunk_df = pd.read_parquet(os.path.join(base_preproc_input_dir, chunk_parquet))
        # logger.info(f"{chunk_parquet} shape: {chunk_df.shape}")
        # logger.info(f"{chunk_parquet} index[0]: {chunk_df.index[0]}")
        # logger.info(f"{chunk_parquet} index[-1]: {chunk_df.index[-1]}")
        
        
        df_list.append(chunk_df)
        cnt += 1
        # if cnt == 30:
        #     break
    df = pd.concat(df_list, axis=0)
    logger.info(f"Concated DF: {df.shape}")

    # 데이터 타입이 "object"인 칼럼 삭제 (string data를 들고 있는 칼럼)
    logger.info("1/6. Working on Object type delete")
    df = df.select_dtypes(exclude=['object'])

    # 모든 칼럼이 NULL인 곳 제거
    logger.info("2/6. Working on NULL col delete")
    df = df.dropna(axis=1, how='all')

    # Min-Max Scaling
    logger.info("3/6. Working on Min Max Scaling")
    original_index = df.index
    scaler = MinMaxScaler()
    df = pd.DataFrame(scaler.fit_transform(df), columns = df.columns, index = original_index)
    
    # 표준편차가 너무 작은 칼럼들 제거
    logger.info("4/6. Working on small STD delete")
    std_dev = df.std()
    df = df.loc[:, std_dev >= THRES_STD]

    # NULL 값 채우기 
    logger.info("5/6. Working on NULL filling")
    df = df.ffill()
    df = df.bfill()
    
    logger.info(f"Processed DF: {df.shape}")
    df.to_parquet(os.path.join(base_preproc_output_dir,new_file_name), engine='pyarrow')
    logger.info('6/6. Processed Result saved!')

