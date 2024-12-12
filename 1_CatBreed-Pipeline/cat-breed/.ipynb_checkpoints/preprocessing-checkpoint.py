import argparse
import os
import requests
import tempfile
import shutil
from glob import glob
import sys

import logging
import logging.handlers

from sklearn.model_selection import train_test_split


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


if __name__ == "__main__":
    '''
    커맨드 인자 파싱
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_preproc_input_dir', type=str, default='/opt/ml/processing/input')
    parser.add_argument('--base_preproc_output_dir', type=str, default='/opt/ml/processing')
    parser.add_argument('--test_split_rate', type=float, default=0.1)

    
    # parse arguments
    args = parser.parse_args()
    logger.info('#### Argument Info ####')
    logger.info(f"args.base_preproc_input_dir: {args.base_preproc_input_dir}")
    logger.info(f"args.base_preproc_output_dir: {args.base_preproc_output_dir}")
    logger.info(f"args.test_split_rate: {args.test_split_rate}")

    base_preproc_input_dir = args.base_preproc_input_dir
    base_preproc_output_dir = args.base_preproc_output_dir
    test_split_rate = args.test_split_rate
    
    
    #imgs 가져오기
    img_paths = glob(f"{base_preproc_input_dir}/*/*.jpg") # 모든 .jpg들 다 부르기
    img_labels = [os.path.splitext(label)[0].split('/')[-2] for label in img_paths]

    logger.info(f"img_path lenth: {len(img_paths)}")
    
    # train / test 나누기
    X_train, X_test, Y_train, Y_test = train_test_split(img_paths, img_labels, test_size=test_split_rate, random_state=42)

    # output 경로 설정
    os.makedirs(f"{base_preproc_output_dir}/train/", exist_ok=True)
    os.makedirs(f"{base_preproc_output_dir}/test/", exist_ok=True)

    uniq_labels = list(set(img_labels))
    for uniq_label in uniq_labels:
        os.makedirs(f"{base_preproc_output_dir}/train/{uniq_label}", exist_ok=True)
        os.makedirs(f"{base_preproc_output_dir}/test/{uniq_label}", exist_ok=True)

    for path, label in zip(X_train, Y_train):
        shutil.copy(path, f"{base_preproc_output_dir}/train/{label}")
        # logger.info(f"Move {path} to {base_preproc_output_dir}/train/{label}")

    for path, label in zip(X_test, Y_test):
        shutil.copy(path, f"{base_preproc_output_dir}/test/{label}")
        # logger.info(f"Move {path} to {base_preproc_output_dir}/test/{label}")

    logger.info("Split Done!")
    for cat in ["train", "test"]:
        for lab in uniq_labels:
            logger.info(f"{cat}-{lab} Images: {len(glob(f'{base_preproc_output_dir}/{cat}/{lab}/*.jpg'))}")





