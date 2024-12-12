import argparse
import os
import random
import shutil
import sys

import logging
import logging.handlers

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
    parser.add_argument('--base_output_dir', type=str, default='/opt/ml/processing/output')
    parser.add_argument('--base_preproc_input_dir', type=str, default='/opt/ml/processing/input')
    parser.add_argument('--split_rate', type=float, default=0.1)
    
    # parse arguments
    args = parser.parse_args()
    logger.info('#### Argument Info ####')
    logger.info(f"args.base_output_dir: {args.base_output_dir}")
    logger.info(f"args.base_preproc_input_dir: {args.base_preproc_input_dir}")
    logger.info(f"args.split_rate: {args.split_rate}")

    base_output_dir = args.base_output_dir
    base_preproc_input_dir = args.base_preproc_input_dir
    split_rate = args.split_rate

    # shutil 라이브러리를 이용해서 개, 고양이 사진을 9:1 비율로 test / train으로 나눠서 저장

    # 원래 데이터 있던 곳
    raw_cat_dir = f"{base_preproc_input_dir}/cat"
    raw_dog_dir = f"{base_preproc_input_dir}/dog"

    # 데이터 저장할 곳
    output_train_cat_dir = f"{base_output_dir}/train/cat"
    output_train_dog_dir = f"{base_output_dir}/train/dog"
    output_test_cat_dir = f"{base_output_dir}/test/cat"
    output_test_dog_dir = f"{base_output_dir}/test/dog"

    # 디렉토리 생성
    os.makedirs(output_train_cat_dir, exist_ok=True)
    os.makedirs(output_train_dog_dir, exist_ok=True)
    os.makedirs(output_test_cat_dir, exist_ok=True)
    os.makedirs(output_test_dog_dir, exist_ok=True)

    # 원본 이미지 가져오기
    cat_images = [file for file in os.listdir(raw_cat_dir) if file.endswith(".jpg")]
    dog_images = [file for file in os.listdir(raw_dog_dir) if file.endswith(".jpg")]

    # 셔플
    random.shuffle(cat_images)
    random.shuffle(dog_images)

    # split_rate에 따라서 분할하기
    raw_cat_num = len(cat_images)
    raw_dog_num = len(dog_images)

    train_cat_num = int(raw_cat_num * split_rate)
    train_dog_num = int(raw_dog_num * split_rate)

    train_cat_images = cat_images[:train_cat_num]
    train_dog_images = dog_images[:train_dog_num]
    test_cat_images = cat_images[train_cat_num:]
    test_dog_images = dog_images[train_cat_num:]

    for image in train_cat_images:
        shutil.copy(os.path.join(raw_cat_dir, image), output_train_cat_dir)

    for image in train_dog_images:
        shutil.copy(os.path.join(raw_dog_dir, image), output_train_dog_dir)

    for image in test_cat_images:
        shutil.copy(os.path.join(raw_cat_dir, image), output_test_cat_dir)

    for image in test_dog_images:
        shutil.copy(os.path.join(raw_dog_dir, image), output_test_dog_dir)

    logger.info("Dataset Split Complete!")
    logger.info(f"Before: Cat: {raw_cat_num}, Dog: {raw_dog_num}")
    logger.info(f"After: <Train> Cat: {len(os.listdir(output_train_cat_dir))}, Dog: {len(os.listdir(output_train_dog_dir))}")
    logger.info(f"        <Test> Cat: {len(os.listdir(output_test_cat_dir))}, Dog: {len(os.listdir(output_test_dog_dir))}")