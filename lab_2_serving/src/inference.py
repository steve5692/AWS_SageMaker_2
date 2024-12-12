import os
import time
import json
import pickle as pkl
import numpy as np
from io import BytesIO
import xgboost as xgb
import sagemaker_xgboost_container.encoder as xgb_encoders
NUM_FEATURES = 58

# inference.py 에서는 model_fn, input_fn, predict_fn, output_fn 이 4개를 꼭 포함해줘야 한다.

def model_fn(model_dir):
    """
    Deserialize and return fitted model.
    S3에 내가 저장해놨던 모델 아티팩트를 로딩함. (model.tar.gz)
    """
    model_file = "xgboost-model"
    model = xgb.Booster()
    model.load_model(os.path.join(model_dir, model_file))
    return model
                     

def input_fn(request_body, request_content_type):
    """
    The SageMaker XGBoost model server receives the request data body and the content type,
    and invokes the `input_fn`.
    Return a DMatrix (an object that can be passed to predict_fn).

    사용자로 부터 얻어온 request --> parsing --> 전처리
    
    """
    print("Content type: ", request_content_type)
    if request_content_type == "application/x-npy":    # input이 numpy 일 때     
        stream = BytesIO(request_body)
        array = np.frombuffer(stream.getvalue())
        array = array.reshape(int(len(array)/NUM_FEATURES), NUM_FEATURES)
        return xgb.DMatrix(array)
    elif request_content_type == "text/csv":            # input이 csv 일 때
        return xgb_encoders.csv_to_dmatrix(request_body.rstrip("\n"))
    elif request_content_type == "text/libsvm":         # input이 libsvm 일 떄
        return xgb_encoders.libsvm_to_dmatrix(request_body)
    else:
        raise ValueError(
            "Content type {} is not supported.".format(request_content_type)
        )
        

def predict_fn(input_data, model):
    """
    SageMaker XGBoost model server invokes `predict_fn` on the return value of `input_fn`.

    Return a two-dimensional NumPy array (predictions and scores)
    inpuf_fn 에서 받아온 데이터를 모델에 넣어서 prediction 내어주면 됨.
    """
    start_time = time.time()
    y_probs = model.predict(input_data)
    print("--- Inference time: %s secs ---" % (time.time() - start_time))    
    y_preds = [1 if e >= 0.5 else 0 for e in y_probs] 
    #feature_contribs = model.predict(input_data, pred_contribs=True, validate_features=False)
    return np.vstack((y_preds, y_probs))


def output_fn(predictions, content_type="application/json"):
    """
    After invoking predict_fn, the model server invokes `output_fn`.

    predictions = predict_fn의 return 값.
    이걸 내가 원하는 type으로 바꿔서 내보내는 것.
    """
    if content_type == "text/csv":
        return ','.join(str(x) for x in outputs)
    elif content_type == "application/json":
        outputs = json.dumps({
            'pred': predictions[0,:].tolist(),
            'prob': predictions[1,:].tolist()
        })        
        
        return outputs
    else:
        raise ValueError("Content type {} is not supported.".format(content_type))
