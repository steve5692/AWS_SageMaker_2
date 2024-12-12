import json
import boto3
import time

def lambda_handler(event, context):
    sm_client = boto3.client("sagemaker")

    current_time = time.strftime('%m-%d-%H-%M-%S', time.localtime())
    model_name = event["model_name"] + current_time
    model_package_arn = event["model_package_arn"]
    endpoint_config_name = event["endpoint_config_name"]+current_time
    endpoint_name=event["endpoint_name"]
    role = event["role"]

    container = {"ModelPackageName": model_package_arn}

    create_model_response = sm_client.create_model(ModelName=model_name,
                                                 ExecutionRoleArn=role,
                                                 PrimaryContainer=container)

    create_endpoint_config_response = sm_client.create_endpoint_config(
        EndpointConfigName=endpoint_config_name,
        ProductionVariants=[
            {
                "InstanceType": "ml.m5.large",
                "InitialVariantWeight": 1,
                "InitialInstanceCount": 1,
                "ModelName": model_name,
                "VariantName": "AllTraffic",
            }
        ]
    )

    try:
        create_endpoint_response = sm_client.create_endpoint(EndpointName=endpoint_name,
                                                            EndpointConfigName=endpoint_config_name)
    except Exception as e:
        print(e)
        print("Update Endpoint!")

        sm_client.update_endpoint(EndpointName=endpoint_name, EndpointConfigName=endpoint_config_name)

    return {
        "statusCode": 200,
        "body": json.dumps("Created Endpoint!")
    }
















