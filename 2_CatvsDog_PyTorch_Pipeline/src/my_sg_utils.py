
def get_proc_artifact(execution, client, kind):
    
    '''
    preprocess 후 전처리 결과물의 S3경로를 획득
    kind = 0 --> train
    kind = 1 --> test
    '''
    response = execution.list_steps()

    proc_arn = response[-1]['Metadata']['ProcessingJob']['Arn']
    # print(proc_arn)
    
    proc_job_name = proc_arn.split('/')[-1]
    # print(proc_job_name)

    response = client.describe_processing_job(ProcessingJobName = proc_job_name)
    test_preprocessed_file = response['ProcessingOutputConfig']['Outputs'][kind]['S3Output']['S3Uri']
    return test_preprocessed_file