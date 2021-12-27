# -*- coding: utf-8 -*-
import sys
import json
import os
import warnings
import flask
import boto3
import io

import time
import jittor as jt
from jittor import Module
from jittor import nn
import numpy as np


# The flask app for serving predictions
app = flask.Flask(__name__)

s3_client = boto3.client('s3')


class Model(Module):
    def __init__(self):
        self.layer1 = nn.Linear(1, 10)
        self.relu = nn.Relu() 
        self.layer2 = nn.Linear(10, 1)
    def execute (self,x) :
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x
    
model = Model()
model.load('/opt/ml/model/net.pkl')

batch_size = 1

print('init done.')

def get_data(n): # generate random data for training test.
    for i in range(n):
        x = np.random.rand(batch_size, 1)
        y = x*x
        yield jt.float32(x), jt.float32(y)
        

@app.route('/ping', methods=['GET'])
def ping():
    """Determine if the container is working and healthy. In this sample container, we declare
    it healthy if we can load the model successfully."""
    # health = ScoringService.get_model() is not None  # You can insert a health check here
    health = 1

    status = 200 if health else 404
    # print("===================== PING ===================")
    return flask.Response(response="{'status': 'Healthy'}\n", status=status, mimetype='application/json')

@app.route('/invocations', methods=['POST'])
def invocations():
    """Do an inference on a single batch of data. In this sample server, we take data as CSV, convert
    it to a pandas data frame for internal use and then convert the predictions back to CSV (which really
    just means one prediction per line, since there's a single column.
    """
    data = None
    print("================ INVOCATIONS =================")

    #parse json in request
    print ("<<<< flask.request.content_type", flask.request.content_type)

#     if flask.request.content_type == 'application/x-image':
#         image_as_bytes = io.BytesIO(flask.request.data)
#         img = Image.open(image_as_bytes)
#         download_file_name = '/tmp/tmp.jpg'
#         img.save(download_file_name)
#         print ("<<<<download_file_name ", download_file_name)
#     else:
#         data = flask.request.data.decode('utf-8')
#         data = json.loads(data)

#         bucket = data['bucket']
#         image_uri = data['image_uri']

#         download_file_name = '/tmp/'+image_uri.split('/')[-1]
#         print ("<<<<download_file_name ", download_file_name)

#         try:
#             s3_client.download_file(bucket, image_uri, download_file_name)
#         except:
#             #local test
#             download_file_name = './bus.jpg'

#         print('Download finished!')

    for x, y in get_data(1):
        inference_result = model(x).numpy().tolist()
        
    _payload = json.dumps(inference_result,ensure_ascii=False)

    return flask.Response(response=_payload, status=200, mimetype='application/json')
