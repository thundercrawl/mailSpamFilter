import grpc
import tensorflow as tf
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
from tensorflow.python.framework import tensor_util
import numpy as np
from utils import getVocabulary
from utils import loadModel
from collections import Counter
import argparse
#define the parameter
server = 'localhost:8500'

def convertMAT(mailtopath,features):
    featurevector = np.zeros(shape=(1,len(features)),
                                   dtype=float)
    #print(featurevector)
    with open(mailtopath, encoding ="UTF-8") as f:
        _raw = f.read()
        tokens = getVocabulary(_raw).split()
        fileUniDist = Counter(tokens)
        for key,value in fileUniDist.items():
            if key in features:
                featurevector[0,features[key]] = value
        totalWords = np.sum(featurevector[0],axis=0)
        print("totalwords:",totalWords)
        if totalWords>0:
            featurevector = np.multiply(featurevector,(1/totalWords))
        else:
            print("error in features!")


    return featurevector

def predictGRPC(server,path2mail):
    #load mail feature
    vector=convertMAT(path2mail,loadModel("./data/features.pkl"))
    # create the RPC stub
    channel = grpc.insecure_channel(server)
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'mlmodel'
    request.model_spec.signature_name = 'predict'
    request.inputs['emails'].CopyFrom(tf.contrib.util.make_tensor_proto(vector,dtype=tf.float32))
    #request.inputs['scores'].CopyFrom(tf.contrib.util.make_tensor_proto([0.,0.]))

    result_future = stub.Predict(request)
    scores = result_future.outputs['scores']

    print("mail is:",scores)
    print(type(scores))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mail",
        type=str,
        default="",
        help="mail file path as txt -UTF8"
    )
    predictGRPC(server,parser.parse_args().mail)