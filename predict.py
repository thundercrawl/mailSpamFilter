import numpy as np
import tensorflow as tf
import tarfile
import os
import argparse
from utils import loadModel
from collections import Counter
from utils import getVocabulary
import json

def MODEL_REST_CALL( uri, featureVector):
       
    curstr = "{\"columns\":["
    columns = ""
    for i in dictionary:
        columns = columns + "\"" + i + "\","

    curstr = curstr + columns[0:-1] + "],"

    dic={}
    dic['data']=features_test.tolist()
    dicJson = json.dumps(dic)
    json1 = dicJson.replace("{","")
    json2 = json1.replace("}","")
    curstr = curstr + json2 + "}"

    savejson_to_file(path[path.rfind("/") + 1:] + ".log",curstr)
    response = requests.post(url=uri,
                             data=curstr,
                             headers={"Content-Type": "application/json; format=pandas-split"})

    if response.status_code != 200:
        raise Exception("Status Code {status_code}. {text}".format(
            status_code=response.status_code,
            text=response.text
        ))
    return response

def covertMAT(mailtopath,features):
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

def csv_to_numpy_array(filePath, delimiter):
    return np.genfromtxt(filePath, delimiter=delimiter, dtype=None)

def import_data():
    if "data" not in os.listdir(os.getcwd()):
        # Untar directory of data if we haven't already
        tarObject = tarfile.open("data.tar.gz")
        tarObject.extractall()
        tarObject.close()
        print("Extracted tar to current directory")
    else:
        # we've already extracted the files
        pass

    print("loading training data")
    trainX = csv_to_numpy_array("data/trainX.csv", delimiter="\t")
    trainY = csv_to_numpy_array("data/trainY.csv", delimiter="\t")
    print("loading test data")
    testX = csv_to_numpy_array("data/testX.csv", delimiter="\t")
    testY = csv_to_numpy_array("data/testY.csv", delimiter="\t")
    return trainX,trainY,testX,testY


trainX,trainY,testX,testY = import_data()
numFeatures = trainX.shape[1]
numLabels = trainY.shape[1]
print(numFeatures,numLabels)
#create a tensorflow session
sess = tf.Session()


####################
### PLACEHOLDERS ###
####################

# X = X-matrix / feature-matrix / data-matrix... It's a tensor to hold our email
# data. 'None' here means that we can hold any number of emails
X = tf.placeholder(tf.float32, [None, numFeatures])
# yGold = Y-matrix / label-matrix / labels... This will be our correct answers
# matrix. Every row has either [1,0] for SPAM or [0,1] for HAM. 'None' here
# means that we can hold any number of emails
yGold = tf.placeholder(tf.float32, [None, numLabels])


#################
### VARIABLES ###
#################

#all values must be initialized to a value before loading can occur

weights = tf.Variable(tf.zeros([numFeatures,numLabels]))

bias = tf.Variable(tf.zeros([1,numLabels]))

########################
### OPS / OPERATIONS ###
########################

#since we don't have to train the model, the only Ops are the prediction operations

apply_weights_OP = tf.matmul(X, weights, name="apply_weights")
add_bias_OP = tf.add(apply_weights_OP, bias, name="add_bias")
activation_OP = tf.nn.sigmoid(add_bias_OP, name="activation")


# argmax(activation_OP, 1) gives the label our model thought was most likely
# argmax(yGold, 1) is the correct label
correct_predictions_OP = tf.equal(tf.argmax(activation_OP,1),tf.argmax(yGold,1))

# False is 0 and True is 1, what was our average?
accuracy_OP = tf.reduce_mean(tf.cast(correct_predictions_OP, "float"))

# Initializes everything we've defined made above, but doesn't run anything
# until sess.run()
init_OP = tf.initialize_all_variables()

sess.run(init_OP)       #initialize variables BEFORE loading

#load variables from file
saver = tf.train.Saver()
saver.restore(sess, "trained_variables.ckpt")

#####################
### RUN THE GRAPH ###
#####################

# Initialize all tensorflow objects
# sess.run(init_OP)

#method for converting tensor label to string label
def labelToString(label):
    if np.argmax(label) == 1:
        return "ham"
    else:
        return "spam"



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Flags for defining the tf.train.ClusterSpec
    parser.add_argument(
        "--type",
        type=str,
        default="",
        help="local|http"
    )
    parser.add_argument(
        "--uri",
        type=str,
        default="",
        help="http://localhost:51000/invocations"
    )
    parser.add_argument(
        "--mail",
        type=str,
        default="",
        help="mail file path as txt -UTF8"
    )
    if parser.parse_args().type== 'http':
        MODEL_REST_CALL(parser.parse_args().uri,)
    elif parser.parse_args().mail == '':
        #show predictions and accuracy of entire test set
        prediction, evaluation = sess.run([activation_OP, accuracy_OP], feed_dict={X: testX, yGold: testY})

        for i in range(len(testX)):
            print("predicts email %s as %s actually: %s -- %s" %(str(i + 1), labelToString(prediction[i]), labelToString(testY[i]),labelToString(prediction[i])==labelToString(testY[i])))
        print("overall accuracy of dataset: %s percent" %str(evaluation))
    else:
        featurevect=covertMAT(parser.parse_args().mail,loadModel("./data/features.pkl"))
        prediction = sess.run([activation_OP],feed_dict={X: featurevect})
        print(prediction[0])
        print("predict mail(%s) as %s" %(parser.parse_args().mail,labelToString(prediction[0])))
