

from __future__ import division
import tensorflow as tf
import numpy as np
import tarfile
import os
import matplotlib.pyplot as plt
import time
import json
import argparse
import mlflow
import mlflow.tensorflow
from tensorflow.python.saved_model import tag_constants
from utils import cleanPath

FLAGS = None
log_dir = '/logdir'

tf_config = tf.ConfigProto()
tf_config.intra_op_parallelism_threads = 44
tf_config.inter_op_parallelism_threads = 44
export_dir="./model"



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
def tfmain(server,n,multi=True,num_epochs=25000):
    trainX,trainY,testX,testY = import_data()

    print("shape",trainX.shape,trainY.shape)
    builder=tf.saved_model
    numFeatures = trainX.shape[1]
    numLabels = trainY.shape[1]
    numEpochs = int(num_epochs/n)
    learningRate = tf.train.exponential_decay(learning_rate=0.008,
                                            global_step= 1,
                                            decay_steps=trainX.shape[0],
                                            decay_rate= 0.95,
                                            staircase=True)
    X = tf.placeholder(tf.float32, [None, numFeatures])
    # yGold = Y-matrix / label-matrix / labels... This will be our correct answers
    # matrix. Every row has either [1,0] for SPAM or [0,1] for HAM. 'None' here 
    # means that we can hold any number of emails
    yGold = tf.placeholder(tf.float32, [None, numLabels])


    #################
    ### VARIABLES ###
    #################

    # Values are randomly sampled from a Gaussian with a standard deviation of:
    #     sqrt(6 / (numInputNodes + numOutputNodes + 1))

    weights = tf.Variable(tf.random_normal([numFeatures,numLabels],
                                        mean=0,
                                        stddev=(np.sqrt(6/(numFeatures+
                                                            numLabels+1))),
                                        name="weights"))
    
    bias = tf.Variable(tf.random_normal([1,numLabels],
                                        mean=0,
                                        stddev=(np.sqrt(6/(numFeatures+numLabels+1))),
                                        name="bias"))


    ######################
    ### PREDICTION OPS ###
    ######################

    # INITIALIZE our weights and biases
    init_OP = tf.global_variables_initializer()

    # PREDICTION ALGORITHM i.e. FEEDFORWARD ALGORITHM
    apply_weights_OP = tf.matmul(X, weights, name="apply_weights")
    #apply_weights_OP=X*weights
    add_bias_OP = tf.add(apply_weights_OP, bias, name="add_bias") 
    activation_OP = tf.nn.sigmoid(add_bias_OP, name="activation")


    #####################
    ### EVALUATION OP ###
    #####################

    # COST FUNCTION i.e. MEAN SQUARED ERROR
    cost_OP = tf.nn.l2_loss(activation_OP-yGold, name="squared_error_cost")


    #######################
    ### OPTIMIZATION OP ###
    #######################

    # OPTIMIZATION ALGORITHM i.e. GRADIENT DESCENT
    training_OP = tf.train.GradientDescentOptimizer(learningRate).minimize(cost_OP)

    ## Ops for vizualization
    # argmax(activation_OP, 1) gives the label our model thought was most likely
    # argmax(yGold, 1) is the correct label
    correct_predictions_OP = tf.equal(tf.argmax(activation_OP,1),tf.argmax(yGold,1))
    # False is 0 and True is 1, what was our average?
    accuracy_OP = tf.reduce_mean(tf.cast(correct_predictions_OP, "float"))
    # Summary op for regression output
    activation_summary_OP = tf.summary.histogram("output", activation_OP)
    # Summary op for accuracy
    accuracy_summary_OP = tf.summary.scalar("accuracy", accuracy_OP)
    # Summary op for cost
    cost_summary_OP = tf.summary.scalar("cost", cost_OP)
    

    ###########################
    ### GRAPH LIVE UPDATING ###
    ###########################
    
    epoch_values=[]
    accuracy_values=[]
    cost_values=[]
    # Turn on interactive plottingm
 

    #####################
    ### RUN THE GRAPH ###
    #####################
    is_chief = (FLAGS.task_index == 0)
    # Create a tensorflow session
    #sess=tf.Session()
    if multi:
        sess=tf.Session(
        target=server.target,
        config=tf_config)
    else:
        sess=tf.Session(config=tf_config)
        # Summary ops to check how variables (W, b) are updating after each iterationemails
        #weightSummary = tf.summary.histogram("weights", weights.eval(session=sess))emails
        #biasSummary = tf.summary.histogram("biases", bias.eval(session=sess))
        # Initialize all tensorflow variables
        sess.run(init_OP)
        # Merge all summaries
        #all_summary_OPS = tf.summary.merge_all()
        # Summary writer
        #writer = tf.summary.FileWriter("summary_logs", sess.graph)

        # Initialize reporting variables
        cost = 0
        diff = 1

        # Training epochs
        for i in range(numEpochs):
            if i > 1 and diff < .0001:
                print("change in cost %g; convergence."%diff)
                break
            else:
                # Run training step
                step = sess.run(training_OP, feed_dict={X: trainX, yGold: trainY})
                # Report occasional stats
                if i % 10 == 0:
                    # Add epoch to epoch_values
                    epoch_values.append(i)
                    # Generate accuracy stats on test data
                    train_accuracy, newCost = sess.run(
                        [ accuracy_OP, cost_OP], 
                        feed_dict={X: trainX, yGold: trainY}
                    )
                    # Add accuracy to live graphing variable
                    accuracy_values.append(train_accuracy)
                    # Add cost to live graphing variable
                    cost_values.append(newCost)
                    # Write summary stats to writer
                    #writer.add_summary(summary_results, i)
                    # Re-assign values for variables
                    diff = abs(newCost - cost)
                    cost = newCost

                    #generate print statements
                    print("step %d, training accuracy %g"%(i, train_accuracy))
                    print("step %d, cost %g"%(i, newCost))
                    print("step %d, change in cost %g"%(i, diff))

        
                    #time.sleep(1)

        
        
        saver = tf.train.Saver()
        # Save variables to .ckpt file
        saver.save(sess, "trained_variables.ckpt")
        # How well do we perform on held-out test data?
        print("final accuracy on test set: %s" %str(sess.run(accuracy_OP, 
                                                                feed_dict={X: testX, 
                                                                    yGold: testY})))
        #add builder saver
        cleanPath(export_dir)
        export_path_base = export_dir
        export_path = os.path.join(
        tf.compat.as_bytes(export_path_base),
        tf.compat.as_bytes(str(1)))
        builder = tf.saved_model.builder.SavedModelBuilder(export_path)
        
       # Build the signature_def_map, not provide the classificate signature currently, 
       #will support it  in future, that provide production mutrainning maybe
        # classification_inputs = tf.saved_model.utils.build_tensor_info(
        #     X)
        # classification_outputs_classes = tf.saved_model.utils.build_tensor_info(
        #     prediction_classes)
        # classification_outputs_scores = tf.saved_model.utils.build_tensor_info(values)

        # classification_signature = (
        #     tf.saved_model.signature_def_utils.build_signature_def(
        #         inputs={
        #             tf.saved_model.signature_constants.CLASSIFY_INPUTS:
        #                 classification_inputs
        #         },
        #         outputs={
        #             tf.saved_model.signature_constants.CLASSIFY_OUTPUT_CLASSES:
        #                 classification_outputs_classes,
        #             tf.saved_model.signature_constants.CLASSIFY_OUTPUT_SCORES:
        #                 classification_outputs_scores
        #         },
        #         method_name=tf.saved_model.signature_constants.CLASSIFY_METHOD_NAME))

        tensor_info_x = tf.saved_model.utils.build_tensor_info(X)
        tensor_info_y = tf.saved_model.utils.build_tensor_info(activation_OP)

        prediction_signature = (
            tf.saved_model.signature_def_utils.build_signature_def(
                inputs={'emails': tensor_info_x},
                outputs={'scores': tensor_info_y},
                method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))

        builder.add_meta_graph_and_variables(
            sess, [tf.saved_model.tag_constants.SERVING],
            signature_def_map={
                'predict':
                    prediction_signature
            })

        builder.save(as_text=True)
                


        sess.close()

    print("close training,exit")
def main():

    ########################
    ##Distribution approach
    ########################

    with open('TF_CONFIG.json') as json_file:
        tfconfig = json.load(json_file)
        #os.environ["TF_CONFIG"] = json.dumps(tfconfig)
        cluster=tfconfig['cluster']
        print("multi:",FLAGS.multi)
        if FLAGS.multi == 2:
            tfmain(None,1,multi=False,num_epochs=FLAGS.numEpochs)
        elif FLAGS.job_name == 'ps':  # checks if parameter server
            server = tf.train.Server(cluster,
                                    job_name="ps",
                                    task_index=FLAGS.task_index)
            server.join()
        else:
            # checks if this is the chief node
            #is_chief = (FLAGS.task_index == 0)
            server = tf.train.Server(cluster,
                                    job_name="worker",
                                    task_index=FLAGS.task_index)
            tfmain(server,len(cluster["worker"]),num_epochs=FLAGS.numEpochs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--multi",type=int,default=1,help="1 = multi, 2 = single")
    parser.add_argument("--numEpochs",type=int,default=25000)
    # Flags for defining the tf.train.ClusterSpec
    parser.add_argument(
        "--job_name",
        type=str,
        default="",
        help="One of 'ps', 'worker'"
    )
  # Flags for defining the tf.train.Server
    parser.add_argument(
        "--task_index",
        type=int,
        default=0,
        help="Index of task within the job"
    )
    FLAGS, unparsed = parser.parse_known_args()
    main()
    #  # Create ML saver Saver
    # with mlflow.start_run():
    #     mlflow.tensorflow.log_model(tf_saved_model_dir=export_dir,
    #                                 tf_meta_graph_tags=[tag_constants.SERVING],
                                    
    #                                 tf_signature_def_key="predict",
    #                                 artifact_path="mlmodel")