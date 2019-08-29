import tensorflow as tf
from tensorflow.python.saved_model import tag_constants
def loadModel(path):
    print("load model from disk:",path)
    with tf.Session() as sess:
        tf.saved_model.loader.load(sess,[tag_constants.SERVING],path)
        

if __name__ == '__main__':
    path="./model"
    loadModel(path)

