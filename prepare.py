import glob
import random
import re
from collections import Counter
import numpy as np
from utils import getVocabulary
from utils import modelSave
import os

class MailReader():
    def __init__(self):
        pass

    def create_bag_of_words(self,filePaths):
        '''
        Input:
          filePaths: Array. A list of absolute filepaths
        Returns:
          bagOfWords: Array. All tokens in files
        '''
        bagOfWords = []
        for filePath in filePaths:
            with open(filePath, encoding ="UTF-8") as f:
                raw = f.read()
                tokens=getVocabulary(raw,echo=False).split()
                for token in tokens:
                    if len(token) == 1:
                        if token.isalpha():
                            bagOfWords.append(token)
                    elif len(token)<20:
                        bagOfWords.append(token)
        return bagOfWords

    def get_feature_matrix(self,filePaths, featureDict):
        '''
        create feature/x matrix from multiple text files
        rows = files, cols = features
        '''
        featureMatrix = np.zeros(shape=(len(filePaths),
                                          len(featureDict)),
                                   dtype=float)
        for i,filePath in enumerate(filePaths):
            #print(filePath)
            with open(filePath, encoding ="UTF-8") as f:
                _raw = f.read()
                tokens = getVocabulary(_raw).split()
                fileUniDist = Counter(tokens)
                #print(fileUniDist)
                for key,value in fileUniDist.items():
                    if key in featureDict:
                        featureMatrix[i,featureDict[key]] = value
        return featureMatrix

    def regularize_vectors(self,featureMatrix):
        '''
        Input:
          featureMatrix: matrix, where docs are rows and features are columns
        Returns:
          featureMatrix: matrix, updated by dividing each feature value by the total
          number of features for a given document
        '''
        for i in range(featureMatrix.shape[0]):
            totalWords = np.sum(featureMatrix[i,:],axis=0)
            if totalWords>0:
                featureMatrix[i,:] = np.multiply(featureMatrix[i,:],(1/totalWords))
            else:
                print("error in :",i)
        return featureMatrix

    def input_data(self,hamDir,spamDir,percentTest,cutoff):
        ''' 
        Input:
          hamDir: String. dir of ham text files
          spamDir: String. dir of spam text file
          percentTest: Float. percentage of all data to be assigned to testset
        Returns:
          trainPaths: Array. Absolute paths to training emails
          trainY: Array. Training labels, 0 or 1 int.
          testPaths: Array. Absolute paths to testing emails
          testY: Array. Testing labels, 0 or 1 int.
        '''
        pathLabelPairs={}
        for hamPath in glob.glob(hamDir+'*'):
            pathLabelPairs.update({hamPath:(0,1)})
        for spamPath in glob.glob(spamDir+'*'):
            pathLabelPairs.update({spamPath:(1,0)})
        
        # get test set as random subsample of all data
        numTest = int(percentTest * len(pathLabelPairs))
        testing = set(random.sample(pathLabelPairs.items(),numTest))
        
        # delete testing data from superset of all data
        for entry in testing:
            del pathLabelPairs[entry[0]]

        # split training tuples of (path,label) into separate lists
        trainPaths=[]
        trainY=[]
        for item in pathLabelPairs.items():
            trainPaths.append(item[0])
            trainY.append(item[1])

        # split testing tuples of (path,label) into separate lists
        testPaths=[]
        testY=[]
        for item in testing:
            testPaths.append(item[0])
            testY.append(item[1])

        # create feature dictionary of n-grams
        bagOfWords = self.create_bag_of_words(trainPaths)
        #print(bagOfWords)
        # throw out low freq words
        freqDist = Counter(bagOfWords)
        newBagOfWords=[]
        for word,freq in freqDist.items():
            if freq > cutoff:
                newBagOfWords.append(word)
        features = set(newBagOfWords)
        featureDict = {feature:i for i,feature in enumerate(features)}
        # make feature matrices
        trainX = self.get_feature_matrix(trainPaths,featureDict)
        testX = self.get_feature_matrix(testPaths,featureDict)

        # regularize length
        trainX = self.regularize_vectors(trainX)
        testX = self.regularize_vectors(testX)
        #print(trainX[0])
        # cast as ndarrays
        trainY = np.asarray(trainY)
        testY = np.asarray(testY)

        return trainX, trainY, testX, testY,featureDict


def parse_user_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-ham','--hamDir')
    parser.add_argument('-spam','--spamDir')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    import sys, argparse
    # get user input
    args = parse_user_args()
    hamDir = args.hamDir
    spamDir= args.spamDir

    reader = MailReader()
    
    trainX,trainY,testX,testY,features = reader.input_data(hamDir=hamDir,
                                                  spamDir=spamDir,
                                                  percentTest=.1,
                                                  cutoff=15)

    print(trainX.shape)
    print(trainY.shape)    
    print(testX.shape)
    print(testY.shape)    
    files = glob.glob('./data/.*')
    for f in files:
        os.remove(f)
    np.savetxt("./data/trainX.csv", trainX, delimiter="\t")
    np.savetxt("./data/trainY.csv", trainY, delimiter="\t")
    np.savetxt("./data/testX.csv", testX, delimiter="\t")
    np.savetxt("./data/testY.csv", testY, delimiter="\t")
    modelSave(features,"./data/features.pkl")