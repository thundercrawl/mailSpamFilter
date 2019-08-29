import jieba
import pickle
import os, shutil


def modelSave(model,pathtofile):
    pickle.dump(model, open(pathtofile, 'wb'))
def loadModel(pathtofile):
    return pickle.load(open(pathtofile,'rb'))

def getVocabulary(line,echo=False):
    sentence_seged = jieba.cut(line)  
    outstr = ''  
    for word in sentence_seged:  
        if word not in stopwords:  
            if word != '\t':
                if len(word) == 1:
                    if word.isalpha():
                        outstr += word
                        outstr += " "   
                elif len(word)<20:
                        outstr += word
                        outstr += " " 
                 
                
    if  echo:
        print(outstr)
    return outstr
def stopwordslist(stopfilepath):
    stopwords = [line.strip() for line in open(stopfilepath,encoding='UTF-8').readlines()]
    return stopwords
def cleanPath(c_p):
    folder = c_p
    if os.path.isdir(folder):
        for the_file in os.listdir(folder):
            file_path = os.path.join(folder, the_file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path): shutil.rmtree(file_path)
            except Exception as e:
                print(e)
    else:
        print("not a directory:",c_p)

stopwords=stopwordslist('./stopwords/chineseStopWords.txt')
