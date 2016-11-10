import tensorflow as tf
import numpy as np
import os, time
from myCNN_utils import *
from data_utils import *
import pickle


#with open('1029_1545.pickle') as f: 
with open('1107_1202.pickle') as f:  
    testData, saveAtDIR, Params, dictLen = pickle.load(f)

graphCNN = myCNN_graph(seqLen=Params['SeqLen'], classSize=len(Params['pickGenreSet']), vocabSize=dictLen, 
						embeddingSize=Params['embeddingSize'], filterList=Params['filterList'], 
							filterNum=Params['filterNum'], learning_rate=Params['learnRate'])

#===========Test the model===========
print("Test the learned model:--")
# updated testCNN such that testing in applied on each batch of data: if applied on entire testing data ==> memory overflows
testCNN(graphCNN,testData,checkpoint=saveAtDIR)
