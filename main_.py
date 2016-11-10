import tensorflow as tf
import numpy as np
import os, time
from myCNN_utils import *
from data_utils import *
from nltk.stem.porter import PorterStemmer
import pickle
#===========================================================================================================================================
Params=dict(
	SeqLen=800,
	trainRatio=0.8,
	embeddingSize=256,
	filterList=[1,3,5,10],
	filterNum=100, #50,
	learnRate=0.001,
	num_epoch=5,
	batch_size=20,
	keepRate=0.5,#0.5# 0.3
	tableName='lyrics_a_to_z',
	pickGenreSet=['Rock','Hip Hop/Rap','Pop', 'R&B', 'Country'], 	# only pick lyrics of in those sets
	#pickGenreNum=[3000,3000,3000,2000,2000]							# num of songs per genre, if pool has fewer, will pick the maximum avaiable
	pickGenreNum=[10000,8000,8000,5000,7000]
)
#===========================================================================================================================================
# need reset graph, otherwise will gives variable reuse exception
def reset_graph():	
	if 'sess' in globals() and sess:
		sess.close()
	tf.reset_default_graph()
def split_train_test_data(data, trainRatio):
	data=np.array(data)		# must make it np.array then can shuffle
	data=data[np.random.permutation(range(len(data)))]
	idx=int(trainRatio*len(data))
	trainD= data[0:idx]
	testD=data[idx:]
	return trainD, testD
def wordStemmer(lyrics):
	print("Processing word stemmer.")
	stemmer=PorterStemmer()
	text=[]
	for t in lyrics:
		stemmed = [stemmer.stem(i) for i in t]
		text.append(stemmed)
	print("word stemmer completes.")
	return text	
#===========================================================================================================================================
#===========================================================================================================================================
# lyrics: list of list of words
lyrics, labels = get_Lyric_Genre(dataBaseName ='lyrics', tableName=Params['tableName'], 
										genreSet=Params['pickGenreSet'], pickNum=Params['pickGenreNum'])		
# stemmer words
lyrics=wordStemmer(lyrics)
lyrics = textCutPad(lyrics, length=Params['SeqLen'])
lyrics, word2id,id2word, = build_corpus(lyrics)
data=zip(lyrics,labels)
trainData, testData = split_train_test_data(data, trainRatio=Params['trainRatio'])
#===========================================================================================================================================
#===========build graph==============
graphCNN = myCNN_graph(seqLen=Params['SeqLen'], classSize=len(Params['pickGenreSet']), vocabSize=len(word2id), 
						embeddingSize=Params['embeddingSize'], filterList=Params['filterList'], 
							filterNum=Params['filterNum'], learning_rate=Params['learnRate'])
#===========train the CNN============
sig=time.strftime("%m%d_%H%M")
saveAtDIR='./saves/'+sig+'/myTestModel'
trainCNN(graphCNN, trainData, num_epoch=Params['num_epoch'], batch_size=Params['batch_size'], keepRate=Params['keepRate'], saveAt=saveAtDIR)


#===========Test the model===========
#testCNN(graphCNN,testData,checkpoint=saveAtDIR)
print('== No testing is performed ==')

#===========save the data============
pName=sig+'.pickle'
dictLen=len(word2id)
with open(pName, 'w') as f:  
    pickle.dump([testData, saveAtDIR, Params, dictLen], f)
