""" 
Build CNN graph:
	1. convolution
	2. max pool
	3. 

"""
import tensorflow as tf
import numpy as np
import os
def fetch_batch(rawData, batchSize, epochs):
	"""generate mini batch"""
	#rawData=np.array(rawData)
	dataLen=len(rawData)
	numBatch=int(len(rawData)/batchSize)+1
	for i in range(epochs):
		for j in range(numBatch):
			startId=j*batchSize
			endId=startId+batchSize
			if endId>dataLen:
				endId=dataLen
			if startId<endId: # avoid empty list
				yield rawData[startId:endId]
#===========================================================================================================================================				
def convCNN(inputs, convFilter, bias, stride=1):
	conv = tf.nn.conv2d(inputs, convFilter,strides=[1,stride,stride,1],padding='VALID')
	convBiased=tf.nn.bias_add(conv, bias)
	activation = tf.nn.relu(convBiased)
	return activation
#===========================================================================================================================================	
def mpCNN(inputs, k, stride=1):
	maxpool = tf.nn.max_pool(inputs, ksize=[1,k,1,1], strides=[1, stride, stride,1], padding='VALID')
	return maxpool
#===========================================================================================================================================		
def myCNN_graph(seqLen, classSize, vocabSize, embeddingSize, filterList, filterNum, learning_rate=0.001):
	print("Building graph.")
	x = tf.placeholder(tf.int32,[None, seqLen])		## input: take any number of sentences, with each of length=seqLen, where each word is int
	y = tf.placeholder(tf.int32,[None, classSize])	## output: corresponding class lable (one-hot) of the input
	keepRate = tf.placeholder(tf.float32, name='keepRate')			## keep rate of dropout regularization layer
	# Look at embeddings:
	embedding = tf.get_variable('Embedding_matrix',[vocabSize, embeddingSize])
	inputs = tf.nn.embedding_lookup(embedding, x)	## shape is [ None, seqLen, embeddingSize]
	inputs = tf.expand_dims(inputs, -1)				## shape is	[ None, seqLen, embeddingSize, 1]
	
	maxPool=[]
	# Convolution + max-pooling
	for f in filterList:
		with tf.variable_scope("conv-filter-size-%s" %f): ## variable_scope() allows to add prefix to variable defined by get_variable() 
			filterShape = [f, embeddingSize, 1, filterNum]
			filter_init=tf.truncated_normal_initializer(mean=0.0, stddev=0.5, seed=None, dtype=tf.float32)
			bias_init=tf.constant_initializer(value=0.2, dtype=tf.float32)
			convFilter = tf.get_variable(name='Convolution_Filter', shape=filterShape, initializer=filter_init)
			bias = tf.get_variable(name='convBias', shape=[filterNum], initializer=bias_init)	# allow filters having differnt bias
			# Convolution of a given filter size
			conv=convCNN(inputs,convFilter,bias)
			# max-pooling: each filter will just contribute its maximum activation
			maxPool.append(mpCNN(conv, k = seqLen-f+1))
	
	# concatenate all pooled results
	poolAll = tf.concat(concat_dim=3, values=maxPool)
	# flatten into 1-D
	poolAll = tf.reshape(poolAll, [-1, filterNum*len(filterList)]) # use tf.reshape(poolALl,[-1]) ? 
	# dropout regularization
	poolDropped = tf.nn.dropout(poolAll,keepRate)
	# output
	filterNumAll=filterNum*len(filterList)
	W_out = tf.Variable(tf.truncated_normal(shape=[filterNumAll, classSize], stddev=0.2))
	b_out = tf.Variable(tf.constant(0.1, shape=[classSize]))
	#
	pred = tf.matmul(poolDropped, W_out) + b_out
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
	correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
	train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
	print("Graph is built.")
	return dict(
		x=x,
		y=y,
		keepRate=keepRate,
		cost=cost,
		accuracy=accuracy,
		train_step=train,
		saver = tf.train.Saver()
	)
#===========================================================================================================================================
def trainCNN(graphCNN, data, num_epoch=3, batch_size=5, keepRate=0.5, saveAt=False):
	with tf.Session() as sess:
		sess.run(tf.initialize_all_variables())
		losses=[]
		totalSteps=int(num_epoch*len(data)/batch_size)
		for i, epoch in enumerate(fetch_batch(rawData=data, batchSize=batch_size, epochs=num_epoch)):
			temp_loss=0
			steps=0
			state_out=None
			X, Y = zip(*epoch)
			#steps +=1
			#Xt=np.array(X).transpose()
			feed_dict={graphCNN['x']:X, graphCNN['y']:Y, graphCNN['keepRate']:0.5}
			cost_out, accuracy_out, _ = sess.run([graphCNN['cost'],graphCNN['accuracy'], graphCNN['train_step']],feed_dict)
			#temp_loss+= cost_out
			if(i%2==0):
				print "Training loss of step/totalSteps: " + str(i) + "/"+ str(totalSteps)+ " : " + str(cost_out)
		if isinstance(saveAt, str):
			if not os.path.exists('./saves/'+saveAt.split('/')[2]):
				os.makedirs('./saves/'+saveAt.split('/')[2])
			graphCNN['saver'].save(sess, saveAt) 
	return losses
#===========================================================================================================================================
def testCNN(graphCNN,data,checkpoint="./saves/myTestModel"):
	with tf.Session() as sess:
		sess.run(tf.initialize_all_variables())
		graphCNN['saver'].restore(sess, checkpoint)
		correct_counts=0
		for i, epoch in enumerate(fetch_batch(rawData=data, batchSize=50, epochs=1)):
			X, Y = zip(*epoch)		
			feed_dict={graphCNN['x']:X, graphCNN['y']:Y, graphCNN['keepRate']:1}
			cost_out, accuracy_out= sess.run([graphCNN['cost'],graphCNN['accuracy']],feed_dict)	
			correct_counts += accuracy_out * len(epoch)	
			print("Testing accuracy of batch {} is: {}".format(i, accuracy_out))
	print("Average accuracy is: {}".format(correct_counts/len(data)))
	return correct_counts
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
