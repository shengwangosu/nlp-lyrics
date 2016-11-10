import MySQLdb, re
import numpy as np
from nltk.tokenize import RegexpTokenizer
import itertools
"""
+--------+-----------------+------+-----+---------+----------------+
| Field  | Type            | Null | Key | Default | Extra          |
+--------+-----------------+------+-----+---------+----------------+
| id     | int(6) unsigned | NO   | PRI | NULL    | auto_increment |
| artist | varchar(99)     | YES  |     | NULL    |                |
| title  | varchar(99)     | YES  |     | NULL    |                |
| year   | smallint(6)     | YES  |     | NULL    |                |
| lyric  | text            | YES  |     | NULL    |                |
| url    | varchar(300)    | YES  |     | NULL    |                |
| genre  | varchar(30)     | YES  |     | NULL    |                |
+--------+-----------------+------+-----+---------+----------------+
"""
def showGenreStat():
	dataBaseName ='lyrics'
	tableName='lyricsBody_full' 
	db = MySQLdb.connect(host='localhost', user='root',passwd='2099', db=dataBaseName)
	cursor = db.cursor()
	cursor.execute("SELECT DISTINCT genre FROM " + tableName)
	genreSet=cursor.fetchall()
	for g in genreSet:
		#print g[0]
		cursor.execute("""SELECT COUNT(*) FROM lyricsBody_full WHERE genre = %s""", (g[0],))
		count=cursor.fetchall()
		print('{:<20}' '{:<4}'.format(g[0],count[0][0]))
#===========================================================================================================================
def cleanText(s, tokenizer=RegexpTokenizer('\w+')):
	#s=re.sub("\W"," ",s) 	# remove not word: .,;~\
	return tokenizer.tokenize(s)
#===========================================================================================================================
def get_Lyric_Genre(dataBaseName ='lyrics', tableName='lyrics_a_to_z',
						genreSet=['Rock','Hip Hop/Rap','Pop', 'R&B', 'Country'],pickNum=[3000,3000,3000,2000,2000]):
	"""read MySQL and get lyrics and their genre (one-hot encoding)"""
	db = MySQLdb.connect(host='localhost', user='root',passwd='2099', db=dataBaseName)
	cursor = db.cursor()
	#pickGenres = """ WHERE genre = '{}' OR genre ='{}' OR genre='{}' OR genre='{}'""".format(*genreSet)
	#cursor.execute("SELECT lyric, genre FROM " + tableName + pickGenres)	
	#q= cursor.fetchall()
	q=()
	picked=[]
	for genreName, genreNum in zip(genreSet, pickNum):
		print("Limit length of lyric to >100")
		sqlQuote= """ SELECT lyric, genre FROM {} WHERE genre='{}' and length(lyric)>100 LIMIT {}""".format(tableName, genreName, genreNum)
		cursor.execute(sqlQuote)
		qTemp=cursor.fetchall()
		picked.append(len(qTemp))
		q = q+ qTemp
	#q=tuple(q)
	print('====databse: Lyrics and genres Statistics======')
	print('==== # of lyrics: {}'.format(len(q)))
	print('===  {:<15}' '{:<6}' '{:<8}'.format('genre name','total','selected'))
	#
	for t, g in enumerate(genreSet):
		sqlQuery=""" SELECT COUNT(*) FROM {} WHERE genre='{}' """.format(tableName, g)  
		cursor.execute(sqlQuery)
		count=cursor.fetchall()
		#print g, count[0][0]
		print('===  {:<15}' '{:<6}' '{:<8}'.format(g,count[0][0],picked[t]))
	Dict={genre:id for id, genre in enumerate(genreSet)}
	x=[cleanText(i[0].lower()) for i in q]				# element is the lyric
	id2oneHot=np.identity(len(genreSet),dtype=int)
	y=[]	# one hot encoding of the genre
	for i in q:
		id=Dict[i[1]]	# get genre of the lyric
		y.append(id2oneHot[id])

	return x, y
#===========================================================================================================================
def textCutPad(stringList, length):
	# cut or pad each string to the desired length, pad using 'NULL'
	# stringList: a list of lists of words
	# output: a list of lists of words, with fixed length
	newList=[]
	for slist in stringList:
		if(len(slist))>+length:
			newList.append(slist[0:length])
		else:
			newList.append(slist+['NULL']*(length-len(slist)))
	return newList
#===========================================================================================================================================
def build_corpus(lyrics):
	# map word to id, where the padding 'NULL' is 0
	# input: list of list of words
	# output: list of lists of id
	combined = list(itertools.chain.from_iterable(lyrics)) ## concatenate all words in lyrics
	dictionary=(set(combined))
	dictionary.difference_update(set(['NULL']))	# remove ['NULL'] from dictionary
	id2word = {i+1:w for i,w in enumerate(dictionary)}
	word2id = {w:i+1 for i,w in enumerate(dictionary)}
	id2word[0]='NULL'							# add ['NULL'] back to dictionary with id 0
	word2id['NULL']=0
	corpus=[[word2id[word] for word in c] for c in lyrics]			# build corpus using the char2id embedding
	return corpus, word2id, id2word
"""
def fetch_batch(rawData, batchSize, epochs):
	#generate mini batch
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

x,y=get_Lyric_Genre()
batch_iterator = fetch_batch(zip(x,y),batchSize=10, epochs=3)
"""



