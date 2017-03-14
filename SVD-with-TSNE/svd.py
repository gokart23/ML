import numpy as np

def preprocessCorpus(fname):
	with open("text8", "r") as corpus:
		print "Reading file..."
		contents = corpus.readlines()
		print "File read"
		uniqueWords, counts = np.unique(contents[0].split(" "), return_counts = True)
		print "Obtained counts...sorting data"
		idxes = np.lexsort((uniqueWords, counts))[::-1] # [::-1] provides a reverse view to the array
		wordDict = dict( zip( uniqueWords[ idxes ], list( uniqueWords[ idxes ] [:10000] ) + (len(uniqueWords) - 10000) * ["UNK"] ) )
	with open(fname, "a+") as proc:
		with open("text8", "r") as corpus:
			print "Writing out to file"
			proc.write( " ".join( map( lambda x : wordDict[x], corpus.readlines()[0].strip().split(" ") ) ) )
	print "loadData done"		

def getCooccurrenceMatrix(fname, context=5):
	with open(fname, "r") as procCorpus:
		text = procCorpus.readlines()[0].strip().split(" ")
		words = np.unique( text )
	print "Loaded words...creating mapping"
	wordMapping = dict( zip(words, range(len(words)) ) )
	print "Mapping created...creating co-occurrence matrix"
	wordCooccurrence = np.zeros((len(wordMapping), len(wordMapping)), dtype=int)
	for x in range(len(text)):
		for y in range(max(0, x-context), x) + range(x+1, min(len(text), x+context+1)):
			wordCooccurrence[ wordMapping[ text[x] ], wordMapping[ text[y] ] ] += 1
	print "Matrix created..dumping data"
	np.savez("word_co_occurrence_matrix", wordCooccurrence=wordCooccurrence)
	np.savez("word_mapping", wordMapping=wordMapping)
	# x = np.load("word_mapping.npz")['wordMapping'].item() for accessing NPZ file
	
	print "getCooccurrence done"
	return wordMapping, wordCooccurrence

def svd(data, flag=False):
	print "Extracting U"
	x, U = np.linalg.eig( np.dot(data, data.T))
	if flag == True:
		print "Extracting V"
		w, V = np.linalg.eig( np.dot(data.T, data))
		print "Computing S"
		S = np.diag( np.sqrt(x) )
	else:
		S, V = [], []
	print "SVD done"
	return U, S, V

if __name__ == '__main__':
	print "Main started"

	#1. Pre-process the corpus and create new corpus with 10k words
	print "Starting preprocessing"
	preprocessCorpus("processed_text")

	#2. Obtain co-occurrence matrix
	print "Starting matrix generation"
	wM, wC = getCooccurrenceMatrix("processed_text")
	print "Len wM",len(wM)

	#2.1 If co-occurrence matrix is already available, load from file(s)
	# print "Loading data"
	# wM = np.load("word_mapping.npz")['wordMapping'].item()
	# wC = np.load("word_co_occurrence_matrix.npz")['wordCooccurrence']

	#3. Perform SVD, and extract 50 dimensional embeddings
	print "Performing SVD"
	U, _, _ = svd(wC)
	np.savez("u_complete", U=U)
	print "Obtaining word embeddings and saving data"
	wordEmbeddings = U[:, :50]
	np.savez("word_embeddings", wordEmbeddings=zip(list(wordEmbeddings), wM) )

	#3.1 For using the saved word embeddings,
	# print "Loading word embeddings"
	# wE = np.load("word_embeddings.npz")['wordEmbeddings']
	# For extracting the ith word's name and corresponding vector,
	# print "Word:",wE[i][1]
	# print "Vector:",wE[i][0]

	print "Main complete"

