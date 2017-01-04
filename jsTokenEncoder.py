import pickle 
import os 

from sklearn import preprocessing
import pandas as pd

TOKEN_DIR_PATH  = r"jsSample\tokesnLT10k"

TOKEN_LIST      = r"TOKEN_LIST.pickle"
TOKEN_ENCODER   = r"TOKEN_ENCODER.pickle"
LABEL_FILE      = r"jsSample\atse.meta"
LABEL_FILE_ALL  = r"jsSample\atse.meta.all"
SAMPLE_DUMP = "SAMPLELT10k.pickle"

def getTokenList(folder_path=TOKEN_DIR_PATH):
	''' Only used when not know how many token
	'''
	tokenSet = set()
	for sample_file in os.listdir(folder_path):
		with open(folder_path+"\\"+sample_file, "r", encoding = 'utf8') as f:
			while True:
				c = f.read(1)
				if not c:
					break
				tokenSet.add(c)
	tokenList = list(tokenSet)
	print("Number of Token:", len(tokenList))
	print("tokenList:", tokenList)
	return tokenList
    	
def dumpTokenList(dump_path=TOKEN_LIST):
	''' Saved list of token to picke
	'''
	tokenList = getTokenList()
	with open(dump_path, "wb") as f:
		pickle.dump(tokenList, f)
	return tokenList

def loadTokenEncoder(token_list_file=TOKEN_LIST):
	''' Generate labelEncoder from list of token in picke 
	'''
	te = preprocessing.LabelEncoder()
	with open(token_list_file, "rb") as f:
		tokenList = pickle.load(f)
		te.fit(tokenList)
	return te

def loadFormalLabelDataframe(label_file=LABEL_FILE):
	''' File with header Row
	'''
	label_table = pd.read_csv(label_file, 
                    usecols  = [1, 5], 
                    dtype = { "label": str, "contentsha1": str})
	# print(label_table.describe())
	return label_table

def loadCSVLabelDataframe(label_file=LABEL_FILE_ALL):
	''' File without header Row
	This function try to solve following file
	'''
	label_table = pd.read_csv(label_file,
					usecols = [1, 4],
					delimiter = ",",
					names = ['contentsha1', 'label'],
					dtype = {"contentsha1": str, "label": str})
	# print(label_table.describe())
	return label_table
	
def getLabelBySha1(labelDataFrame, contentsha1):
	label_info = labelDataFrame[labelDataFrame["contentsha1"] == contentsha1]
	if (len(label_info)>0):
		label = label_info["label"].iloc[0]
		if label == "good":
			return 1
		elif label == "bad":
			return 0
		else:
			return -2
	return -1

def turnSampleToTuple(tokenDecoder, 
					  feature_dump=SAMPLE_DUMP,
					  sample_folder_path=TOKEN_DIR_PATH,
					  maxLen=None,
					  minLen=20):
	'''read from sample and turn to (x,y) which
		x = list of sample feature in form, [token number ... ]
		y = list of label in integer
	'''
	x = list()
	y = list()
	## Here we need to map sample to label 
	# -> Small Table (but seems more found)
	label_table = loadFormalLabelDataframe()
	# -> Long Table (use if needed)
	# label_table = loadCSVLabelDataframe()
	for sample_file in os.listdir(sample_folder_path):
		lookUpLabel = getLabelBySha1(label_table, sample_file)
		if lookUpLabel >= 0:
			featureList = list()
			with open(sample_folder_path+"\\"+sample_file, "r", encoding = 'utf8') as f:
				for lines in f:
					featureList += list(tokenDecoder.transform(list(lines)))
			if len(featureList) > minLen: # ignore sample length less then 20
				if maxLen is not None:
					if len(featureList)<=maxLen:
						x.append(featureList)
						y.append(lookUpLabel)
				else:
					x.append(featureList)
					y.append(lookUpLabel)
		# print(sample_file, " ", str(getLabelBySha1(label_table, sample_file)))
	with open(feature_dump, "wb") as f:
		pickle.dump((x,y), f)
	return (x,y)

def loadSampleFromPickle(sample_path=SAMPLE_DUMP):
	with open(sample_path, "rb") as f:
		dataset = pickle.load(f)
	# print("Number of x : ", len(dataset[0]))
	# print("Number of y : ", len(dataset[1]))
	return dataset

if __name__ == "__main__":
	## Step0, Get set of Token and save to pickle
	# getTokenList(TOKEN_DIR_PATH)

	## Step0-1, Get Token List from Pickle
	# dumpTokenList(TOKEN_LIST)
	
	## Step1, Build Encoder by token list
	TE = loadTokenEncoder(TOKEN_LIST)
	num_of_token = len(TE.classes_)
	token_list   = list(TE.classes_)
	print(num_of_token)

	## Step2, (opt)build label mapping table (contentsha1 -> label)
	# loadFormalLabelDataframe()
	# loadCSVLabelDataframe()

	## Step3, read from character token seq to number token seq
	turnSampleToTuple(tokenDecoder=TE, feature_dump="SAMPLE20to100k.pickle", maxLen=100000)
	
	print("done save")
	## Step4, load data from turnSampleToTuple() 
	# data = loadSampleFromPickle()
	# print("done load")