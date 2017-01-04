import os 
import numpy
import jsTokenEncoder

def prepare_data(seqs, labels, maxlen=None):
	""" Create the matrices from the datasets 
	
	`pad eaqch sequence to the same length`: 
		the length of the longest sequence or maxlen. 
		if maxlen is set, we will cut all sequence to this maximum length 
	
	Argument:
	- seqs, list of sequence
	- labels, list of label
	- maxlen, (optional) 
	Return: 
	- x, (maxlen of longest sequence, number of samples)
	- x_mask, (maxlen of longest sequence, number of samples)
	- labels, (maxlen of longest sequence) 
	"""
	lengths = [len(s) for s in seqs]
	# print(lengths)
	if maxlen is not None:
		new_seqs    = []
		new_labels  = []
		new_lengths = []
		for l,s,y in zip(lengths, seqs, labels):
			if l < maxlen:
				new_seqs.append(s)
				new_labels.append(y)
				new_lengths.append(l)
		lengths = new_lengths
		seqs    = new_seqs
		labels  = new_labels

		if len(lengths) < 1:
			return None, None, None
	else:
		maxlen    = numpy.max(lengths)

	n_samples = len(seqs)
	
	
	x 	   = numpy.zeros((maxlen, n_samples)).astype('int64')
	x_mask = numpy.zeros((maxlen, n_samples)).astype('float32')
	labels = numpy.array(labels).astype('int32')
	for idx, s in enumerate(seqs): 
		x[     :lengths[idx], idx] = s
		x_mask[:lengths[idx], idx] = 1.

	return x, x_mask, labels

def load_data(pklPath="", n_tokens=107,  \
			  valid_portion=0.1, maxlen=None, sort_by_len=True, drop_some_good=True):
	''' Loads the Datasets
	'''
	# get dataset file
	if pklPath != "":
		train_set = jsTokenEncoder.loadSampleFromPickle(pklPath)
		test_set  = jsTokenEncoder.	loadSampleFromPickle(pklPath)
	else:
		train_set = jsTokenEncoder.loadSampleFromPickle()
		test_set  = jsTokenEncoder.	loadSampleFromPickle()
		
	if maxlen:
		new_train_set_x = []
		new_train_set_y = []
		for x,y in zip(train_set[0], train_set[1]):
			if len(x) < maxlen:
				new_train_set_x.append(x)
				new_train_set_y.append(y)
		train_set = (new_train_set_x, new_train_set_y)
		del new_train_set_x, new_train_set_y
	
	# split training set into valuidation set 
	train_set_x, train_set_y = train_set
	n_samples = len(train_set_x)

	sidx = numpy.random.permutation(n_samples)
	n_train = int(numpy.round(n_samples * (1. - valid_portion)))
	valid_set_x = [train_set_x[s] for s in sidx[n_train:]]
	valid_set_y = [train_set_y[s] for s in sidx[n_train:]]
	train_set_x = [train_set_x[s] for s in sidx[:n_train]]
	train_set_y = [train_set_y[s] for s in sidx[:n_train]]

	train_set = (train_set_x, train_set_y)
	valid_set = (valid_set_x, valid_set_y)

	train_set_x, train_set_y = train_set
	valid_set_x, valid_set_y = valid_set
	test_set_x, test_set_y = test_set
	
	# Already Make sure that token would not exceed
	'''
	def remove_unk(x):
		return [[1 if w >= n_tokens else w for w in sen] for sen in x]		
	train_set_x = remove_unk(train_set_x)
	valid_set_x = remove_unk(valid_set_x)
	test_set_x = remove_unk(test_set_x)
	'''

	# sort by seq lenght 
	def len_argsort(seq):
		return sorted(range(len(seq)), key=lambda x: len(seq[x]))
	
	if sort_by_len:
		sorted_index = len_argsort(train_set_x)
		train_set_x = [train_set_x[i] for i in sorted_index]
		train_set_y = [train_set_y[i] for i in sorted_index]
		
		sorted_index = len_argsort(valid_set_x)
		valid_set_x = [valid_set_x[i] for i in sorted_index]
		valid_set_y = [valid_set_y[i] for i in sorted_index]
		
		sorted_index = len_argsort(test_set_x)
		test_set_x = [test_set_x[i] for i in sorted_index]
		test_set_y = [test_set_y[i] for i in sorted_index]

	# remove some label
	def label_argsort(labels):
		return sorted(range(len(labels)), key=lambda x: labels[x])

	if drop_some_good:
		print("Lets remove some good")

		train_lengths = [len(s) for s in train_set_x]
		print("train before Avg : ", numpy.average(train_lengths))

		sorted_index = label_argsort(train_set_y)
		train_set_x = [train_set_x[i] for i in sorted_index]
		train_set_y = [train_set_y[i] for i in sorted_index]
		length_we_want = round(len(train_set_x) / 8)
		train_set_x = train_set_x[:length_we_want]
		train_set_y = train_set_y[:length_we_want]

		train_lengths = [len(s) for s in train_set_x]
		print("after before Avg : ", numpy.average(train_lengths))
		
		

		valid_lengths = [len(s) for s in valid_set_x]
		print("valid before Avg : ", numpy.average(valid_lengths))

		sorted_index = label_argsort(valid_set_y)
		valid_set_x = [valid_set_x[i] for i in sorted_index]
		valid_set_y = [valid_set_y[i] for i in sorted_index]
		length_we_want = round(len(valid_set_x) / 8)
		valid_set_x = valid_set_x[:length_we_want]
		valid_set_y = valid_set_y[:length_we_want]
		
		valid_lengths = [len(s) for s in valid_set_x]
		print("valid after Avg : ", numpy.average(valid_lengths))
		

		test_lengths = [len(s) for s in test_set_x]
		print("test before Avg : ", numpy.average(test_lengths))

		sorted_index = label_argsort(test_set_y)
		test_set_x = [test_set_x[i] for i in sorted_index]
		test_set_y = [test_set_y[i] for i in sorted_index]
		length_we_want = round(len(test_set_x) / 8)
		test_set_x = test_set_x[:length_we_want]
		test_set_y = test_set_y[:length_we_want]

		
		test_lengths = [len(s) for s in test_set_x]
		print("test after Avg : ", numpy.average(test_lengths))

	train = (train_set_x, train_set_y)
	valid = (valid_set_x, valid_set_y)
	test = (test_set_x, test_set_y)
	return train, valid, test


def load_data_light(pklPath="", n_tokens=107,  \
			  valid_portion=0.1, maxlen=None, sort_by_len=True, drop_some_good=True):
	''' Loads the Datasets
	'''
	# get dataset file
	if pklPath != "":
		train_set = jsTokenEncoder.loadSampleFromPickle(pklPath)
	else:
		train_set = jsTokenEncoder.loadSampleFromPickle()
		
	if maxlen:
		new_train_set_x = []
		new_train_set_y = []
		for x,y in zip(train_set[0], train_set[1]):
			if len(x) < maxlen:
				new_train_set_x.append(x)
				new_train_set_y.append(y)
		train_set = (new_train_set_x, new_train_set_y)
		del new_train_set_x, new_train_set_y
	
	# split training set into valuidation set 
	train_set_x, train_set_y = train_set
	n_samples = len(train_set_x)

	sidx = numpy.random.permutation(n_samples)
	n_train = int(numpy.round(n_samples * (1. - valid_portion)))
	valid_set_x = [train_set_x[s] for s in sidx[n_train:]]
	valid_set_y = [train_set_y[s] for s in sidx[n_train:]]
	train_set_x = [train_set_x[s] for s in sidx[:n_train]]
	train_set_y = [train_set_y[s] for s in sidx[:n_train]]

	train_set = (train_set_x, train_set_y)
	valid_set = (valid_set_x, valid_set_y)

	train_set_x, train_set_y = train_set
	valid_set_x, valid_set_y = valid_set
	

	# sort by seq lenght 
	def len_argsort(seq):
		return sorted(range(len(seq)), key=lambda x: len(seq[x]))
	
	if sort_by_len:
		sorted_index = len_argsort(train_set_x)
		train_set_x = [train_set_x[i] for i in sorted_index]
		train_set_y = [train_set_y[i] for i in sorted_index]
		
		sorted_index = len_argsort(valid_set_x)
		valid_set_x = [valid_set_x[i] for i in sorted_index]
		valid_set_y = [valid_set_y[i] for i in sorted_index]

	# remove some label
	def label_argsort(labels):
		return sorted(range(len(labels)), key=lambda x: labels[x])

	if drop_some_good:
		print("Lets remove some good")

		train_lengths = [len(s) for s in train_set_x]
		print("train before Avg : ", numpy.average(train_lengths))

		sorted_index = label_argsort(train_set_y)
		train_set_x = [train_set_x[i] for i in sorted_index]
		train_set_y = [train_set_y[i] for i in sorted_index]
		length_we_want = round(len(train_set_x) / 8)
		train_set_x = train_set_x[:length_we_want]
		train_set_y = train_set_y[:length_we_want]

		train_lengths = [len(s) for s in train_set_x]
		print("after before Avg : ", numpy.average(train_lengths))
		
		

		valid_lengths = [len(s) for s in valid_set_x]
		print("valid before Avg : ", numpy.average(valid_lengths))

		sorted_index = label_argsort(valid_set_y)
		valid_set_x = [valid_set_x[i] for i in sorted_index]
		valid_set_y = [valid_set_y[i] for i in sorted_index]
		length_we_want = round(len(valid_set_x) / 8)
		valid_set_x = valid_set_x[:length_we_want]
		valid_set_y = valid_set_y[:length_we_want]
		
		valid_lengths = [len(s) for s in valid_set_x]
		print("valid after Avg : ", numpy.average(valid_lengths))
		

	train = (train_set_x, train_set_y)
	valid = (valid_set_x, valid_set_y)
	return train, valid


if __name__ == "__main__":
	t, v, e = load_data()
	print(len(t),len(v),len(e))
	print(len(t[0]),len(t[1]))
	print(len(v[0]),len(v[1]))
	print(len(e[0]),len(e[1]))
	


