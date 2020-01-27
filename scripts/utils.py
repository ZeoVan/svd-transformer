import numpy as np
import pandas as pd
import random
import re
import torch
import torchtext
import torchtext.vocab as vocab
import sklearn.metrics
from sklearn.metrics import confusion_matrix
from gensim.models.word2vec import Word2Vec
from torch.autograd import Variable
from torch import nn, optim
from torch.optim import SGD,Adam


def import_data(path_to_data):
	## Define the field
	CODES = torchtext.data.Field(batch_first=True)
	LABEL = torchtext.data.LabelField(dtype=torch.long)
	fields = {'codes': ('codes', CODES), 'label': ('label', LABEL)}

	## Import the 100K data as TabularDataset
	train_data, valid_data, test_data = torchtext.data.TabularDataset.splits(
	                                        path = path_to_data,
	                                        train = 'train.json',
	                                        validation = 'valid.json',
	                                        test = 'test.json',
	                                        format = 'json',
	                                        fields = fields)
	return(train_data,valid_data,test_data, CODES, LABEL)

def vocab_prepare(CODES, LABEL, train_data, valid_data, test_data, VOCAB_SIZE, BATCH_SIZE):
	##### Build the vocabulary
	MAX_VOCAB_SIZE = VOCAB_SIZE

	CODES.build_vocab(train_data, max_size = MAX_VOCAB_SIZE)
	LABEL.build_vocab(train_data)

	## place into iterators
	train_iterator, valid_iterator, test_iterator = torchtext.data.BucketIterator.splits(
	    (train_data, valid_data, test_data), 
	    batch_size = BATCH_SIZE,
	    sort = False)
	return(train_iterator, valid_iterator, test_iterator)

def load_embeddings(path,model):
	w2v = Word2Vec.load(path)
	weights = torch.FloatTensor(w2v.wv.vectors)
	weights = weights.to(device)
	model.embed = model.embed.from_pretrained(weights)
	del(weights)
	return(model)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def softmax_accuracy(probs,all_labels):
    def getClass(x):
        return(x.index(max(x)))
    
    all_labels = all_labels.tolist()
    probs = pd.Series(probs.tolist())
    all_predicted = probs.apply(getClass)
    all_predicted.reset_index(drop=True, inplace=True)
    vc = pd.value_counts(all_predicted == all_labels)
    try:
        acc = vc[1]/len(all_labels)
    except:
        if(vc.index[0]==False):
            acc = 0
        else:
            acc = 1
    return(acc)

def loss_functions(model):
	## Define optimizer
	#optimizer = SGD(model.parameters(), lr = 0.01)
	optimizer = Adam(model.parameters(), lr=0.0001, weight_decay=0.01)

	## Define loss function
	#criterion = nn.BCELoss().to(device) ## Sigmoid activation function
	#criterion = nn.NLLLoss().to(device) ### Log_softmax activation
	criterion = nn.CrossEntropyLoss() ## No activation function in model bcs softmax included
	return(optimizer, criterion)

def evaluate_testing(all_pred, all_labels):
	def getClass(x):
	    return(x.index(max(x)))

	probs = pd.Series(all_pred)
	all_predicted = probs.apply(getClass)
	all_predicted.reset_index(drop=True, inplace=True)
	vc = pd.value_counts(all_predicted == all_labels)

	probs2=[]
	for x in probs:
	    probs2.append(x[1])

	confusion = sklearn.metrics.confusion_matrix(y_true=all_labels, y_pred=all_predicted)
	print('Confusion matrix: \n',confusion)


	tn, fp, fn, tp = confusion.ravel()
	print('\nTP:',tp)
	print('FP:',fp)
	print('TN:',tn)
	print('FN:',fn)

	## Performance measure
	print('\nAccuracy: '+ str(sklearn.metrics.accuracy_score(y_true=all_labels, y_pred=all_predicted)))
	print('Precision: '+ str(sklearn.metrics.precision_score(y_true=all_labels, y_pred=all_predicted)))
	print('Recall: '+ str(sklearn.metrics.recall_score(y_true=all_labels, y_pred=all_predicted)))
	print('F-measure: '+ str(sklearn.metrics.f1_score(y_true=all_labels, y_pred=all_predicted)))
	print('Precision-Recall AUC: '+ str(sklearn.metrics.average_precision_score(y_true=all_labels, y_score=probs2)))
	print('AUC: '+ str(sklearn.metrics.roc_auc_score(y_true=all_labels, y_score=probs2)))
	print('MCC: '+ str(sklearn.metrics.matthews_corrcoef(y_true=all_labels, y_pred=all_predicted)))

