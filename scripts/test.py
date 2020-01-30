import time
import os
import torch.backends.cudnn as cudnn
import torch
from model import Transformer
from utils import *

global device

## Init seeds and GPU
seed = 1234
torch.manual_seed(seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
multigpu = False
if device == 'cuda':
	multigpu = torch.cuda.device_count() > 1
device = 'cpu'  ### Override
#cudnn.benchmark = True
#cudnn.enabled = True
print(device)

## Training & vocab parameters
DATA_PATH = 'data'
VOCAB_SIZE = 10000
BATCH_SIZE = 84
EPOCHS = 50
BEST_VAL = 9999.9
BEST_MODEL = None

## Model parameters
EMBED_SIZE = VOCAB_SIZE+2
EMBED_DIM = 128
TRANS_HEAD = 8
TRANS_FWD = 2048
TRANS_DROP = 0.1
TRANS_ACTIV = 'relu'
TRANS_LAYER = 1
LSTM_SIZE = 128
LSTM_LAYER = 2
LSTM_BIDIR = True
FC_DROP = 0.3
FC_OUT = 2

## Testing parameter
MODEL_DIR = 'model/'
MODEL_NAME = MODEL_DIR+'model_ep_17.tar'
TRAIN_BY_MULGPU = True

if __name__ == '__main__':

	train_data, valid_data, test_data, codes, label = import_data(DATA_PATH)

	train_iterator, valid_iterator, test_iterator = vocab_prepare(
		codes,
		label,
		train_data,
		valid_data, 
		test_data, 
		VOCAB_SIZE, 
		BATCH_SIZE)

	model = Transformer(EMBED_SIZE, EMBED_DIM, TRANS_HEAD, TRANS_FWD, TRANS_DROP, TRANS_ACTIV, TRANS_LAYER, LSTM_SIZE, LSTM_LAYER, LSTM_BIDIR, FC_DROP, FC_OUT, multigpu)
	if multigpu:
		model = torch.nn.DataParallel(model)
	model.to(device)
	print(model)

	checkpoint = torch.load(MODEL_NAME, map_location='cpu')

	if TRAIN_BY_MULGPU:
		# create new OrderedDict that does not contain `module.`
		from collections import OrderedDict
		new_state_dict = OrderedDict()
		for k, v in checkpoint['model_state_dict'].items():
		    name = k[7:] # remove `module.`
		    new_state_dict[name] = v
		model.load_state_dict(new_state_dict)
	else:
		model.load_state_dict(checkpoint['model_state_dict'])

	optimizer, criterion = loss_functions(model)
	criterion = criterion.to(device)


	## Testing

	model.eval()
	with torch.no_grad():
	    running_acc_test = 0
	    running_loss_test = 0
	    all_pred=[]
	    all_labels=[]
	    for batch in test_iterator:
	        batch.codes, batch.label = batch.codes.to(device), batch.label.to(device)
	        output_test = model(batch.codes)
	        loss_test = criterion(output_test,batch.label)
	        acc_test = softmax_accuracy(output_test,batch.label)
	        running_acc_test += acc_test.item()
	        running_loss_test += loss_test.item()
	        all_pred += output_test.tolist()
	        all_labels += batch.label.tolist()


	print('Test acc: ',running_acc_test/len(test_iterator))
	print('Test loss: ',running_loss_test/len(test_iterator))
	evaluate_testing(all_pred, all_labels)
