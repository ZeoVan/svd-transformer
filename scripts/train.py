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
#os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
cudnn.benchmark = True
cudnn.enabled = True
print(device)

# Batch Size
# 84 - 8 GPU (11GB Each)

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

	print(f'The model has {count_parameters(model):,} trainable parameters')

	optimizer, criterion = loss_functions(model)
	criterion = criterion.to(device)

	print('Training started.....')
	## Training 
	for e in range(EPOCHS):
		running_acc = 0
		running_loss = 0
		timer = time.time()
		model.train()

		for batch in train_iterator:
			batch.codes, batch.label = batch.codes.to(device), batch.label.to(device)
			optimizer.zero_grad()
			output = model(batch.codes)
			loss = criterion(output, batch.label)
			loss.backward()
			optimizer.step()
			acc = softmax_accuracy(output,batch.label)
			running_acc += acc.item()
			running_loss += loss.item()
		else:
			with torch.no_grad():
				model.eval()
				running_acc_val = 0
				running_loss_val = 0
				for batch in valid_iterator:
					batch.codes, batch.label = batch.codes.to(device), batch.label.to(device)
					output_val = model(batch.codes)
					loss_val = criterion(output_val,batch.label)
					acc_val = softmax_accuracy(output_val,batch.label)
					running_acc_val += acc_val.item()
					running_loss_val += loss_val.item()

			print_out = "Epoch {} - Training acc: {:.6f} -Training loss: {:.6f} - Val acc: {:.6f} - Val loss: {:.6f} - Time: {:.4f}s \n".format(e+1,
			running_acc/len(train_iterator),
			running_loss/len(train_iterator),
			running_acc_val/len(valid_iterator),
			running_loss_val/len(valid_iterator),
			(time.time()-timer))

			myfile = open("results.txt", "a")

			if (running_loss_val/len(valid_iterator)) < BEST_VAL:
				print('Val_loss decreased!')
				print(print_out, end='')
				myfile.write('Val_loss decreased!')
				myfile.write(print_out)

				BEST_VAL = (running_loss_val/len(valid_iterator))
				BEST_MODEL = model
				model_name = 'model/model_ep_%d.tar' % (e+1)
				torch.save({
					'epoch': e+1,
					'model_state_dict': model.state_dict(),
					'optimizer_state_dict': optimizer.state_dict(),
					'loss': loss}, model_name)

			else:
				print(print_out, end='')
				myfile.write(print_out)

			myfile.close()

	print('Training completed!')
	print('Testing started.......')
	## Testing
	checkpoint = torch.load('model_name', map_location='cpu')
	model.load_state_dict(checkpoint['model_state_dict'])
	optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

	model.eval()
	with torch.no_grad():
	    running_acc_test = 0
	    running_loss_test = 0
	    all_pred=[]
	    all_labels=[]
	    for batch in test_iterator:
	        batch.codes, batch.label = batch.codes.to(device), batch.label.to(device)
	        output_test = model(batch.codes).squeeze(1)
	        loss_test = criterion(output_test,batch.label)
	        acc_test = softmax_accuracy(output_test,batch.label)
	        running_acc_test += acc_test.item()
	        running_loss_test += loss_test.item()
	        all_pred += output_test.tolist()
	        all_labels += batch.label.tolist()


	print('Test acc: ',running_acc_test/len(test_iterator))
	print('Test loss: ',running_loss_test/len(test_iterator))
	evaluate_testing(all_pred, all_labels)
