from functools import partial
import sys,os
sys.path.insert(0, '../')

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from utils_score_torch import *

if __name__ == "__main__":

	DataSet = 'AVS1K-TE'

	if os.name == 'nt':
		RootDir = 'E:/DataSet/' + DataSet + '/'
	else:
		RootDir = '/home/kao/D/DataSet/' + DataSet + '/'

	ResDir = RootDir + 'Results/Results_avs1k_ft/'

	# keys_order = ['NSS', 'KLD', 'SIM', 'CC']
	keys_order = ['AUC_shuffled', 'NSS', 'AUC_Judd', 'AUC_Borji', 'KLD', 'SIM', 'CC']
	MethodNames = [
		'UAVSal',
	]

	IS_EVAL_SCORES=1
	if IS_EVAL_SCORES:
		evalscores_vid_torch(RootDir, ResDir, DataSet, MethodNames, keys_order, batch_size=64)

	IS_ALL_SCORES = 0
	if IS_ALL_SCORES:
		MaxVideoNums = float('inf')

		# for matlab implementation
		import matlab
		import matlab.engine
		eng = matlab.engine.start_matlab()
		eng.Vid_MeanScore(ResDir, nargout = 0)