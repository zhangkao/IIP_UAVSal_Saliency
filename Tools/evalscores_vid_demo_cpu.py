from functools import partial
import sys
sys.path.insert(0, '../')

from utils_score import *

if __name__ == "__main__":

	DataSet = 'UAV2-TE'

	if os.name == 'nt':
		RootDir = 'E:/DataSet/' + DataSet + '/'
	else:
		RootDir = '/home/kao/D/DataSet/' + DataSet + '/'

	ResDir = RootDir + 'Results/Results_UAVSal/'

	keys_order = ['AUC_shuffled', 'NSS', 'AUC_Judd', 'AUC_Borji', 'KLD', 'SIM', 'CC']
	MethodNames = [
		'UAVSal',
	]

	IS_EVAL_SCORES=1
	if IS_EVAL_SCORES:
		evalscores_vid(RootDir, ResDir, DataSet, MethodNames, keys_order)

	IS_ALL_SCORES = 0
	if IS_ALL_SCORES:
		MaxVideoNums = float('inf')

		# for matlab implementation
		import matlab
		import matlab.engine
		eng = matlab.engine.start_matlab()
		eng.Vid_MeanScore(ResDir, nargout = 0)