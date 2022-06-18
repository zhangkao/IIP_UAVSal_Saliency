from functools import partial
import sys
sys.path.insert(0, '../')


from utils_score import *

if __name__ == "__main__":

	DataSet = 'salicon15'
	if os.name == 'nt':
		RootDir = 'E:/DataSet/' + DataSet + '/val/'
	else:
		RootDir = '/home/kao/DataSet/' + DataSet + '/val/'

	ResDir = RootDir + 'Results/Results_salicon15/'
	keys_order = ['AUC_shuffled', 'NSS', 'AUC_Judd', 'AUC_Borji', 'KLD', 'SIM', 'CC']
	MethodNames = [
		'UAVSal-SRF',
	]

	IS_EVAL_SCORES=1
	if IS_EVAL_SCORES:
		evalscores_img(RootDir, ResDir, DataSet, MethodNames, keys_order)

	IS_ALL_SCORES = 0
	if IS_ALL_SCORES:

		# for matlab implementation
		import matlab
		import matlab.engine
		eng = matlab.engine.start_matlab()
		eng.Img_MeanScore(ResDir, nargout = 0)
		