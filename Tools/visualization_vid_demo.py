import os, cv2
import numpy as np
import hdf5storage as h5io

from utils_vis import *



if __name__ == "__main__":

	DataSet = 'UAV2-TE'

	if os.name == 'nt':
		RootDir = 'E:/DataSet/' + DataSet + '/'
	else:
		RootDir = '/home/kao/D/DataSet/' + DataSet + '/'

	ResDir = RootDir + 'Results/Results_UAVSal/'
	MethodNames = [
		'UAVSal',
	]

	WITH_FIX = 0
	WITH_COLOT = 1
	visual_vid(RootDir, ResDir, DataSet, MethodNames, with_color=WITH_COLOT, with_fix=WITH_FIX)
