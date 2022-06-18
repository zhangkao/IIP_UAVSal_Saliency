import math,random,os,cv2,torch
import numpy as np
import hdf5storage as h5io


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#########################################################################
# Data and model Path														#
#########################################################################
TrainDataSet = 'UAV2'
DataSet = 'UAV2-TE'

# replace the datadir to your path
if os.name == 'nt':
    dataDir = 'E:/DataSet/'
    saveModelDir = 'E:/Models/models_'+ TrainDataSet.lower() +'/'
    preModelDir = 'E:/Models/pre_models/'
else:
    dataDir = '/home/kao/D/DataSet/'
    saveModelDir = '/home/kao/D/Models/models_'+ TrainDataSet.lower() +'/pytorch_models_uav/'
    preModelDir = '/home/kao/D/Models/models_'+ TrainDataSet.lower() +'/pre_models/'


train_dataDir = dataDir + '/' + TrainDataSet + '/'
test_dataDir = dataDir + '/' + DataSet + '/'
test_input_path = test_dataDir + 'Videos/'
test_result_path = test_dataDir + 'Results/Results_'+ TrainDataSet.lower() +'/'
test_output_path = test_result_path + 'Saliency/'

pre_sf_path = preModelDir + 'zk-st_final.pkl'
pre_dy_path = preModelDir + 'zk-dy_final.pkl'
pre_model_path = {
    'pre_sf_path': pre_sf_path,
    'pre_dy_path': pre_dy_path,
}

imgs_data_path = dataDir + '/salicon-15'
test_img_dataDir = dataDir + '/salicon-15/val/'
test_img_input_path = test_img_dataDir + 'images/'
test_img_result_path = test_img_dataDir + 'Results/Results_st_15/'
test_img_output_path = test_img_result_path + 'Saliency/'

#########################################################################
# Training Settings											            #
#########################################################################
IS_EARLY_STOP = True
IS_BEST_ONLY = False
Shuffle_Train = True
Max_patience = 4
Num_workers = 32
ext = '.mp4'
Max_TrainFrame = float('inf')
Max_ValFrame = float('inf')
saveFrames = float('inf')


if TrainDataSet in ['UAV2']:
    ext = '.avi'

