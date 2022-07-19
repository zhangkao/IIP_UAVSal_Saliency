import torch, os, cv2, sys, math, shutil, copy, time

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from model import *
from utils_data import *
from utils_score_torch import *
from utils_vis import *
from utils_data import normalize_data as norm_data

import numpy as np

def get_bias(bias_type=[1, 1, 1], batch_size=2, shape_r=45, shape_c=80):
    if bias_type[0]:
        x_cb_gauss = get_guasspriors(batch_size, shape_r, shape_c, channels=8).transpose((0, 3, 1, 2))
        x_cb_gauss = torch.tensor(x_cb_gauss).float()
    else:
        x_cb_gauss = torch.tensor([]).float()

    if bias_type[1]:
        x_cb_ob = get_ob_priors('', DataSet_Train, 'train', batch_size, shape_r, shape_c).transpose((0, 3, 1, 2))
        x_cb_ob = torch.tensor(x_cb_ob).float()
    else:
        x_cb_ob = torch.tensor([]).float()

    return [x_cb_gauss.to(device), x_cb_ob.to(device)]


def test(input_path, output_path, model_path, method_name='UAVSal', saveFrames=float('inf'), time_dims=5, iosize=[480, 640, 60, 80],
         batch_size=4, bias_type=[1, 1, 1]):

    model = UAVSal(cnn_type='mobilenet_v2', time_dims=time_dims, num_stblock=2, bias_type=bias_type,
                             iosize=iosize, planes=256, pre_model_path='')
    model = model.to(device)

    if os.path.exists(model_path):
        print("Load UAVSal weights")
        model.load_state_dict(torch.load(model_path).state_dict())
    else:
        raise ValueError

    # if os.path.exists(model_path):
    #     print("Load UAVSal weights")
    #     model = torch.load(model_path)
    #     model = model.to(device)
    # else:
    #     raise ValueError

    output_path = output_path + method_name + '/'
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    shape_r, shape_c, shape_r_out, shape_c_out = iosize
    use_cb = np.sum(np.array(bias_type) > 0)
    x_cb = get_bias(bias_type, batch_size * time_dims, shape_r_out, shape_c_out)

    file_names = [f for f in os.listdir(input_path) if (f.endswith('.avi') or f.endswith('.AVI') or f.endswith('.mp4'))]
    file_names.sort()
    nb_videos_test = len(file_names)

    model.eval()
    with torch.no_grad():
        for idx_video in range(nb_videos_test):
            print("%d/%d   " % (idx_video + 1, nb_videos_test) + file_names[idx_video])

            ovideo_path = output_path + (file_names[idx_video])[:-4] + '.mat'
            if os.path.exists(ovideo_path):
                continue

            ivideo_path = input_path + file_names[idx_video]
            vidimgs, nframes, height, width = preprocess_videos(ivideo_path, shape_r, shape_c, saveFrames, mode='RGB',
                                                                normalize=False)

            count_bs = nframes // time_dims
            isaveframes = count_bs * time_dims
            vidimgs = vidimgs[0:isaveframes].transpose((0, 3, 1, 2))

            pred_mat = np.zeros((isaveframes, height, width, 1), dtype=np.uint8)
            count_input = batch_size * time_dims
            bs_steps = math.ceil(count_bs / batch_size)
            x_state = None
            for idx_bs in range(bs_steps):
                x_imgs = vidimgs[idx_bs * count_input:(idx_bs + 1) * count_input]
                x_imgs = torch.tensor(norm_data(x_imgs)).float()

                if use_cb and x_imgs.shape[0] != count_input:
                    x_cb_input = get_bias(bias_type, x_imgs.shape[0], shape_r_out, shape_c_out)
                else:
                    x_cb_input = x_cb

                bs_out, out_state = model(x_imgs.to(device), x_cb_input, x_state)
                x_state = [out_state[0].detach()]
                bs_out = bs_out.data.cpu().numpy()

                for idx_pre in range(bs_out.shape[0]):
                    isalmap = postprocess_predictions(bs_out[idx_pre, 0, :, :], height, width)
                    pred_mat[idx_bs * count_input + idx_pre, :, :, 0] = np2mat(isalmap)

            iSaveFrame = min(isaveframes, saveFrames)
            pred_mat = pred_mat[0:iSaveFrame, :, :, :].transpose((1, 2, 3, 0))
            h5io.savemat(ovideo_path, {'salmap': pred_mat})



if __name__ == '__main__':

    DataSet_Test = 'UAV2-TE'
    model_path = './weights/uavsal-mobilenet_v2-uav2-v2-288-512.pth'
    ext = '.avi'

    # DataSet_Test = 'AVS1K-TE'
    # model_path = './weights/uavsal-mobilenet_v2-avs1k-v2.pth'
    # ext = '.mp4'

    method_name = 'UAVSal'
    batch_size = 4
    iosize = [360, 640, 45, 80]

    if os.name == 'nt':
        dataDir = 'D:/DataSet/'
    else:
        dataDir = '/home/kao/D/DataSet/'
    test_dataDir = dataDir + '/' + DataSet_Test + '/'
    test_input_path = test_dataDir + 'Videos/'
    test_result_path = test_dataDir + 'Results/'
    test_output_path = test_result_path + 'Saliency/'

    DataSet_Train = DataSet_Test[:-3]

    test(test_input_path, test_output_path, model_path, method_name=method_name, saveFrames=float('inf'),
         iosize=iosize, batch_size=batch_size, time_dims=5, bias_type=[1, 1, 1])

    evalscores_vid_torch(test_dataDir, test_result_path, DataSet=DataSet_Test, MethodNames=[method_name], batch_size=32)

    visual_vid(test_dataDir, test_result_path, DataSet=DataSet_Test, MethodNames=[method_name], with_color=1,
               with_fix=0)
