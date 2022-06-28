import torch, os, cv2, sys, math, shutil, copy, time

import torchvision
import torch.nn as nn
import torch.optim as optim

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from model import *
from dataset import *
from utils_data import *
from loss_functions import *
from utils_score_torch import *
from utils_vis import *
from utils_data import normalize_data as norm_data


def get_bias(bias_type=[1, 1, 1], batch_size=2, shape_r=45, shape_c=80):
    if bias_type[0]:
        x_cb_gauss = get_guasspriors(batch_size, shape_r, shape_c, channels=8).transpose((0, 3, 1, 2))
        x_cb_gauss = torch.tensor(x_cb_gauss).float()
    else:
        x_cb_gauss = torch.tensor([]).float()

    if bias_type[1]:
        x_cb_ob = get_ob_priors(train_dataDir, DataSet_Train, 'train', batch_size, shape_r, shape_c).transpose((0, 3, 1, 2))
        x_cb_ob = torch.tensor(x_cb_ob).float()
    else:
        x_cb_ob = torch.tensor([]).float()

    return [x_cb_gauss.to(device), x_cb_ob.to(device)]


def train(method_name='uavsal',
          cnn_type='mobilenet_v2',
          iosize=[480, 640, 60, 80],
          time_dims=5,
          num_stblock=2,
          bias_type=[1, 1, 1],
          batch_size=4,
          epochs=20,
          pre_model_path=''):

    tmdir = saveModelDir + method_name
    save_model_path = tmdir + '/' + method_name + '_'
    if not os.path.exists(tmdir):
        os.makedirs(tmdir)

    #################################################################
    # Build the model
    #################################################################
    print("Build UAVSAL Model: " + method_name)
    model = UAVSal(cnn_type=cnn_type, time_dims=time_dims, num_stblock=num_stblock, bias_type=bias_type,
                   iosize=iosize, planes=256, pre_model_path=pre_model_path)
    model = model.to(device)

    # When fine-tuning the model, you can fix some parameters to improve the training speed
    for p in model.sfnet.parameters():
        p.requires_grad = False
    for p in model.st_layer.parameters():
        p.requires_grad = False

    shape_r, shape_c, shape_r_out, shape_c_out = iosize

    criterion = loss_fu
    # When fine-tuning the model, it is recommended to use a smaller learning rate, like lr=1e-5, weight_decay=0.000005
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad == True], lr=1e-4,
                                 betas=(0.9, 0.999), weight_decay=0.00005)

    #################################################################
    # Train the model
    #################################################################
    print("Training UAVSal Model")
    min_val_loss = 10000
    num_patience = 0
    if IS_EARLY_STOP:
        max_patience = Max_patience
    else:
        max_patience = epochs + 1

    use_cb = np.sum(np.array(bias_type) > 0)
    x_cb = get_bias(bias_type, batch_size * time_dims, shape_r_out, shape_c_out)

    for epoch in range(epochs):
        print("\nEpochs: %d / %d " % (epoch + 1, epochs))
        for phase in ['train', 'val']:
            num_step = 0
            run_loss = 0.0
            if phase == 'train':
                model.train()
                shuffle = Shuffle_Train
                Max_TrainValFrame = Max_TrainFrame
            else:
                model.eval()
                shuffle = False
                Max_TrainValFrame = Max_ValFrame

            videos_list, vidmaps_list, vidfixs_list = read_video_list(train_dataDir, phase, shuffle=shuffle, ext=ext)

            for idx_video in range(len(videos_list)):
                print("Videos: %d / %d, %s with data from: %s" % (
                    idx_video + 1, len(videos_list), phase.upper(), videos_list[idx_video]))

                vidmaps = preprocess_vidmaps(vidmaps_list[idx_video], shape_r_out, shape_c_out, Max_TrainValFrame)
                vidfixs = preprocess_vidfixs(vidfixs_list[idx_video], shape_r_out, shape_c_out, Max_TrainValFrame)
                vidimgs, nframes, height, width = preprocess_videos(videos_list[idx_video], shape_r, shape_c,
                                                                    Max_TrainValFrame, mode='RGB', normalize=False)
                nframes = min(min(vidfixs.shape[0], vidmaps.shape[0]), nframes)

                count_bs = nframes // time_dims
                trainFrames = count_bs * time_dims
                vidimgs = vidimgs[0:trainFrames].transpose((0, 3, 1, 2))
                vidgaze = np.concatenate((vidmaps[0:trainFrames], vidfixs[0:trainFrames]), axis=-1).transpose(
                    (0, 3, 1, 2))

                count_input = batch_size * time_dims
                bs_steps = math.ceil(count_bs / batch_size)
                video_loss = 0.0
                x_state = None
                for idx_bs in range(bs_steps):
                    x_imgs = vidimgs[idx_bs * count_input:(idx_bs + 1) * count_input]
                    y_gaze = vidgaze[idx_bs * count_input:(idx_bs + 1) * count_input]

                    if not np.any(y_gaze, axis=(2, 3)).all():
                        continue

                    if use_cb and x_imgs.shape[0] != count_input:
                        x_cb_input = get_bias(bias_type, x_imgs.shape[0], shape_r_out, shape_c_out)
                    else:
                        x_cb_input = x_cb

                    x_imgs = torch.tensor(norm_data(x_imgs)).float()
                    y_gaze = torch.tensor(y_gaze).float()

                    optimizer.zero_grad()
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs, out_state = model(x_imgs.to(device), x_cb_input, x_state)
                        loss = criterion(outputs, y_gaze.to(device))
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                        x_state = [out_state[0].detach()]

                    batch_loss = loss.data.item()
                    video_loss += batch_loss
                    run_loss += batch_loss
                    num_step += 1

                    print("    Batch: [%d / %d], %s loss : %.4f " % (idx_bs + 1, bs_steps, phase.upper(), batch_loss))

                print("    Mean %s loss: %.4f " % (phase.upper(), video_loss / bs_steps))

            mean_run_loss = run_loss / num_step
            print("Epoch: %d / %d, Mean %s loss: %.4f" % (epoch + 1, epochs, phase.upper(), mean_run_loss))

        if not IS_BEST_ONLY:
            output_modename = save_model_path + "%02d_%.4f.pth" % (epoch, mean_run_loss)
            torch.save(model, output_modename)
        if mean_run_loss < min_val_loss:
            min_val_loss = mean_run_loss
            num_patience = 0
            best_model_wts = copy.deepcopy(model.state_dict())
        else:
            num_patience += 1
            if num_patience >= max_patience:
                print('Early stop')
                break

    # Save the best model
    finalmode_name = save_model_path + "final.pth"
    model.load_state_dict(best_model_wts)
    torch.save(model, finalmode_name)


def test(input_path, output_path, method_name,
         saveFrames=float('inf'),
         time_dims=5,
         iosize=[480, 640, 60, 80],
         batch_size=4,
         bias_type=[1, 1, 1]):

    model_path = saveModelDir + method_name + '/' + method_name + '_final.pth'
    model = torch.load(model_path)
    model = model.to(device)

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
            time_sum = 0
            for idx_bs in range(bs_steps):
                x_imgs = vidimgs[idx_bs * count_input:(idx_bs + 1) * count_input]
                x_imgs = torch.tensor(norm_data(x_imgs)).float()

                if use_cb and x_imgs.shape[0] != count_input:
                    x_cb_input = get_bias(bias_type, x_imgs.shape[0], shape_r_out, shape_c_out)
                else:
                    x_cb_input = x_cb

                time_start = time.time()
                bs_out, out_state = model(x_imgs.to(device), x_cb_input, x_state)
                time_end = time.time()
                time_sum = time_sum + time_end - time_start

                x_state = [out_state[0].detach()]
                bs_out = bs_out.data.cpu().numpy()

                for idx_pre in range(bs_out.shape[0]):
                    isalmap = postprocess_predictions(bs_out[idx_pre, 0, :, :], height, width)
                    pred_mat[idx_bs * count_input + idx_pre, :, :, 0] = np2mat(isalmap)

            print('time cost: ', time_sum, 's')
            print('frames: ',str(nframes))

            iSaveFrame = min(isaveframes, saveFrames)
            pred_mat = pred_mat[0:iSaveFrame, :, :, :].transpose((1, 2, 3, 0))
            h5io.savemat(ovideo_path, {'salmap': pred_mat})


#########################################################################
# Training Settings											            #
#########################################################################
IS_EARLY_STOP = True
IS_BEST_ONLY = False
Shuffle_Train = True
Max_patience = 4

Max_TrainFrame = float('inf')
Max_ValFrame = float('inf')
saveFrames = float('inf')

################################################################
# DATASET PARAMETERS
################################################################
# replace the datadir to your path
if os.name == 'nt':
    dataDir = 'E:/DataSet/'
else:
    dataDir = '/home/kao/D/DataSet/'

DataSet_Train  = 'UAV2'
DataSet_Test = 'UAV2-TE'

train_dataDir = dataDir + '/' + DataSet_Train  + '/'
test_dataDir = dataDir + '/' + DataSet_Test + '/'

test_input_path = test_dataDir + 'Videos/'
test_result_path = test_dataDir + 'Results/Results_UAVSal/'
test_output_path = test_result_path + 'Saliency/'

saveModelDir = './weights/temp_weights/'
pre_model_path = './weights/uavsal-mobilenet_v2-uav2-v2.pth'

if DataSet_Train  in ['UAV2']:
    ext = '.avi'
else:
    ext = '.mp4'


if __name__ == '__main__':

    method_name = 'UAVSal'
    epochs = 20
    batch_size = 2

    time_dims = 5
    num_stblock = 2
    bias_type = [1, 1, 1]

    train(cnn_type='mobilenet_v2', time_dims=time_dims, num_stblock=num_stblock, bias_type=bias_type,
          iosize=[360, 640, 45, 80], batch_size=batch_size, epochs=epochs, pre_model_path=pre_model_path)

    test(test_input_path, test_output_path, method_name=method_name, saveFrames=saveFrames, iosize=[360, 640, 45, 80],
         batch_size=batch_size, time_dims=time_dims, bias_type=bias_type)

    evalscores_vid_torch(test_dataDir, test_result_path, DataSet=DataSet_Test, MethodNames=[method_name], batch_size=32)

    visual_vid(test_dataDir, test_result_path, DataSet=DataSet_Test, MethodNames=[method_name], with_color=1,
               with_fix=0)
