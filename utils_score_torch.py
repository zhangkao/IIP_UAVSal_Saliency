#####################################################
# STRNN metrics and SalUAV metrics (based on numpy and pytorch)
# Written by kao zhang (kaozhang@outlook.com), 20211212
#####################################################

from functools import partial
import numpy as np

import hdf5storage as h5io
import os, cv2, torch, math, time


EPS = 2.2204e-16

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

keys_order = ['AUC_shuffled', 'NSS', 'AUC_Judd', 'AUC_Borji', 'KLD', 'SIM', 'CC']


def get_sum(input):
    size_h, size_w = input.shape[2:]
    v_sum = torch.sum(input, (2, 3), keepdim=True)
    return v_sum.repeat(1, 1, size_h, size_w)


def get_max(input):
    size_h, size_w = input.shape[2:]
    v_max = torch.max(torch.max(input, 2, keepdim=True)[0], 3, keepdim=True)[0]
    return v_max.repeat(1, 1, size_h, size_w)


def get_min(input):
    size_h, size_w = input.shape[2:]
    v_max = torch.min(torch.min(input, 2, keepdim=True)[0], 3, keepdim=True)[0]
    return v_max.repeat(1, 1, size_h, size_w)


def get_mean(input):
    size_h, size_w = input.shape[2:]
    v_mean = torch.mean(input, (2, 3), keepdim=True)
    return v_mean.repeat(1, 1, size_h, size_w)


def get_std(input):
    size_h, size_w = input.shape[2:]
    # v_mean = torch.mean(input, (2,3), keepdim=True)
    # tmp = torch.sum((input-v_mean)**2,(2,3),keepdim=True) / (size_h*size_w-1)
    # return torch.sqrt(tmp).repeat(1,1, size_h, size_w)
    v_std = torch.std(input, (2, 3), keepdim=True)
    return v_std.repeat(1, 1, size_h, size_w)


def auc_j(S, F):
    if not torch.any(S > 0) or not torch.any(F > 0):
        return torch.tensor(float('nan'))

    S_fix = S[F]  # Saliency map values at fixation locations
    n_fix = S_fix.shape[0]
    n_pixels = S.shape[0]
    # Calculate AUC
    thresholds, _ = torch.sort(S_fix, descending=True)
    tp = torch.zeros(n_fix + 2)
    fp = torch.zeros(n_fix + 2)
    tp[0] = 0
    tp[-1] = 1
    fp[0] = 0
    fp[-1] = 1

    tp[1:-1] = (torch.arange(0, n_fix) + 1) / float(n_fix)
    above_th = torch.tensor([torch.sum(S >= thresh) for thresh in thresholds])
    fp[1:-1] = (above_th - torch.arange(0, n_fix) - 1) / float(n_pixels - n_fix)

    return torch.trapz(tp, fp)  # y, x


def metric_auc_j(y_pred, y_true, jitter=1):
    y_true = y_true[:, 1:2, :, :] > 0.5

    if jitter == True:
        y_pred = y_pred + (torch.rand(y_pred.shape) * 1e-7).to(y_pred.device)
    y_pred = (y_pred - get_min(y_pred)) / (get_max(y_pred) - get_min(y_pred) + EPS)

    S = torch.flatten(y_pred, 1, -1)
    F = torch.flatten(y_true, 1, -1)

    score = torch.Tensor([auc_j(S[i], F[i]) for i in range(S.shape[0])])

    return score.unsqueeze(1)


def auc_b(S, F):
    if not np.any(S > 0) or not np.any(F > 0):
        return torch.tensor(float('nan'))

    n_rep = 100
    step_size = 0.1

    S_fix = S[F]  # Saliency map values at fixation locations
    n_fix = len(S_fix)
    n_pixels = len(S)

    r = np.random.randint(0, n_pixels, [n_fix, n_rep])
    S_rand = S[r]

    # Calculate AUC
    auc = np.zeros(n_rep) * np.nan
    for rep in range(n_rep):
        thresholds = np.r_[0:np.max(np.r_[S_fix, S_rand[:, rep]]):step_size][::-1]
        tp = np.zeros(len(thresholds) + 2)
        fp = np.zeros(len(thresholds) + 2)
        tp[0] = 0;
        tp[-1] = 1
        fp[0] = 0;
        fp[-1] = 1
        for k, thresh in enumerate(thresholds):
            tp[k + 1] = np.sum(S_fix >= thresh) / float(n_fix)
            fp[k + 1] = np.sum(S_rand[:, rep] >= thresh) / float(n_fix)
        auc[rep] = np.trapz(tp, fp)
    return np.mean(auc)  # Average across random splits


def metric_auc_b(y_pred, y_true):
    y_true = y_true[:, 1:2, :, :] > 0.5
    y_pred = (y_pred - get_min(y_pred)) / (get_max(y_pred) - get_min(y_pred) + EPS)

    S = torch.flatten(y_pred, 1, -1).numpy()
    F = torch.flatten(y_true, 1, -1).numpy()

    score = torch.Tensor([auc_b(S[i], F[i]) for i in range(S.shape[0])])

    return score.unsqueeze(1)


def auc_s(S, F, Oth):
    if not np.any(S > 0) or not np.any(F > 0):
        return torch.tensor(float('nan'))

    n_rep = 100
    step_size = 0.1

    S_fix = S[F]  # Saliency map values at fixation locations
    n_fix = len(S_fix)

    ind = np.nonzero(Oth)[0]
    n_ind = len(ind)
    n_fix_oth = min(n_fix, n_ind)

    r = np.random.randint(0, n_ind, [n_ind, n_rep])[:n_fix_oth, :]
    S_rand = S[ind[r]]

    auc = np.zeros(n_rep) * np.nan
    for rep in range(n_rep):
        thresholds = np.r_[0:np.max(np.r_[S_fix, S_rand[:, rep]]):step_size][::-1]
        tp = np.zeros(len(thresholds) + 2)
        fp = np.zeros(len(thresholds) + 2)
        tp[0] = 0;
        tp[-1] = 1
        fp[0] = 0;
        fp[-1] = 1
        for k, thresh in enumerate(thresholds):
            tp[k + 1] = np.sum(S_fix >= thresh) / float(n_fix)
            fp[k + 1] = np.sum(S_rand[:, rep] >= thresh) / float(n_fix_oth)
        auc[rep] = np.trapz(tp, fp)
    return np.mean(auc)


def metric_auc_s(y_pred, y_true, shuff_map):
    y_true = y_true[:, 1:2, :, :] > 0.5
    y_pred = (y_pred - get_min(y_pred)) / (get_max(y_pred) - get_min(y_pred) + EPS)

    S = torch.flatten(y_pred, 1, -1).numpy()
    F = torch.flatten(y_true, 1, -1).numpy()
    O = torch.flatten(shuff_map, 1, -1).numpy()

    score = torch.Tensor([auc_s(S[i], F[i], O[i]) for i in range(S.shape[0])])

    return score.unsqueeze(1)


def metric_kl(y_pred, y_true):
    y_true = y_true[:, 0:1, :, :]
    y_true = y_true / (get_sum(y_true) + EPS)
    y_pred = y_pred / (get_sum(y_pred) + EPS)

    return torch.sum(y_true * torch.log((y_true / (y_pred + EPS)) + EPS), (2, 3))


def metric_cc(y_pred, y_true):
    y_true = y_true[:, 0:1, :, :]
    y_true = (y_true - get_mean(y_true)) / (get_std(y_true) + EPS)
    y_pred = (y_pred - get_mean(y_pred)) / (get_std(y_pred) + EPS)

    y_true = y_true - get_mean(y_true)
    y_pred = y_pred - get_mean(y_pred)
    r1 = torch.sum(y_true * y_pred, (2, 3))
    r2 = torch.sqrt(torch.sum(y_pred * y_pred, (2, 3)) * torch.sum(y_true * y_true, (2, 3)))
    return r1 / (r2 + EPS)


def metric_nss(y_pred, y_true):
    y_true = y_true[:, 1:2, :, :]
    y_pred = (y_pred - get_mean(y_pred)) / (get_std(y_pred) + EPS)

    return torch.sum(y_true * y_pred, dim=(2, 3)) / (torch.sum(y_true, dim=(2, 3)) + EPS)


def metric_sim(y_pred, y_true):
    y_true = y_true[:, 0:1, :, :]
    y_true = (y_true - get_min(y_true)) / (get_max(y_true) - get_min(y_true) + EPS)
    y_pred = (y_pred - get_min(y_pred)) / (get_max(y_pred) - get_min(y_pred) + EPS)

    y_true = y_true / (get_sum(y_true) + EPS)
    y_pred = y_pred / (get_sum(y_pred) + EPS)

    diff = torch.min(y_true, y_pred)
    score = torch.sum(diff, dim=(2, 3))

    return score


metrics = {
    "AUC_shuffled": metric_auc_s,  # Binary fixation map
    "AUC_Judd": metric_auc_j,  # Binary fixation map
    "AUC_Borji": metric_auc_b,  # Binary fixation map
    "NSS": metric_nss,  # Binary fixation map
    "CC": metric_cc,  # Saliency map
    "SIM": metric_sim,  # Saliency map
    "KLD": metric_kl,  # Saliency map
}

shuff_size = {
    "SALICON": (480, 640),
    "DIEM": (480, 640),
    "DIEM20": (480, 640),
    "CITIUS": (240, 320),
    "SFU": (288, 352),
    "LEDOV": (1080, 1920),
    "LEDOV41": (1080, 1920),
    "UAV2-TE": (720, 1280),
    "UAV2": (720, 1280),
    "default": (480, 640),
    "AVS1K-TE": (720, 1280),
    "AVS1K": (720, 1280),
}


def resize_fixation(img, rows=480, cols=640):
    out = np.zeros((rows, cols), np.uint8)
    factor_scale_r = rows / img.shape[0]
    factor_scale_c = cols / img.shape[1]

    coords = np.argwhere(img)
    for coord in coords:
        r = int(np.round(coord[0] * factor_scale_r))
        c = int(np.round(coord[1] * factor_scale_c))
        if r == rows:
            r -= 1
        if c == cols:
            c -= 1
        out[r, c] = 1

    return out


def getSumFix_vid(fixsDir, DataSet='DIEM20', size=None, maxframes=float('inf')):
    DataSet = DataSet.upper()
    if size is None:
        if DataSet in shuff_size.keys():
            size = shuff_size[DataSet]
        else:
            size = shuff_size["default"]

    fix_names = [f for f in os.listdir(fixsDir) if f.endswith('.mat')]
    fix_names.sort()
    fix_num = len(fix_names)

    # if DataSet == 'CITIUS':
    # 	fix_num = 45

    if DataSet == 'DIEM20':
        maxframes = 300

    ShufMap = np.zeros(size)
    for idx_n in range(fix_num):

        fixpts = h5io.loadmat(fixsDir + fix_names[idx_n])["fixLoc"]
        useframes = min(maxframes, fixpts.shape[3])
        fixpts = fixpts[:, :, :, :useframes]

        if fixpts.shape[:2] != size:
            fixpts = np.array(
                [resize_fixation(fixpts[:, :, 0, i], size[0], size[1]) for i in range(useframes)]).transpose((1, 2, 0))
            # fixpts = np.array([cv2.resize(fixpts[:, :, 0, i], (size[1], size[0]),interpolation=cv2.INTER_NEAREST) for i in range(useframes)]).transpose((1, 2, 0))
            fixpts = np.expand_dims(fixpts, axis=2)

        ShufMap += np.sum(fixpts[:, :, 0, :], axis=2)
        ShufMap = np.round(ShufMap)

    h5io.savemat('./Tools/Shuffle_' + DataSet + '.mat', {'ShufMap': ShufMap})
    return ShufMap


def getALLFix_vid(fixsDir, DataSet='DIEM20', maxframes=float('inf')):
    fix_names = [f for f in os.listdir(fixsDir) if f.endswith('.mat')]
    fix_names.sort()
    fix_num = len(fix_names)

    DataSet = DataSet.upper()
    if DataSet == 'CITIUS':
        fix_num = 45

    if DataSet == 'DIEM20':
        maxframes = 300

    ALLFixPts = []
    # ALLFixMaps = []
    for idx_n in range(fix_num):

        fixpts = h5io.loadmat(fixsDir + fix_names[idx_n])["fixLoc"]
        useframes = min(maxframes, fixpts.shape[3])
        fixpts = fixpts[:, :, :, :useframes]

        for idx_f in range(useframes):
            # ALLFixMaps.append(fixpts[:,:,0,idx_f])

            fx, fy = np.where(fixpts[:, :, 0, idx_f])
            fx = fx / fixpts.shape[0]
            fy = fy / fixpts.shape[1]
            f_xy = np.concatenate((np.expand_dims(fx, 1), np.expand_dims(fy, 1)), 1)

            ALLFixPts.append(f_xy)

    np.save('./Tools/ALLFixPts_' + DataSet + '.npy', ALLFixPts)
    # np.save('../Tools/ALLFixMaps_' + DataSet + '.npy', ALLFixMaps)
    return ALLFixPts  # , ALLFixMaps


def getshufmap(ALLFixPts, size=(480, 640), nframes=10):
    nframes = min(nframes, len(ALLFixPts))
    shuf_idx = np.random.randint(0, len(ALLFixPts), int(nframes))

    fix_nf = ALLFixPts[shuf_idx[0]]
    for i in range(1, nframes):
        fix_f = ALLFixPts[shuf_idx[i]]
        fix_nf = np.concatenate((fix_nf, fix_f), 0)

    fix_nf[:, 0] *= size[0]
    fix_nf[:, 1] *= size[1]

    fix_nf = np.round(fix_nf).astype(np.int)

    bound_fix = (fix_nf[:, 0] < size[0]) * (fix_nf[:, 1] < size[1])
    fix_nf = fix_nf[bound_fix]

    shufmap = np.zeros(size, dtype=np.uint8)
    shufmap[fix_nf[:, 0], fix_nf[:, 1]] = 1

    # shufmap = np.zeros(size, dtype=np.uint8)
    # for i in range(fix_nf.shape[0]):
    # 	shufmap[fix_nf[i, 0], fix_nf[i, 1]] = 1

    return shufmap


###TwoS metrics (Matlab version):
# ishuffle_map = cv2.resize(shuffle_map, (fixpts.shape[1], fixpts.shape[0]), interpolation=cv2.INTER_NEAREST)
# to
# ishuffle_map = cv2.resize(shuffle_map, (fixpts.shape[1], fixpts.shape[0]), interpolation=cv2.INTER_LINEAR)

###STRNN metrics
def evalscores_vid_torch_sum(RootDir, SalDir, DataSet, MethodNames, keys_order=keys_order, batch_size=64):
    mapsDir = RootDir + 'maps/'
    fixsDir = RootDir + 'fixations/maps/'

    salsDir = SalDir + 'Saliency/'
    scoreDir = SalDir + 'Scores_sum/'
    if not os.path.exists(scoreDir):
        os.makedirs(scoreDir)

    print('\nEvaluate Metrics: ' + str(keys_order))
    if 'AUC_shuffled' in keys_order:
        shuff_path = './Tools/Shuffle_' + DataSet.upper() + '.mat'
        if not os.path.exists(shuff_path):
            shuffle_map = getSumFix_vid(fixsDir, DataSet)
        else:
            shuffle_map = h5io.loadmat(shuff_path)["ShufMap"]
    else:
        shuffle_map = []

    for idx_m in range(len(MethodNames)):
        print("---" + str(idx_m + 1) + "/" + str(len(MethodNames)) + "---: " + MethodNames[idx_m])

        score_path = scoreDir + 'Score_' + MethodNames[idx_m] + '.mat'
        if os.path.exists(score_path):
            continue

        iscoreDir = scoreDir + MethodNames[idx_m] + '/'
        if not os.path.exists(iscoreDir):
            os.makedirs(iscoreDir)

        salmap_dir = salsDir + MethodNames[idx_m] + '/'
        sal_names = [f for f in os.listdir(salmap_dir) if f.endswith('.mat')]
        sal_names.sort()

        scores = {}
        # for idx_n in range(len(sal_names)):
        for idx_n in range(len(sal_names)):
            print(str(idx_n + 1) + "/" + str(len(sal_names)) + ": " + sal_names[idx_n])

            file_name = sal_names[idx_n][:-4]
            iscore_path = iscoreDir + 'Score_' + file_name + '.mat'
            if os.path.exists(iscore_path):
                iscores = h5io.loadmat(iscore_path)["iscore"]
                scores[file_name] = iscores
                continue

            salmap = h5io.loadmat(salmap_dir + file_name + '.mat')["salmap"]
            fixmap = h5io.loadmat(mapsDir + file_name + '_fixMaps.mat')["fixMap"]
            fixpts = h5io.loadmat(fixsDir + file_name + '_fixPts.mat')["fixLoc"]

            if shuffle_map != [] and shuffle_map.shape != fixpts.shape[:2]:
                ishuffle_map = resize_fixation(shuffle_map, fixpts.shape[0], fixpts.shape[1])
            # ishuffle_map = cv2.resize(shuffle_map, (fixpts.shape[1], fixpts.shape[0]), interpolation=cv2.INTER_NEAREST)
            else:
                ishuffle_map = shuffle_map
            ishuffle_map = torch.tensor(ishuffle_map).float()

            nframes = min(salmap.shape[3], min(fixpts.shape[3], fixmap.shape[3]))
            iscores = np.zeros((nframes, len(keys_order)))

            salmap = salmap[:, :, :, 0:nframes].transpose((3, 2, 0, 1))
            fixmap = np.concatenate((fixmap[:, :, :, 0:nframes], fixpts[:, :, :, 0:nframes]), axis=2).transpose(
                (3, 2, 0, 1))

            assert salmap.shape[2:] == fixmap.shape[2:]

            t1 = time.time()
            for idx_m, metric in enumerate(keys_order):
                func = metrics[metric]

                count_bs = math.ceil(nframes / batch_size)
                for idx_bs in range(count_bs):
                    ipred = salmap[idx_bs * batch_size:(idx_bs + 1) * batch_size, :, :, :]
                    itrue = fixmap[idx_bs * batch_size:(idx_bs + 1) * batch_size, :, :, :]
                    ipred = torch.tensor(ipred).float()
                    itrue = torch.tensor(itrue).float()

                    if metric in ['AUC_shuffled']:
                        m = func(ipred, itrue, ishuffle_map)
                    elif metric in ['AUC_Borji']:
                        m = func(ipred, itrue)
                    else:
                        m = func(ipred.to(device), itrue.to(device))
                    iscores[idx_bs * batch_size:(idx_bs + 1) * batch_size, idx_m] = m.data.cpu()[:, 0]

            for idx_f in range(nframes):
                isalmap = salmap[idx_f, 0, :, :]
                ifixmap = fixmap[idx_f, :, :, :]
                if not np.any(isalmap) or not np.any(ifixmap, axis=(1, 2)).all():
                    iscores[idx_f] = np.NaN
                    print(str(idx_f + 1) + "/" + str(nframes) + ": failed!")
                    continue

            t2 = time.time() - t1
            print('\t\t time: %.3f S' % (t2))

            scores[file_name] = iscores
            h5io.savemat(iscore_path, {'iscore': iscores})

    # h5io.savemat(score_path, {'scores': scores})


###SalUAV metrics
def evalscores_vid_torch(RootDir, SalDir, DataSet, MethodNames, keys_order=keys_order, batch_size=64):
    mapsDir = RootDir + 'maps/'
    fixsDir = RootDir + 'fixations/maps/'

    salsDir = SalDir + 'Saliency/'
    scoreDir = SalDir + 'Scores/'
    if not os.path.exists(scoreDir):
        os.makedirs(scoreDir)

    print('\nEvaluate Metrics: ' + str(keys_order))
    if 'AUC_shuffled' in keys_order:
        ALLFixPts_path = './Tools/ALLFixPts_' + DataSet.upper() + '.npy'
        if not os.path.exists(ALLFixPts_path):
            ALLFixPts = getALLFix_vid(fixsDir, DataSet)
        else:
            ALLFixPts = np.load(ALLFixPts_path, allow_pickle=True)
    else:
        ALLFixPts = []

    for idx_m in range(len(MethodNames)):
        print("---" + str(idx_m + 1) + "/" + str(len(MethodNames)) + "---: " + MethodNames[idx_m])
        method_t1 = time.time()

        score_path = scoreDir + 'Score_' + MethodNames[idx_m] + '.mat'
        if os.path.exists(score_path):
            continue

        iscoreDir = scoreDir + MethodNames[idx_m] + '/'
        if not os.path.exists(iscoreDir):
            os.makedirs(iscoreDir)

        salmap_dir = salsDir + MethodNames[idx_m] + '/'
        sal_names = [f for f in os.listdir(salmap_dir) if f.endswith('.mat')]
        sal_names.sort()

        scores = {}
        for idx_n in range(len(sal_names)):
            # for idx_n in range(160,250):
            print(str(idx_n + 1) + "/" + str(len(sal_names)) + ": " + sal_names[idx_n])

            file_name = sal_names[idx_n][:-4]
            iscore_path = iscoreDir + 'Score_' + file_name + '.mat'
            if os.path.exists(iscore_path):
                iscores = h5io.loadmat(iscore_path)["iscore"]
                scores[file_name] = iscores
                continue

            vid_t1 = time.time()
            salmap = h5io.loadmat(salmap_dir + file_name + '.mat')["salmap"]
            fixmap = h5io.loadmat(mapsDir + file_name + '_fixMaps.mat')["fixMap"]
            fixpts = h5io.loadmat(fixsDir + file_name + '_fixPts.mat')["fixLoc"]

            nframes = min(salmap.shape[3], min(fixpts.shape[3], fixmap.shape[3]))
            iscores = np.zeros((nframes, len(keys_order)))

            if salmap.shape[:2] != fixmap.shape[:2]:
                # salmap = np.expand_dims(cv2.resize(salmap[:, :, 0, :nframes], (fixmap.shape[1], fixmap.shape[0])).transpose((2, 0, 1)), 1)
                salmap_rs = np.zeros((nframes, 1, fixmap.shape[0], fixmap.shape[1]))
                for idx_rs in range(nframes):
                    salmap_rs[idx_rs, 0, :, :] = cv2.resize(salmap[:, :, 0, idx_rs], (fixmap.shape[1], fixmap.shape[0]))
                salmap = salmap_rs
            else:
                salmap = salmap[:, :, :, :nframes].transpose((3, 2, 0, 1))

            fixmap = np.concatenate((fixmap[:, :, :, :nframes], fixpts[:, :, :, :nframes]), axis=2).transpose((3, 2, 0, 1))

            # t1 = time.time()
            for idx_k, metric in enumerate(keys_order):
                metric_t1 = time.time()

                func = metrics[metric]
                count_bs = math.ceil(nframes / batch_size)
                for idx_bs in range(count_bs):
                    ipred = salmap[idx_bs * batch_size:(idx_bs + 1) * batch_size, :, :, :]
                    itrue = fixmap[idx_bs * batch_size:(idx_bs + 1) * batch_size, :, :, :]
                    ipred = torch.tensor(ipred).float()
                    itrue = torch.tensor(itrue).float()

                    if metric in ['AUC_shuffled']:
                        ishuffle_map = np.array(
                            [getshufmap(ALLFixPts, size=fixmap.shape[2:]) for i in range(itrue.shape[0])])
                        ishuffle_map = torch.tensor(ishuffle_map).float().unsqueeze(1)
                        m = func(ipred, itrue, ishuffle_map)
                    elif metric in ['AUC_Borji']:
                        m = func(ipred, itrue)
                    else:
                        m = func(ipred.to(device), itrue.to(device))
                    iscores[idx_bs * batch_size:(idx_bs + 1) * batch_size, idx_k] = m.data.cpu()[:, 0]

                metric_t2 = time.time() - metric_t1
                print('\t\t\t' + metric + ' time: %.3f S' % (metric_t2))

            for idx_f in range(nframes):
                isalmap = salmap[idx_f, 0, :, :]
                ifixmap = fixmap[idx_f, :, :, :]
                if not np.any(isalmap) or not np.any(ifixmap, axis=(1, 2)).all():
                    iscores[idx_f] = np.NaN
                    print(str(idx_f + 1) + "/" + str(nframes) + ": failed!")
                    continue

            vid_t2 = time.time() - vid_t1
            print('\t\t Total time: %.3f S' % (vid_t2))

            scores[file_name] = iscores
            h5io.savemat(iscore_path, {'iscore': iscores})

        method_t2 = time.time() - method_t1
        print('\t' + MethodNames[idx_m] + ' time: %.3f S' % (method_t2))
    # h5io.savemat(score_path, {'scores': scores})
