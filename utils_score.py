#####################################################
# STRNN metrics and SalUAV metrics (based on numpy)
# Written by kao zhang (kaozhang@outlook.com), 20211212
#####################################################

from functools import partial
import numpy as np
from numpy import random
from skimage.transform import resize

import hdf5storage as h5io
import os, cv2, time

EPS = 2.2204e-16

def normalize(x, method='standard', axis=None):

    x = np.array(x, copy=True)
    if axis is not None:
        y = np.rollaxis(x, axis).reshape([x.shape[axis], -1])
        shape = np.ones(len(x.shape))
        shape[axis] = x.shape[axis]
        if method == 'standard':
            res = (x - np.mean(y, axis=1).reshape(shape)) / np.std(y, axis=1).reshape(shape)
        elif method == 'range':
            res = (x - np.min(y, axis=1).reshape(shape)) / (np.max(y, axis=1) - np.min(y, axis=1)).reshape(shape)
        elif method == 'sum':
            res = x / np.float_(np.sum(y, axis=1).reshape(shape))
        else:
            raise ValueError('method not in {"standard", "range", "sum"}')
    else:
        if method == 'standard':
            res = (x - np.mean(x)) / np.std(x)
        elif method == 'range':
            res = (x - np.min(x)) / (np.max(x) - np.min(x) + EPS)
        elif method == 'sum':
            res = x / float(np.sum(x))
        else:
            raise ValueError('method not in {"standard", "range", "sum"}')
    return res

def AUC_Judd(saliency_map, fixation_map, jitter=True):

    s_map = np.array(saliency_map, copy=True)
    f_map = np.array(fixation_map, copy=True) > 0.5
    # If there are no fixation to predict, return NaN
    if not np.any(f_map):
        print('no fixation to predict')
        return np.nan
    # Make the saliency_map the size of the fixation_map
    if s_map.shape != f_map.shape:
        s_map = resize(s_map, f_map.shape, order=3, mode='nearest')
    # Jitter the saliency map slightly to disrupt ties of the same saliency value
    if jitter:
        s_map += random.rand(*s_map.shape) * 1e-7
    # Normalize saliency map to have values between [0,1]
    s_map = normalize(s_map, method='range')

    S = s_map.ravel()
    F = f_map.ravel()
    S_fix = S[F] # Saliency map values at fixation locations
    n_fix = len(S_fix)
    n_pixels = len(S)
    # Calculate AUC
    thresholds = sorted(S_fix, reverse=True)
    tp = np.zeros(n_fix+2)
    fp = np.zeros(n_fix+2)
    tp[0] = 0; tp[-1] = 1
    fp[0] = 0; fp[-1] = 1
    for k, thresh in enumerate(thresholds):
        above_th = np.sum(S >= thresh) # Total number of saliency map values above threshold
        tp[k+1] = (k + 1) / float(n_fix) # Ratio saliency map values at fixation locations above threshold
        fp[k+1] = (above_th - k - 1) / float(n_pixels - n_fix) # Ratio other saliency map values above threshold
    return np.trapz(tp, fp) # y, x

def AUC_Borji(saliency_map, fixation_map, n_rep=100, step_size=0.1, rand_sampler=None):

    s_map = np.array(saliency_map, copy=True)
    f_map = np.array(fixation_map, copy=True) > 0.5
    # If there are no fixation to predict, return NaN
    if not np.any(f_map):
        print('no fixation to predict')
        return np.nan
    # Make the saliency_map the size of the fixation_map
    if s_map.shape != f_map.shape:
        s_map = resize(s_map, f_map.shape, order=3, mode='nearest')
    # Normalize saliency map to have values between [0,1]
    s_map = normalize(s_map, method='range')

    S = s_map.ravel()
    F = f_map.ravel()
    S_fix = S[F] # Saliency map values at fixation locations
    n_fix = len(S_fix)
    n_pixels = len(S)
    # For each fixation, sample n_rep values from anywhere on the saliency map
    if rand_sampler is None:
        r = random.randint(0, n_pixels, [n_fix, n_rep])
        S_rand = S[r] # Saliency map values at random locations (including fixated locations!? underestimated)
    else:
        S_rand = rand_sampler(S, F, n_rep, n_fix)
    # Calculate AUC per random split (set of random locations)
    auc = np.zeros(n_rep) * np.nan
    for rep in range(n_rep):
        thresholds = np.r_[0:np.max(np.r_[S_fix, S_rand[:,rep]]):step_size][::-1]
        tp = np.zeros(len(thresholds)+2)
        fp = np.zeros(len(thresholds)+2)
        tp[0] = 0; tp[-1] = 1
        fp[0] = 0; fp[-1] = 1
        for k, thresh in enumerate(thresholds):
            tp[k+1] = np.sum(S_fix >= thresh) / float(n_fix)
            fp[k+1] = np.sum(S_rand[:,rep] >= thresh) / float(n_fix)
        auc[rep] = np.trapz(tp, fp)
    return np.mean(auc) # Average across random splits

def AUC_shuffled(saliency_map, fixation_map, other_map, n_rep=100, step_size=0.1):

    s_map = np.array(saliency_map, copy=True)
    f_map = np.array(fixation_map, copy=True) > 0.5
    o_map = np.array(other_map, copy=True) > 0.5
    if other_map.shape != fixation_map.shape:
        raise ValueError('other_map.shape != fixation_map.shape')
    if not np.any(f_map):
        print('no fixation to predict')
        return np.nan
    if s_map.shape != f_map.shape:
        s_map = resize(s_map, f_map.shape, order=3, mode='nearest')
    s_map = normalize(s_map, method='range')

    S = s_map.ravel()
    F = f_map.ravel()
    Oth = o_map.ravel()

    S_fix = S[F] # Saliency map values at fixation locations
    n_fix = len(S_fix)

    ind = np.nonzero(Oth)[0]
    n_ind = len(ind)
    n_fix_oth = min(n_fix,n_ind)

    r = random.randint(0, n_ind, [n_ind, n_rep])[:n_fix_oth,:]
    S_rand = S[ind[r]]

    auc = np.zeros(n_rep) * np.nan
    for rep in range(n_rep):
        thresholds = np.r_[0:np.max(np.r_[S_fix, S_rand[:,rep]]):step_size][::-1]
        tp = np.zeros(len(thresholds)+2)
        fp = np.zeros(len(thresholds)+2)
        tp[0] = 0; tp[-1] = 1
        fp[0] = 0; fp[-1] = 1
        for k, thresh in enumerate(thresholds):
            tp[k+1] = np.sum(S_fix >= thresh) / float(n_fix)
            fp[k+1] = np.sum(S_rand[:,rep] >= thresh) / float(n_fix_oth)
        auc[rep] = np.trapz(tp, fp)
    return np.mean(auc)

def NSS(saliency_map, fixation_map):

    s_map = np.array(saliency_map, copy=True)
    f_map = np.array(fixation_map, copy=True) > 0.5
    if s_map.shape != f_map.shape:
        s_map = resize(s_map, f_map.shape)
    # Normalize saliency map to have zero mean and unit std
    s_map = normalize(s_map, method='standard')
    # Mean saliency value at fixation locations
    return np.mean(s_map[f_map])

def KLD(saliency_map1, saliency_map2):

    map1 = np.array(saliency_map1, copy=True)
    map2 = np.array(saliency_map2, copy=True)
    if map1.shape != map2.shape:
        map1 = resize(map1, map2.shape, order=3, mode='nearest') # bi-cubic/nearest is what Matlab imresize() does by default
    # Normalize the two maps to have zero mean and unit std
    map1 = normalize(map1, method='sum')
    map2 = normalize(map2, method='sum')
    return np.sum(map2 * np.log(EPS + map2 / (map1+EPS)))

def CC(saliency_map1, saliency_map2):

    map1 = np.array(saliency_map1, copy=True)
    map2 = np.array(saliency_map2, copy=True)
    if map1.shape != map2.shape:
        map1 = resize(map1, map2.shape, order=3, mode='nearest') # bi-cubic/nearest is what Matlab imresize() does by default
    # Normalize the two maps to have zero mean and unit std
    map1 = normalize(map1, method='standard')
    map2 = normalize(map2, method='standard')
    # Compute correlation coefficient
    return np.corrcoef(map1.ravel(), map2.ravel())[0,1]

def SIM(saliency_map1, saliency_map2):

    map1 = np.array(saliency_map1, copy=True)
    map2 = np.array(saliency_map2, copy=True)
    if map1.shape != map2.shape:
        map1 = resize(map1, map2.shape, order=3, mode='nearest') # bi-cubic/nearest is what Matlab imresize() does by default
    # Normalize the two maps to have values between [0,1] and sum up to 1
    map1 = normalize(map1, method='range')
    map2 = normalize(map2, method='range')
    map1 = normalize(map1, method='sum')
    map2 = normalize(map2, method='sum')
    # Compute histogram intersection
    intersection = np.minimum(map1, map2)
    return np.sum(intersection)


metrics = {
	"AUC_shuffled": [AUC_shuffled, 'fix', True], # Binary fixation map
	"AUC_Judd": [AUC_Judd, 'fix', False], # Binary fixation map
	"AUC_Borji": [AUC_Borji, 'fix', False], # Binary fixation map
	"NSS": [NSS, 'fix', False], # Binary fixation map
	"CC": [CC, 'sal', False], # Saliency map
	"SIM": [SIM, 'sal', False], # Saliency map
	"KLD": [KLD, 'sal', False],  # Saliency map
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

keys_order = ['AUC_shuffled', 'NSS', 'AUC_Judd', 'AUC_Borji', 'KLD', 'SIM', 'CC']

def resize_fixation(img, rows=480, cols=640):
    out = np.zeros((rows, cols),np.uint8)
    factor_scale_r = rows / img.shape[0]
    factor_scale_c = cols / img.shape[1]

    coords = np.argwhere(img)
    for coord in coords:
        r = int(np.round(coord[0]*factor_scale_r))
        c = int(np.round(coord[1]*factor_scale_c))
        if r == rows:
            r -= 1
        if c == cols:
            c -= 1
        out[r, c] = 1

    return out

def getSumFix_img(fixsDir, DataSet='SALICON', size=None):

	DataSet = DataSet.upper()
	if size is None:
		if DataSet in shuff_size.keys():
			size = shuff_size[DataSet]
		else:
			size = shuff_size["default"]

	fix_names = [f for f in os.listdir(fixsDir) if f.endswith('.mat')]
	fix_names.sort()

	ShufMap = np.zeros(size)
	for idx_n in range(len(fix_names)):

		fixpts = h5io.loadmat(fixsDir + fix_names[idx_n])["I"] > 0.5
		if fixpts.shape != size:
			# fixpts = cv2.resize(fixpts, (size[1], size[0]), interpolation=cv2.INTER_NEAREST)
			fixpts = resize_fixation(fixpts,size[0],size[1])

		ShufMap += fixpts

	return ShufMap

def getSumFix_vid(fixsDir, DataSet='DIEM20', size=None, maxframes = float('inf')):

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
			fixpts = np.array([resize_fixation(fixpts[:, :, 0, i], size[0], size[1]) for i in range(useframes)]).transpose((1, 2, 0))
			# fixpts = np.array([cv2.resize(fixpts[:, :, 0, i], (size[1], size[0]),interpolation=cv2.INTER_NEAREST) for i in range(useframes)]).transpose((1, 2, 0))
			fixpts = np.expand_dims(fixpts,axis=2)

		ShufMap += np.sum(fixpts[:,:,0,:],axis=2)
		ShufMap = np.round(ShufMap)

	return ShufMap

def getALLFix_img(fixsDir, DataSet='SALICON'):

	fix_names = [f for f in os.listdir(fixsDir) if f.endswith('.mat')]
	fix_names.sort()
	fix_num = len(fix_names)

	ALLFixPts = []
	# ALLFixMaps = []
	for idx_n in range(fix_num):

		fixpts = h5io.loadmat(fixsDir + fix_names[idx_n])["I"] > 0.5
		# ALLFixMaps.append(fixpts)

		fx, fy = np.where(fixpts)
		fx = fx / fixpts.shape[0]
		fy = fy / fixpts.shape[1]
		f_xy = np.concatenate((np.expand_dims(fx, 1), np.expand_dims(fy, 1)), 1)

		ALLFixPts.append(f_xy)

	return ALLFixPts#, ALLFixMaps

def getALLFix_vid(fixsDir, DataSet='DIEM20', maxframes = float('inf')):

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

			fx,fy = np.where(fixpts[:,:,0,idx_f])
			fx = fx / fixpts.shape[0]
			fy = fy / fixpts.shape[1]
			f_xy = np.concatenate((np.expand_dims(fx,1),np.expand_dims(fy,1)),1)

			ALLFixPts.append(f_xy)

	return ALLFixPts#, ALLFixMaps


def getshufmap(ALLFixPts, size=(480, 640), nframes = 10):
	if ALLFixPts is None or len(ALLFixPts)==0:
		return None

	nframes = min(nframes,len(ALLFixPts))
	shuf_idx = random.randint(0,len(ALLFixPts),int(nframes))

	fix_nf = ALLFixPts[shuf_idx[0]]
	for i in range(1,nframes):
		fix_f = ALLFixPts[shuf_idx[i]]
		fix_nf = np.concatenate((fix_nf,fix_f),0)

	fix_nf[:, 0] *= size[0]
	fix_nf[:, 1] *= size[1]

	fix_nf = np.round(fix_nf).astype(np.int)

	shufmap = np.zeros(size,dtype=np.uint8)
	shufmap[fix_nf[:, 0], fix_nf[:, 1]] = 1

	# shufmap = np.zeros(size, dtype=np.uint8)
	# for i in range(fix_nf.shape[0]):
	# 	shufmap[fix_nf[i, 0], fix_nf[i, 1]] = 1

	return shufmap


def getSimVal(keys_order, salmap1, salmap2, fixation_map=None, othermap=None):
	values = []
	for metric in keys_order:

		func = metrics[metric][0]
		compType = metrics[metric][1]
		sim = metrics[metric][2]

		if compType == "fix":
			if sim and not type(None) in [type(fixation_map), type(othermap)]:
				m = func(salmap1, fixation_map, othermap)
			else:
				m = func(salmap1, fixation_map)
		else:
			m = func(salmap1, salmap2)
		values.append(m)
	return values

def evalscores_vid_sum(RootDir, SalDir, DataSet, MethodNames, keys_order=keys_order):

	mapsDir = RootDir + 'maps/'
	fixsDir = RootDir + 'fixations/maps/'

	salsDir = SalDir + 'Saliency/'
	scoreDir = SalDir + 'Scores_sum/'
	if not os.path.exists(scoreDir):
		os.makedirs(scoreDir)

	print('\nEvaluate Metrics: ' + str(keys_order))
	if 'AUC_shuffled' in keys_order:
		shuff_path = RootDir + 'Shuffle_' + DataSet.upper() + '.mat'
		# shuff_path = '../Tools/Shuffle_' + DataSet.upper() + '.mat'
		if not os.path.exists(shuff_path):
			shuffle_map = getSumFix_vid(fixsDir,DataSet)
			h5io.savemat(shuff_path, {'ShufMap': shuffle_map})
		else:
			shuffle_map = h5io.loadmat(shuff_path)["ShufMap"]
	else:
		shuffle_map = None

	for idx_m in range(len(MethodNames)):
		print("---" + str(idx_m + 1) + "/" + str(len(MethodNames)) + "---: " + MethodNames[idx_m])

		score_path = scoreDir + 'Score_' +MethodNames[idx_m] + '.mat'
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
			print(str(idx_n + 1) + "/" + str(len(sal_names)) + ": " + sal_names[idx_n])

			file_name = sal_names[idx_n][:-4]
			iscore_path = iscoreDir + 'Score_' + file_name + '.mat'
			if os.path.exists(iscore_path):
				iscores = h5io.loadmat(iscore_path)["iscore"]
				scores[file_name] = iscores
				continue

			salmap = np.rint(h5io.loadmat(salmap_dir + file_name + '.mat')["salmap"]).astype(np.uint8)
			fixmap = h5io.loadmat(mapsDir + file_name + '_fixMaps.mat')["fixMap"]
			fixpts = h5io.loadmat(fixsDir + file_name + '_fixPts.mat')["fixLoc"]

			if not shuffle_map is None and shuffle_map.shape != fixpts.shape[:2]:
				ishuffle_map = resize_fixation(shuffle_map, fixpts.shape[0], fixpts.shape[1])
				# ishuffle_map = cv2.resize(shuffle_map, (fixpts.shape[1], fixpts.shape[0]), interpolation=cv2.INTER_NEAREST)
			else:
				ishuffle_map = shuffle_map

			nframes = min(salmap.shape[3],min(fixpts.shape[3],fixmap.shape[3]))
			iscores = np.zeros((nframes, len(keys_order)))
			t1 = time.time()
			for idx_f in range(nframes):
				isalmap = salmap[:, :, 0, idx_f]/255.0
				ifixmap = fixmap[:, :, 0, idx_f]/255.0
				ifixpts = fixpts[:, :, 0, idx_f]
				if not np.any(isalmap) or not np.any(ifixmap) or not np.any(ifixpts):
					iscores[idx_f] = np.NaN
					print(str(idx_f + 1) + "/" + str(nframes) + ": failed!")
					continue

				values = getSimVal(keys_order, isalmap, ifixmap, ifixpts, ishuffle_map)
				iscores[idx_f] = values
				# print(str(idx_f + 1) + "/" + str(nframes) + ": finished!")

			t2 = time.time()-t1
			print('\t\t time: %.3f S'%(t2))

			scores[file_name] = iscores
			h5io.savemat(iscore_path, {'iscore': iscores})

		h5io.savemat(score_path, {'scores': scores})

def evalscores_vid(RootDir, SalDir, DataSet, MethodNames, keys_order=keys_order):

	mapsDir = RootDir + 'maps/'
	fixsDir = RootDir + 'fixations/maps/'

	salsDir = SalDir + 'Saliency/'
	scoreDir = SalDir + 'Scores/'
	if not os.path.exists(scoreDir):
		os.makedirs(scoreDir)

	print('\nEvaluate Metrics: ' + str(keys_order))
	if 'AUC_shuffled' in keys_order:
		ALLFixPts_path = RootDir + 'ALLFixPts_' + DataSet.upper() + '.npy'
		# ALLFixPts_path = '../Tools/ALLFixPts_' + DataSet.upper() + '.npy'
		if not os.path.exists(ALLFixPts_path):
			ALLFixPts = getALLFix_vid(fixsDir,DataSet)
			np.save(ALLFixPts_path, ALLFixPts)
		else:
			ALLFixPts = np.load(ALLFixPts_path,allow_pickle=True)
	else:
		ALLFixPts = None

	for idx_m in range(len(MethodNames)):
		print("---" + str(idx_m + 1) + "/" + str(len(MethodNames)) + "---: " + MethodNames[idx_m])

		score_path = scoreDir + 'Score_' +MethodNames[idx_m] + '.mat'
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
			print(str(idx_n + 1) + "/" + str(len(sal_names)) + ": " + sal_names[idx_n])

			file_name = sal_names[idx_n][:-4]
			iscore_path = iscoreDir + 'Score_' + file_name + '.mat'
			if os.path.exists(iscore_path):
				iscores = h5io.loadmat(iscore_path)["iscore"]
				scores[file_name] = iscores
				continue

			salmap = np.rint(h5io.loadmat(salmap_dir + file_name + '.mat')["salmap"]).astype(np.uint8)
			fixmap = h5io.loadmat(mapsDir + file_name + '_fixMaps.mat')["fixMap"]
			fixpts = h5io.loadmat(fixsDir + file_name + '_fixPts.mat')["fixLoc"]

			nframes = min(salmap.shape[3],min(fixpts.shape[3],fixmap.shape[3]))
			iscores = np.zeros((nframes, len(keys_order)))
			t1 = time.time()
			for idx_f in range(nframes):
				isalmap = salmap[:, :, 0, idx_f]/255.0
				ifixmap = fixmap[:, :, 0, idx_f]/255.0
				ifixpts = fixpts[:, :, 0, idx_f]
				ishuffle_map = getshufmap(ALLFixPts, size=ifixpts.shape)
				if not np.any(isalmap) or not np.any(ifixmap) or not np.any(ifixpts):
					iscores[idx_f] = np.NaN
					print(str(idx_f + 1) + "/" + str(nframes) + ": failed!")
					continue

				values = getSimVal(keys_order, isalmap, ifixmap, ifixpts, ishuffle_map)
				iscores[idx_f] = values
				print(str(idx_f + 1) + "/" + str(nframes) + ": finished!")

			t2 = time.time()-t1
			print('\t\t time: %.3f S'%(t2))

			scores[file_name] = iscores
			h5io.savemat(iscore_path, {'iscore': iscores})

		h5io.savemat(score_path, {'scores': scores})

def evalscores_img_sum(DataDir, ResDir, DataSet, MethodNames, keys_order=keys_order):

	mapsDir = DataDir + 'maps/'
	fixsDir = DataDir + 'fixations/maps/'

	salsDir = ResDir + 'Saliency/'
	scoreDir = ResDir + 'Scores_sum/'
	if not os.path.exists(scoreDir):
		os.makedirs(scoreDir)

	print('\nEvaluate Metrics: ' + str(keys_order))
	if 'AUC_shuffled' in keys_order:
		shuff_path = DataDir + 'Shuffle_' + DataSet.upper() + '.mat'
		# shuff_path = '../Tools/Shuffle_' + DataSet.upper() + '.mat'
		if not os.path.exists(shuff_path):
			shuffle_map = getSumFix_img(fixsDir,DataSet)
			h5io.savemat(shuff_path, {'ShufMap': shuffle_map})
		else:
			shuffle_map = h5io.loadmat(shuff_path)["ShufMap"]
	else:
		shuffle_map = None

	for idx_m in range(len(MethodNames)):
		print("---" + str(idx_m + 1) + "/" + str(len(MethodNames)) + "---: " + MethodNames[idx_m])

		score_path = scoreDir + 'Score_' +MethodNames[idx_m] + '.mat'
		if os.path.exists(score_path):
			continue

		salmap_dir = salsDir + MethodNames[idx_m] + '/'
		sal_names = [f for f in os.listdir(salmap_dir) if f.endswith('.png')]
		sal_names.sort()

		scores = np.zeros((len(sal_names),len(keys_order)))
		for idx_n in range(len(sal_names)):
			file_name = sal_names[idx_n]

			salmap = cv2.imread(salmap_dir + file_name,-1)/255.0
			fixmap = cv2.imread(mapsDir + file_name,-1)/255.0
			fixpts = h5io.loadmat(fixsDir + file_name[:-4] + '.mat')["I"]

			if not shuffle_map is None and shuffle_map.shape != fixpts.shape[:2]:
				# ishuffle_map = cv2.resize(shuffle_map, (fixpts.shape[1], fixpts.shape[0]),interpolation=cv2.INTER_NEAREST)
				ishuffle_map = resize_fixation(shuffle_map, fixpts.shape[0], fixpts.shape[1])

			else:
				ishuffle_map = shuffle_map

			if not np.any(salmap) or not np.any(fixmap) or not np.any(fixpts):
				scores[idx_n] = np.NaN
				print(str(idx_n) + "/" + str(len(sal_names)) + ": failed!")
				continue

			values = getSimVal(keys_order, salmap, fixmap, fixpts, ishuffle_map)
			scores[idx_n] = values
			print(str(idx_n+1) + "/" + str(len(sal_names)) + ": finished!")

		h5io.savemat(score_path, {'scores': scores})

def evalscores_img(DataDir, ResDir, DataSet, MethodNames, keys_order=keys_order):

	mapsDir = DataDir + 'maps/'
	fixsDir = DataDir + 'fixations/maps/'

	salsDir = ResDir + 'Saliency/'
	scoreDir = ResDir + 'Scores/'
	if not os.path.exists(scoreDir):
		os.makedirs(scoreDir)

	print('\nEvaluate Metrics: ' + str(keys_order))
	if 'AUC_shuffled' in keys_order:
		ALLFixPts_path = DataDir + 'ALLFixPts_' + DataSet.upper() + '.npy'
		# ALLFixPts_path = '../Tools/ALLFixPts_' + DataSet.upper() + '.npy'
		if not os.path.exists(ALLFixPts_path):
			ALLFixPts = getALLFix_img(fixsDir,DataSet)
			np.save(ALLFixPts_path, ALLFixPts)
		else:
			ALLFixPts = np.load(ALLFixPts_path,allow_pickle=True)
	else:
		ALLFixPts = None

	for idx_m in range(len(MethodNames)):
		print("---" + str(idx_m + 1) + "/" + str(len(MethodNames)) + "---: " + MethodNames[idx_m])

		score_path = scoreDir + 'Score_' +MethodNames[idx_m] + '.mat'
		if os.path.exists(score_path):
			continue

		salmap_dir = salsDir + MethodNames[idx_m] + '/'
		sal_names = [f for f in os.listdir(salmap_dir) if f.endswith('.png')]
		sal_names.sort()

		scores = np.zeros((len(sal_names),len(keys_order)))
		for idx_n in range(len(sal_names)):
			file_name = sal_names[idx_n]

			salmap = cv2.imread(salmap_dir + file_name,-1)/255.0
			fixmap = cv2.imread(mapsDir + file_name,-1)/255.0
			fixpts = h5io.loadmat(fixsDir + file_name[:-4] + '.mat')["I"]
			shuffle_map = getshufmap(ALLFixPts, size=fixpts.shape)

			if not np.any(salmap) or not np.any(fixmap) or not np.any(fixpts):
				scores[idx_n] = np.NaN
				print(str(idx_n) + "/" + str(len(sal_names)) + ": failed!")
				continue

			values = getSimVal(keys_order, salmap, fixmap, fixpts, shuffle_map)
			scores[idx_n] = values
			print(str(idx_n+1) + "/" + str(len(sal_names)) + ": finished!")

		h5io.savemat(score_path, {'scores': scores})


