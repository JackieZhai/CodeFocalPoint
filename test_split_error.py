from cloudvolume.skeleton import Skeleton, fastremap
import kimimaro
import imageio
import cv2
import numpy as np
from h5py import File
from sklearn.neighbors import NearestNeighbors

from tqdm import tqdm
import time
from pathlib import Path
import pickle
import math
from multiprocessing import Pool, Manager
from copy import deepcopy
from functools import partial

from utils.load_remap import read_binary_dat, read_file_system
from utils.compute_affinity import affinity_in_skel

def o_time(s_time, note=None):
    if note is not None:
        print('*** Time of \'{:s}\': {:.2f} min'.format(note, (time.time() - s_time) / 60))
    else:
        print('*** Time: {:.2f} min'.format((time.time() - s_time) / 60))
s_time = time.time()
m_time = time.time()

root_dir = Path('/dc1/SCN/wafer14/')
seg_dir = root_dir.joinpath('seg/')
aff_dir = root_dir.joinpath('affnity/')
map_dir = root_dir.joinpath('merge.dat')


stack_name_list = ['1601', '1701', '1801', '1901']
anis = (10, 10, 40)
dust = 10000
ovlp = 200
resz = 4
core = 24
jora = 1600
spld = 800
spln = 3
spve = 16
spra = 0.45  # 0.3216

arr_msg, arr_read = read_binary_dat('/dc1/SCN/wafer14/merge.dat')
print(arr_msg)
arr = [None for i in range(30)]
arr[16] = arr_read[0, 13, 0]
arr[17] = arr_read[0, 14, 0]
arr[18] = arr_read[0, 15, 0]
arr[19] = arr_read[0, 16, 0]

big_skels = {}  # all skels across stacks
stack_sk = {}  # find all skels in each stack
for stack_name in tqdm(stack_name_list):
    # print('--- Stack:', stack_name)
    assert len(stack_name) == 4
    stack_y = int(stack_name[0:2])
    stack_x = int(stack_name[2:4])

    img_original_stack = []
    img_stack = []
    for img_dir in sorted(seg_dir.joinpath(stack_name).glob('*.tif')):
        img_original = imageio.imread(img_dir).astype(np.int32)
        sz = img_original.shape
        img = cv2.resize(img_original, (sz[1]//resz, sz[0]//resz), interpolation=cv2.INTER_NEAREST)
        img_original_stack.append(img_original[np.newaxis, :, :])  # z,y,x
        img_stack.append(img[np.newaxis, :, :])  # z,y,x
    labels_original = np.vstack(img_original_stack)
    labels = np.vstack(img_stack)
    labels = np.transpose(labels, (2, 1, 0))

    skels = kimimaro.skeletonize(
        labels,
        teasar_params={
        'scale': 2,
        'const': 200, # physical radius
        'pdrf_exponent': 4,
        'pdrf_scale': 100000,
        'soma_detection_threshold': 1100, # physical radius
        'soma_acceptance_threshold': 3500, # physical radius
        'soma_invalidation_scale': 0.5,
        'soma_invalidation_const': 0, # physical radius
        'max_paths': None, # default None
        },
        # object_ids=[XXX, \
        #     XXX], # this line are external labels to be tested
            # process only the specified labels
        # extra_targets_before=[ (27,33,100), (44,45,46) ], # target points in voxels
        # extra_targets_after=[ (27,33,100), (44,45,46) ], # target points in voxels
        dust_threshold=int(dust/(resz**2)),  # skip connected components with fewer than this many voxels
        anisotropy=(anis[0]*resz, anis[1]*resz, anis[2]), # physical units
        fix_branching=True,  # default True
        fix_borders=True,  # default True
        fill_holes=False,  # default False
        fix_avocados=False,  # default False
        progress=False,  # default False, show progress bar
        parallel=core,  # <= 0 all cpu, 1 single process, 2+ multiprocess
        parallel_chunk_size=600,  # how many skeletons to process before updating progress bar
    )

    stack_pad = [(stack_x-1) * 500 * (anis[0]*resz), \
        (stack_y-16) * 500 * (anis[1]*resz), \
        0 * (anis[2])]
    for sk in skels.keys():
        skels[sk].vertices = skels[sk].vertices + stack_pad
        skelmap = arr[stack_y + (stack_x-1)]
        skels[sk].id = skelmap[sk]
        if skelmap[sk] in big_skels.keys():
            big_skels[skelmap[sk]].append(skels[sk])
        else:
            big_skels[skelmap[sk]] = [skels[sk]]
        if stack_name in stack_sk.keys():
            stack_sk[stack_name].append(skelmap[sk])
        else:
            stack_sk[stack_name] = [skelmap[sk]]

m_time = time.time()
for sk in big_skels.keys():
    if len(big_skels[sk]) == 1:
        big_skels[sk] = big_skels[sk][0]
    elif len(big_skels[sk]) > 1:
        merge_skel = kimimaro.join_close_components(big_skels[sk], radius=jora)
        merge_skel = kimimaro.postprocess(merge_skel, dust_threshold=0, tick_threshold=0)
        big_skels[sk] = merge_skel
    else:
        raise Exception
print('Extract \'all\' skels number:', len(big_skels))
o_time(m_time, 'merge skels')

big_edges = {}  # all edges across skels
for sk in big_skels.keys():
    ses = big_skels[sk].edges
    big_edges[sk] = {}
    for se in range(ses.shape[0]):
        f = ses[se, 0]
        t = ses[se, 1]
        if f in big_edges[sk]:
            big_edges[sk][f].append(t)
        else:
            big_edges[sk][f] = [t]
        if t in big_edges[sk]:
            big_edges[sk][t].append(f)
        else:
            big_edges[sk][t] = [f]

endpoints = {}
endpoints_vector = {}
for sk in tqdm(big_skels.keys()):
    endpoints[sk] = {}
    endpoints_vector[sk] = {}
    svs = big_skels[sk].vertices
    for sv in range(svs.shape[0]):
        if len(big_edges[sk][sv]) == 1:
            endpoints[sk][sv] = [sv]
            now_sv = sv
            for num in range(spve):
                next_sv = big_edges[sk][now_sv][0]
                next_sv_list = big_edges[sk][next_sv]
                next_sv_list.remove(now_sv)
                if len(next_sv_list) == 0:
                    break
                elif len(next_sv_list) > 1:
                    break
                else:
                    endpoints[sk][sv].append(next_sv)
                next_sv_list.append(now_sv)
                now_sv = next_sv
            if len(endpoints[sk][sv]) == 1:
                endpoints_vector[sk][sv] = [0.0, 0.0, 0.0]
            else:
                e = svs[sv]
                s = svs[endpoints[sk][sv][1]]
                for m in range(2, len(endpoints[sk][sv])):
                    s += svs[endpoints[sk][sv][m]]
                s /= len(endpoints[sk][sv]) - 1
                norm = math.sqrt((e[0] - s[0]) * (e[0] - s[0]) + (e[1] - s[1]) * (e[1] - s[1]) + (e[2] - s[2]) * (e[2] - s[2]))
                endpoints_vector[sk][sv] = [(e[0] - s[0]) / norm, (e[1] - s[1]) / norm, (e[2] - s[2]) / norm]

eps = []
eps_sk = []
eps_vec = []
for sk in big_skels.keys():
    svs = big_skels[sk].vertices
    ends = endpoints[sk]
    ends_ve = endpoints_vector[sk]
    for sv in ends.keys():
        eps.append(svs[sv])
        eps_sk.append(sk)
        eps_vec.append(ends_ve[sv])
print('Endpoints number:', len(eps))
eps = np.vstack(eps).astype(np.float32)
eps_sk = np.array(eps_sk, dtype=np.uint32)
eps_vec = np.array(eps_vec, dtype=np.float32)
eps_nbr = NearestNeighbors(n_neighbors=spln+1, algorithm='auto').fit(eps)
diss, inds = eps_nbr.kneighbors(eps)
print(eps.shape, eps_sk.shape, eps_vec.shape)
tmps = []
for e in range(eps.shape[0]):
    for en in range(1, spln+1):
        dis = diss[e, en]
        ind = inds[e, en]
        if dis > spld:
            continue
        v1 = eps_vec[e]
        v2 = eps_vec[ind]
        dot_product = v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]
        # if (dot_product == 0) or (-dot_product >= math.cos(spra)):
        if -dot_product >= math.cos(spra):
            tmps.append(eps[e])
            tmps.append(eps[ind])

# pickle.dump(touch_merge_dic, open('touch_merge_dic.pkl', 'wb'), protocol=4)

big_fields_aff = np.zeros((550, 2050, 209), dtype=np.uint8)
print('--- Split error number:', len(tmps))
for sv in range(len(tmps)):
    sna = tmps[sv]
    psna = [int(sna[0]/(anis[0]*resz)), int(sna[1]/(anis[1]*resz)), int(sna[2]/anis[2])]
    big_fields_aff[psna[0], psna[1], psna[2]] = 255

imageio.volwrite(root_dir.joinpath('test_fields_spl.tif'), np.transpose(big_fields_aff, (2, 1, 0)), bigtiff=True)    
o_time(s_time)
