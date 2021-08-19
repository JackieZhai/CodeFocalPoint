import kimimaro
import imageio
import cv2
import numpy as np
from h5py import File

from tqdm import tqdm
import time
from pathlib import Path
import pickle
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
plnm = 2

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
        parallel=core//plnm,  # <= 0 all cpu, 1 single process, 2+ multiprocess
        parallel_chunk_size=600//plnm,  # how many skeletons to process before updating progress bar
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
m_time = time.time()

big_fields = np.zeros((550, 2050, 209), dtype=np.int32)
big_edges = {}  # all edges across skels
for sk in big_skels.keys():
    svs = big_skels[sk].vertices
    for sv in range(svs.shape[0]):
        sn = svs[sv]
        psn = [int(sn[0]/(anis[0]*resz)), int(sn[1]/(anis[1]*resz)), int(sn[2]/anis[2])]
        big_fields[psn[0], psn[1], psn[2]] = sk
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
        # if abs(svs[f][0]-svs[t][0])>40 or abs(svs[f][1]-svs[t][1])>40 or abs(svs[f][2]-svs[t][2])>40:
        #     print(svs[f], svs[t])

big_fields_aff = np.zeros((550, 2050, 209), dtype=np.uint8)
for stack_name in tqdm(stack_name_list):
    stack_y = int(stack_name[0:2])
    stack_x = int(stack_name[2:4])
    aff_name = 'elastic_' + stack_name + '.h5'
    affs = File(aff_dir.joinpath(aff_name), 'r')['vol0'][:]
    affs = np.transpose(affs, (0, 3, 2, 1))
    stack_pad = [(stack_x-1) * 500 * (anis[0]*resz), \
        (stack_y-16) * 500 * (anis[1]*resz), \
        0 * (anis[2])]
    
    for sk in stack_sk[stack_name]:
        svs = big_skels[sk].vertices
        edg = big_edges[sk]
        for sv in range(svs.shape[0]):
            sn = svs[sv] - stack_pad
            psn = [int(sn[0]/(anis[0]*resz)), int(sn[1]/(anis[1]*resz)), int(sn[2]/anis[2])]
            all_aff_sv = 0
            for tv in edg[sv]:
                tn = svs[tv] - stack_pad
                ptn = [int(tn[0]/(anis[0]*resz)), int(tn[1]/(anis[1]*resz)), int(tn[2]/anis[2])]
                aff_sv = affinity_in_skel(affs, psn, ptn, resz=resz)
                all_aff_sv += aff_sv
            sna = svs[sv]
            psna = [int(sna[0]/(anis[0]*resz)), int(sna[1]/(anis[1]*resz)), int(sna[2]/anis[2])]
            big_fields_aff[psna[0], psna[1], psna[2]] = all_aff_sv / len(edg[sv])

for bin in range(1, 255, 10):
    print(bin, bin+10)
    print(np.sum(np.logical_and(big_fields_aff >= bin, big_fields_aff < bin+10)))
print(np.sum(big_fields_aff >=1))

imageio.volwrite(root_dir.joinpath('test_fields.tif'), np.transpose(big_fields, (2, 1, 0)), bigtiff=True)  
imageio.volwrite(root_dir.joinpath('test_fields_aff.tif'), np.transpose(big_fields_aff, (2, 1, 0)), bigtiff=True)    
o_time(s_time)
