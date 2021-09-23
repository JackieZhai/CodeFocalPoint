'''
t1: This gonna process whole skeletons of wafer 14.
'''

import kimimaro
from cloudvolume import PrecomputedSkeleton
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


root_dir = Path('/dc1/SCN/wafer14/')
seg_dir = root_dir.joinpath('seg/')
map_dir = root_dir.joinpath('merge.dat')


anis = (10, 10, 40)
cuts = (2000, 2000, 209)
resz = 4
core = 24
dust = 10000
jora = 1600

arr_msg, arr_read = read_binary_dat(map_dir)
arr_fls = read_file_system(seg_dir)
print('Stack property:', arr_msg)

big_skels = {}  # all skels across stacks
stack_sk = {}  # find all skels in each stack

for remap_z in range(arr_read.shape[0]):
    for remap_y in range(arr_read.shape[1]):
        for remap_x in range(arr_read.shape[2]):
            stack_name = arr_fls[remap_z][remap_y][remap_x]
            print(stack_name)
            assert len(stack_name) == 4

            img_stack = []
            for img_dir in sorted(seg_dir.joinpath(stack_name).glob('*.tif')):
                img_original = imageio.imread(img_dir).astype(np.int32)
                sz = img_original.shape
                img = cv2.resize(img_original, (sz[1]//resz, sz[0]//resz), interpolation=cv2.INTER_NEAREST)
                img_stack.append(img[np.newaxis, :, :])  # z,y,x
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

            stack_pad = [remap_x * cuts[0] * anis[0], \
                remap_y * cuts[1] * anis[1], \
                remap_z * cuts[2] * anis[2]]
            for sk in skels.keys():
                skels[sk].vertices = skels[sk].vertices + stack_pad
                skelmap = arr_read[remap_z, remap_y, remap_x]
                skels[sk].id = skelmap[sk]
                if skelmap[sk] in big_skels.keys():
                    big_skels[skelmap[sk]].append(deepcopy(skels[sk]))
                else:
                    big_skels[skelmap[sk]] = [deepcopy(skels[sk])]
                if stack_name in stack_sk.keys():
                    stack_sk[stack_name].append(skelmap[sk])
                else:
                    stack_sk[stack_name] = [skelmap[sk]]


def _merge_operation(sk, big_skels, lock):
    if len(big_skels[sk]) == 1:
        lock.acquire()
        big_skels[sk] = big_skels[sk][0]
        lock.release()
    elif len(big_skels[sk]) > 1:
        merge_skel = kimimaro.join_close_components(big_skels[sk], radius=jora)
        merge_skel = kimimaro.postprocess(merge_skel, dust_threshold=0, tick_threshold=0)
        lock.acquire()
        big_skels[sk] = merge_skel
        lock.release()
    else:
        raise Exception

big_skels = Manager().dict(big_skels)
mo_lock = Manager().Lock()
mo_partial = partial(_merge_operation, big_skels=big_skels, lock=mo_lock)
with Pool(processes=core) as pool:
    pool.map(mo_partial, big_skels.keys())
print('Extract \'all\' skels number:', len(big_skels))

big_edges = {}  # all edges across skels
for sk in tqdm(big_skels.keys()):
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
print('Extracted \'all\' edges.')

pickle.dump(big_skels, open('big_skels.pkl', 'wb'), protocol=4)
pickle.dump(stack_sk, open('stack_sk.pkl', 'wb'), protocol=4)
pickle.dump(big_edges, open('big_edges.pkl', 'wb'), protocol=4)
o_time(s_time)
