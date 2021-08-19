import kimimaro
import imageio
import cv2
import numpy as np

from tqdm import tqdm
import time
from pathlib import Path
import pickle

from utils.load_remap import read_binary_dat, read_file_system

def o_time(s_time):
    print('*** Time: ', time.time() - s_time)
s_time = time.time()


root_dir = Path('/dc1/SCN/wafer14/seg/')
stack_name_list = ['1601', '1701', '1801', '1901', '2001', '2101', '2201', '2301', '2302']
anis = (10, 10, 40)
dust = 10000
ovlp = 200
resz = 4
core = 24
jora = 1600
plnm = 4

arr_msg, arr_read = read_binary_dat('/dc1/SCN/wafer14/merge.dat')
print(arr_msg)
arr = [None for i in range(30)]
arr[16] = arr_read[0, 13, 0]
arr[17] = arr_read[0, 14, 0]
arr[18] = arr_read[0, 15, 0]
arr[19] = arr_read[0, 16, 0]
arr[20] = arr_read[0, 17, 0]
arr[21] = arr_read[0, 18, 0]
arr[22] = arr_read[0, 19, 0]
arr[23] = arr_read[0, 20, 0]
arr[24] = arr_read[0, 20, 1]  # 2302

big_skels = {}
for stack_name in tqdm(stack_name_list):
    # print('--', stack_name)
    assert len(stack_name) == 4
    stack_y = int(stack_name[0:2])
    stack_x = int(stack_name[2:4])

    img_original_stack = []
    img_stack = []
    for img_dir in sorted(root_dir.joinpath(stack_name).glob('*.tif')):
        img_original = imageio.imread(img_dir).astype(np.int32)
        sz = img_original.shape
        img = cv2.resize(img_original, (sz[1]//resz, sz[0]//resz), interpolation=cv2.INTER_NEAREST)
        img_original_stack.append(img_original[np.newaxis, :, :])  # z,y,x
        img_stack.append(img[np.newaxis, :, :])  # z,y,x
    labels_original = np.vstack(img_original_stack)
    labels = np.vstack(img_stack)
    # print(labels.shape, labels.dtype)
    # imageio.volwrite(root_dir.joinpath('test_labels.tif'), labels, bigtiff=True)
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
        # object_ids=[10434, 10598, 10782, 10810, 11323, 14653, 14695, 14972, 15283, \
        #     16532, 2268, 397, 6003, 6500, 6725, 9826, \
        #     15299, 8602, 15115], # this line are external labels to be tested
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
    # pickle.dump(skels, open('skels.pkl', 'wb'), protocol=4)
    print('Extract skels number:', len(skels))

    for sk in skels.keys():
        skels[sk].vertices = skels[sk].vertices + [(stack_x-1) * 500 *(anis[0]*resz), \
            (stack_y-16) * 500 * (anis[1]*resz), 0 * (anis[2])]
        skelmap = arr[stack_y + (stack_x-1)]
        skels[sk].id = skelmap[sk]
        if skelmap[sk] in big_skels.keys():
            big_skels[skelmap[sk]].append(skels[sk])
        else:
            big_skels[skelmap[sk]] = [skels[sk]]

for sk in big_skels.keys():
    if len(big_skels[sk]) == 1:
        big_skels[sk] = big_skels[sk][0]
    elif len(big_skels[sk]) > 1:
        merge_skel = kimimaro.join_close_components(big_skels[sk], radius=jora)
        merge_skel = kimimaro.postprocess(merge_skel, dust_threshold=0, tick_threshold=0)
        big_skels[sk] = merge_skel
    else:
        raise Exception
print('Extract all skels number:', len(big_skels))

big_fields = np.zeros((1050, 4050, 209), dtype=np.int32)
big_edges = {}
for sk in big_skels.keys():
    svs = big_skels[sk].vertices
    for sv in range(svs.shape[0]):
        sn = svs[sv]
        big_fields[int(sn[0]/(anis[0]*resz)), int(sn[1]/(anis[1]*resz)), int(sn[2]/anis[2])] = sk
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

imageio.volwrite(root_dir.joinpath('test_fields.tif'), np.transpose(big_fields, (2, 1, 0)), bigtiff=True)    
o_time(s_time)