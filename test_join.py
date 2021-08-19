import kimimaro
import imageio
import cv2
import numpy as np

from tqdm import tqdm
import time
from pathlib import Path
import pickle

def o_time(s_time):
    print('*** Time: ', time.time() - s_time)
s_time = time.time()


root_dir = Path('/dc1/SCN/wafer14/seg/')
stack_name_list = ['1601', '1701']
# '1801', '1901', '2001', '2101', '2201', '2301', '2302']
anis = (10, 10, 40)
dust = 10000
ovlp = 200
resz = 4
core = 24

big_fields = np.zeros((209, 1050, 550), dtype=np.int32)
kimimaro.postprocess
for stack_name in stack_name_list:
    print('--', stack_name)
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
    print(labels.shape, labels.dtype)
    # imageio.volwrite(root_dir.joinpath('test_labels.tif'), labels, bigtiff=True)
    labels = np.transpose(labels, (2, 1, 0))
    o_time(s_time)

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
        progress=True,  # default False, show progress bar
        parallel=core,  # <= 0 all cpu, 1 single process, 2+ multiprocess
        parallel_chunk_size=600,  # how many skeletons to process before updating progress bar
    )
    # pickle.dump(skels, open('skels.pkl', 'wb'), protocol=4)
    print('Extract skels number:', len(skels))

    edges = {}
    fields = np.zeros(labels.shape, dtype=np.int32)
    for sk in skels.keys():
        svs = skels[sk].vertices
        for sv in range(svs.shape[0]):
            sn = svs[sv]
            fields[int(sn[0]/(anis[0]*resz)), int(sn[1]/(anis[1]*resz)), int(sn[2]/anis[2])] = sk
        ses = skels[sk].edges
        edges[sk] = {}
        for se in range(ses.shape[0]):
            f = ses[se, 0]
            t = ses[se, 1]
            if f in edges[sk]:
                edges[sk][f].append(t)
            else:
                edges[sk][f] = [t]
            if t in edges[sk]:
                edges[sk][t].append(f)
            else:
                edges[sk][t] = [f]
    fields_s = np.transpose(fields, (2, 1, 0))
    big_fields[:, (stack_y-16)*500:(stack_y-16)*500+550, :] += fields_s

imageio.volwrite(root_dir.joinpath('test_fields.tif'), big_fields, bigtiff=True)    
o_time(s_time)