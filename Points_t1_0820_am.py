'''
t1: This gonna process whole skeletons of 'preprocessed' wafer 14 (from ZhangYC).
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
seg_dir = root_dir.joinpath('seg_after/')
map_dir = root_dir.joinpath('merge.dat')


anis = (20, 20, 40)
cuts = (1000, 1000, 209)
resz = 2

core = 24
chuk = 2400
dust = 342  # edge length of the cubic physical volume
jora = 1600

# obias, omin, omax should be calcuated before
oanis = (5, 5, 40)
obias = (0, 8000, 0)  # should start from 01, but 03
omin = (0 * oanis[0], 8000 * oanis[1], 0 * oanis[2])
omax = ((72000+400) * oanis[0], (136000+400) * oanis[1], 209 * oanis[2])

arr_msg, arr_read = read_binary_dat(map_dir)
arr_fls = read_file_system(seg_dir)
print('Stack property:', arr_msg)

amputate_dic = {}

for remap_z in range(arr_read.shape[0]):
    for remap_y in tqdm(range(arr_read.shape[1])):
        for remap_x in range(arr_read.shape[2]):
            stack_name = arr_fls[remap_z][remap_y][remap_x]
            if len(stack_name) != 4:
                continue

            stack_pad = [remap_x * cuts[0] * anis[0], \
                remap_y * cuts[1] * anis[1], \
                remap_z * cuts[2] * anis[2]]
            skelmap = arr_read[remap_z, remap_y, remap_x]
            
            amp_dir = seg_dir.joinpath(stack_name).joinpath('focus.pkl')
            amputate_mat = pickle.load(open(amp_dir, 'rb'))
            for label_pair in amputate_mat.keys():
                amp_list = amputate_mat[label_pair]
                ans_list = []
                for amp in amp_list:
                    ans_dic = {}
                    ans_dic['pos'] = (np.array(amp[4])*anis/oanis+obias).astype(np.uint32).tolist()
                    ans_dic['min'] = (np.array(amp[1])*anis/oanis+obias).astype(np.uint32).tolist()
                    ans_dic['max'] = (np.array(amp[0])*anis/oanis+obias).astype(np.uint32).tolist()
                    ans_dic['sample1'] = (np.array(amp[2])*anis/oanis+obias).astype(np.uint32).tolist()
                    ans_dic['sample2'] = (np.array(amp[3])*anis/oanis+obias).astype(np.uint32).tolist()
                    ans_dic['score'] = amp[5]
                    ans_list.append(ans_dic)
                label1 = skelmap[label_pair[0]]
                label2 = skelmap[label_pair[1]]
                amputate_dic[(label1, label2)] = ans_list


pickle.dump(amputate_dic, open('amputate_error_2p.pkl', 'wb'), protocol=4)
