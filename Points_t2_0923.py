'''
t2: This gonna process whole 4 different types at same time from configs.
'''
import numpy as np
from h5py import File
from tqdm import tqdm
import time
import pickle
from multiprocessing import Manager
from os.path import dirname, abspath, join

from scripts.skel import process_skeletonize, process_skel_merging, skels_to_edges
from scripts.amputate import process_amputate_conversion
from scripts.split import skels_to_endpoints, process_endpoint_split_checking
from scripts.divide import process_branch_checking, process_strange_checking, process_branstran_post
from scripts.merge import process_affinity_merge_check, _components, process_merge_post
from configs.config import cfg
from utils.load_remap import read_binary_dat, read_file_system

def o_time(s_time, note=None):
    if note is not None:
        print('*** Time of \'{:s}\': {:.2f} min'.format(note, (time.time() - s_time) / 60))
    else:
        print('*** Time: {:.2f} min'.format((time.time() - s_time) / 60))
s_time = time.time()


data_dir = '/dc1/SCN/wafer14/'
seg_dir = join(data_dir, 'seg_after')
aff_dir = join(data_dir, 'affnity')
map_dir = join(data_dir, 'merge.dat')
amp_dir = None  # define below, may change

root_dir = dirname(abspath(__file__))
cfg.merge_from_file(join(root_dir, 'configs/t2.yaml'))
cfg.OMIN = [0 * cfg.OANIS[0], 8000 * cfg.OANIS[1], 0 * cfg.OANIS[2]]
cfg.OMAX = [(72000+400) * cfg.OANIS[0], (136000+400) * cfg.OANIS[1], 209 * cfg.OANIS[2]]
# TODO: OMIN / OMAX should not allow

arr_msg, arr_read = read_binary_dat(map_dir)
arr_fls = read_file_system(seg_dir)
print('Stack property:', arr_msg)

big_skels = {}  # all skels across stacks
stack_sk = {}  # find all skels in each stack
amputate_dic = {}

for remap_z in range(arr_read.shape[0]):
    for remap_y in tqdm(range(arr_read.shape[1])):
        for remap_x in range(arr_read.shape[2]):
            stack_name = arr_fls[remap_z][remap_y][remap_x]
            if len(stack_name) != 4:
                continue

            imgs_dir = join(seg_dir, stack_name, 'seg.h5')
            labels = File(imgs_dir, 'r')['data'][:]
            labels = labels[::cfg.RESZ, ::cfg.RESZ, :]

            stack_pad = [remap_x * cfg.CUTS[0] * cfg.ANIS[0], \
                remap_y * cfg.CUTS[1] * cfg.ANIS[1], \
                remap_z * cfg.CUTS[2] * cfg.ANIS[2]]
            skelmap = arr_read[remap_z, remap_y, remap_x]

            # Skeleton
            process_skeletonize(big_skels, stack_sk, stack_name, labels, stack_pad, skelmap, cfg)            
            
            # Amputate
            amp_dir = join(seg_dir, stack_name, 'focus.pkl')
            amputate_mat = pickle.load(open(amp_dir, 'rb'))
            process_amputate_conversion(amputate_dic, amputate_mat, skelmap, cfg)

process_skel_merging(big_skels, cfg)
print('Extract \'all\' skels number:', len(big_skels))

pickle.dump(big_skels, open(join(root_dir, 'results/big_skels.pkl'), 'wb'), protocol=4)
pickle.dump(stack_sk, open(join(root_dir, 'results/stack_sk.pkl'), 'wb'), protocol=4)
pickle.dump(amputate_dic, open(join(root_dir, 'results/amputate_error_2p.pkl'), 'wb'), protocol=4)
print('--- Amputate error number:', len(amputate_dic.keys()))
o_time(s_time, 'skeletonizing & amputate error finding')

big_edges = {}  # all edges across skels
skels_to_edges(big_skels, big_edges)
print('Extracted all edges.')

endpoints = {}  # endpoint and its backtracking set
endpoints_vector = {}
skels_to_endpoints(endpoints, endpoints_vector, big_skels, big_edges, cfg)
print('Extracted all endpoints and corresponding vectors.')

# Split
touch_split_dic = {}
touch_split_num = 0
process_endpoint_split_checking(touch_split_dic, touch_split_num, big_skels, endpoints, endpoints_vector, cfg)

pickle.dump(touch_split_dic, open(join(root_dir, 'results/split_error_2p.pkl'), 'wb'), protocol=4)
print('--- Split error number:', touch_split_num)
o_time(s_time, 'split error finding')

# Divide
branch_dic = {}
branch_num = 0
process_branch_checking(branch_dic, branch_num, big_skels, big_edges, cfg)
o_time(s_time, 'branch errors finding')

strange_dic = Manager().dict()
ps_lock = Manager().Lock()
strange_num = 0
process_strange_checking(strange_dic, strange_num, ps_lock, big_skels, cfg)
o_time(s_time, 'strange errors finding')

# combine branch/strange as divide errors
branstran_dic = {}
branstran_num = 0
process_branstran_post(branstran_dic, branstran_num, branch_dic, strange_dic, big_skels, cfg)

pickle.dump(branstran_dic, open(join(root_dir, 'results/divide_error_1p.pkl'), 'wb'), protocol=4)
print('--- Branch error number:', branch_num)
print('--- Strange error number:', strange_num)
print('--- Divide error number:', branstran_num)
o_time(s_time, 'post processing divide errors')

# Merge
big_merge_error = None
big_merge_error_sk = None  # corresponding skel number
big_merge_error_af = None  # corresponding affinity
for remap_z in range(arr_read.shape[0]):
    for remap_y in tqdm(range(arr_read.shape[1])):
        for remap_x in range(arr_read.shape[2]):
            stack_name = arr_fls[remap_z][remap_y][remap_x]
            if len(stack_name) != 4:
                continue
            aff_name = 'elastic_' + stack_name + '.h5'
            affs = File(join(aff_dir, aff_name), 'r')['vol0'][:]
            affs = np.transpose(affs, (0, 3, 2, 1))
            stack_pad = [remap_x * cfg.CUTS[0] * cfg.ANIS[0], \
                        remap_y * cfg.CUTS[1] * cfg.ANIS[1], \
                        remap_z * cfg.CUTS[2] * cfg.ANIS[2]]
            
            process_affinity_merge_check(big_merge_error, big_merge_error_sk, big_merge_error_af, affs, \
                big_skels, big_edges, stack_sk[stack_name], stack_pad, cfg)     

print('Maped all affinities.')
del big_edges

big_merge_error_af = np.array(big_merge_error_af, dtype=np.float32)
big_merge_error_sk = np.array(big_merge_error_sk, dtype=np.uint32)

pmes, pafs, psks = _components(big_merge_error, big_merge_error_af, big_merge_error_sk)
touch_merge_dic = {}
process_merge_post(touch_merge_dic, pmes, pafs, psks, cfg)

pickle.dump(touch_merge_dic, open(join(root_dir, 'results/merge_error_1p.pkl'), 'wb'), protocol=4)
print('--- Merge error number:', len(touch_merge_dic.keys()))    
o_time(s_time, 'merge errors finding')
