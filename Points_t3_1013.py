'''
t3: the split error checking process need to be improved.
improved idea from YuanJB, see "split.py".
'''
import numpy as np
from h5py import File
from tqdm import tqdm
import time
import pickle
import argparse
from os.path import dirname, abspath, join
from cloudvolume.skeleton import Skeleton

from scripts.skel import process_skeletonize, process_skel_merging, skels_to_edges
from scripts.amputate import process_amputate_conversion
from scripts.split import skels_to_endpoints, process_endpoint_split_checking
from scripts.divide import process_branch_checking, process_strange_checking, process_branstran_post
from scripts.merge import _components, process_merge_post
from configs.config import cfg
from utils.load_remap import read_binary_dat, read_file_system
from utils.compute_affinity import affinity_in_skel

def o_time(s_time, note=None):
    if note is not None:
        print('*** Time of \'{:s}\': {:.2f} min'.format(note, (time.time() - s_time) / 60))
    else:
        print('*** Time: {:.2f} min'.format((time.time() - s_time) / 60))
s_time = time.time()

parser = argparse.ArgumentParser(description='FocalPoint_Points_t3')
parser.add_argument('--resume', action="store_true", help='whether to load pre-computed skel results')
args = parser.parse_args()

data_dir = '/dc1/SCN/wafer14/'
seg_dir = join(data_dir, 'seg_after')
aff_dir = join(data_dir, 'affnity')
map_dir = join(data_dir, 'merge.dat')
amp_dir = None  # define below, may change

root_dir = dirname(abspath(__file__))
cfg.merge_from_file(join(root_dir, 'configs/t3.yaml'))
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

            stack_pad = [remap_x * cfg.CUTS[0] * cfg.ANIS[0], \
                remap_y * cfg.CUTS[1] * cfg.ANIS[1], \
                remap_z * cfg.CUTS[2] * cfg.ANIS[2]]
            skelmap = arr_read[remap_z, remap_y, remap_x]

            # Skeleton
            if not args.resume:
                imgs_dir = join(seg_dir, stack_name, 'seg.h5')
                labels = File(imgs_dir, 'r')['data'][:]
                labels = labels[::cfg.RESZ, ::cfg.RESZ, :]
                process_skeletonize(big_skels, stack_sk, stack_name, labels, stack_pad, skelmap, cfg)            
            
            # # Amputate
            # amp_dir = join(seg_dir, stack_name, 'focus.pkl')
            # amputate_mat = pickle.load(open(amp_dir, 'rb'))
            # process_amputate_conversion(amputate_dic, amputate_mat, skelmap, cfg)

# pickle.dump(amputate_dic, open(join(root_dir, 'results/amputate_error_2p.pkl'), 'wb'), protocol=4)
# print('--- Amputate error number:', len(amputate_dic.keys()))
if not args.resume:
    process_skel_merging(big_skels, cfg)
    pickle.dump(big_skels, open(join(root_dir, 'results/big_skels.pkl'), 'wb'), protocol=4)
    pickle.dump(stack_sk, open(join(root_dir, 'results/stack_sk.pkl'), 'wb'), protocol=4)
    print('Extracted \'all\' skels number:', len(big_skels))
else:
    big_skels = pickle.load(open(join(root_dir, 'results/big_skels.pkl'), 'rb'))
    stack_sk = pickle.load(open(join(root_dir, 'results/stack_sk.pkl'), 'rb'))
    print('Loaded \'all\' skels number:', len(big_skels))
o_time(s_time, 'skeletonizing & amputate error finding')

# all edges across skels
big_edges = skels_to_edges(big_skels)
print('Extracted \'all\' edges (1st time).')

# endpoint and its backtracking set
if not args.resume:
    endpoints, endpoints_vector = skels_to_endpoints(big_skels, big_edges, cfg)
    pickle.dump(endpoints, open(join(root_dir, 'results/endpoints.pkl'), 'wb'), protocol=4)
    pickle.dump(endpoints_vector, open(join(root_dir, 'results/endpoints_vector.pkl'), 'wb'), protocol=4)
    print('Extracted \'all\' endpoints and corresponding vectors.')
else:
    endpoints = pickle.load(open(join(root_dir, 'results/endpoints.pkl'), 'rb'))
    endpoints_vector = pickle.load(open(join(root_dir, 'results/endpoints_vector.pkl'), 'rb'))
    print('Loaded \'all\' endpoints and corresponding vectors.')
del big_edges

# Split
touch_split_dic, touch_split_num = process_endpoint_split_checking(big_skels, endpoints, endpoints_vector, cfg)

pickle.dump(touch_split_dic, open(join(root_dir, 'results/split_error_2p.pkl'), 'wb'), protocol=4)
print('--- Split error number:', touch_split_num)
o_time(s_time, 'split error finding')

# # Divide
# big_edges = skels_to_edges(big_skels)
# print('Extracted \'all\' edges (2nd time).')
# branch_dic, branch_num = process_branch_checking(big_skels, big_edges, cfg)
# o_time(s_time, 'branch error finding')

# strange_dic, strange_num = process_strange_checking(big_skels, big_edges, cfg)
# o_time(s_time, 'strange error finding')
# del big_edges

# # combine branch/strange as divide errors
# branstran_dic, branstran_num = process_branstran_post(branch_dic, strange_dic, big_skels, cfg)

# pickle.dump(branstran_dic, open(join(root_dir, 'results/divide_error_1p.pkl'), 'wb'), protocol=4)
# print('--- Branch error number:', branch_num)
# print('--- Strange error number:', strange_num)
# print('--- Divide error number:', branstran_num)
# o_time(s_time, 'post processing divide error')

# # Merge
# big_edges = skels_to_edges(big_skels)
# print('Extracted \'all\' edges (3rd time).')
# big_merge_error = None
# big_merge_error_sk = None  # corresponding skel number
# big_merge_error_af = None  # corresponding affinity
# for remap_z in range(arr_read.shape[0]):
#     for remap_y in tqdm(range(arr_read.shape[1])):
#         for remap_x in range(arr_read.shape[2]):
#             stack_name = arr_fls[remap_z][remap_y][remap_x]
#             if len(stack_name) != 4:
#                 continue
#             aff_name = 'elastic_' + stack_name + '.h5'
#             affs = File(join(aff_dir, aff_name), 'r')['vol0'][:]
#             affs = np.transpose(affs, (0, 3, 2, 1))
#             stack_pad = [remap_x * cfg.CUTS[0] * cfg.ANIS[0], \
#                         remap_y * cfg.CUTS[1] * cfg.ANIS[1], \
#                         remap_z * cfg.CUTS[2] * cfg.ANIS[2]]
            
#             for sk in stack_sk[stack_name]:
#                 svs = big_skels[sk].vertices
#                 edg = big_edges[sk]
#                 mesv = []
#                 mesa = []  # corresponding affinity
#                 mese = []
#                 for sv in range(svs.shape[0]):
#                     sn = svs[sv] - stack_pad
#                     psn = [int(sn[0]/(cfg.ANIS[0]*cfg.RESZ)), int(sn[1]/(cfg.ANIS[1]*cfg.RESZ)), int(sn[2]/cfg.ANIS[2])]
#                     all_aff_sv = 0
#                     for tv in edg[sv]:
#                         tn = svs[tv] - stack_pad
#                         ptn = [int(tn[0]/(cfg.ANIS[0]*cfg.RESZ)), int(tn[1]/(cfg.ANIS[1]*cfg.RESZ)), int(tn[2]/cfg.ANIS[2])]
#                         aff_sz = affs.shape[1]//(cfg.CUTS[0]//cfg.RESZ)
#                         aff_sv = affinity_in_skel(affs, psn, ptn, resz=aff_sz)
#                         all_aff_sv += aff_sv
#                     avg_aff_sv = all_aff_sv / len(edg[sv])

#                     if 0 < avg_aff_sv < cfg.MERGE.AFFT:
#                         mesv.append(svs[sv])
#                         mesa.append(avg_aff_sv)
#                         messz = len(mesv)
#                         if big_merge_error is None:
#                             bigsz = 0
#                         else:
#                             bigsz = len(big_merge_error.vertices)
#                         if messz > 1 and \
#                             (abs(mesv[messz-2][0]-mesv[messz-1][0]) ** 2 + \
#                             abs(mesv[messz-2][1]-mesv[messz-1][1]) ** 2 + \
#                             abs(mesv[messz-2][2]-mesv[messz-1][2]) ** 2) < (cfg.MERGE.JOSK ** 2):
#                             mese.append(np.array([messz-2+bigsz, messz-1+bigsz]))
#                 if len(mesv) > 1 and len(mese) > 0:
#                     mesv = np.vstack(mesv).astype(np.float32)
#                     mese = np.vstack(mese).astype(np.uint32)
#                     mesr = (np.max([cfg.ANIS[0]*cfg.RESZ, cfg.ANIS[1]*cfg.RESZ, cfg.ANIS[2]]) * \
#                         np.ones(mesv.shape)).astype(np.float32)
#                     mest = np.zeros(mesv.shape, dtype=np.uint8)
#                     if big_merge_error is None:
#                         big_merge_error = Skeleton(vertices=mesv, \
#                         edges=mese, radii=mesr, vertex_types=mest, segid=1)  # default segid = 1 for merge error
#                         big_merge_error_af = mesa
#                         big_merge_error_sk = [sk for i in range(len(mesa))]
#                     else:
#                         big_merge_error.vertices = np.concatenate([big_merge_error.vertices, mesv])
#                         big_merge_error.edges = np.concatenate([big_merge_error.edges, mese])
#                         big_merge_error.radii = np.concatenate([big_merge_error.radii, mesr])  
#                         big_merge_error.vertex_types = np.concatenate([big_merge_error.vertex_types, mest])
#                         big_merge_error_af += mesa
#                         big_merge_error_sk += [sk for i in range(len(mesa))]

# print('Maped all affinities.')
# del big_edges

# big_merge_error_af = np.array(big_merge_error_af, dtype=np.float32)
# big_merge_error_sk = np.array(big_merge_error_sk, dtype=np.uint32)

# pmes, pafs, psks = _components(big_merge_error, big_merge_error_af, big_merge_error_sk)
# touch_merge_dic = process_merge_post(pmes, pafs, psks, cfg)

# pickle.dump(touch_merge_dic, open(join(root_dir, 'results/merge_error_1p.pkl'), 'wb'), protocol=4)
# print('--- Merge error number:', len(touch_merge_dic.keys()))    
# o_time(s_time, 'merge error finding')
