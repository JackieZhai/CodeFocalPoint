'''
t3: the split error checking process need to be improved.
improved idea from YuanJB, see "split.py" and "merge.py".
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


touch_merge_dic = pickle.load(open(join(root_dir, 'results/merge_error_1p.pkl'), 'rb'))
print('--- Merge error number:', len(touch_merge_dic.keys()))
touch_num = 0
for sk in touch_merge_dic.keys():
    touch_num += len(touch_merge_dic[sk]) - 1
print('--- Merge error real number:', touch_num)

def _point_iou(point1, point2):
    epminmin = np.min(np.array([point1['min'], point2['min']]), axis=0).tolist() 
    epminmax = np.max(np.array([point1['min'], point2['min']]), axis=0).tolist()
    epmaxmin = np.min(np.array([point1['max'], point2['max']]), axis=0).tolist()
    epmaxmax = np.max(np.array([point1['max'], point2['max']]), axis=0).tolist()
    if epminmax[0] >= epmaxmin[0] or \
        epminmax[1] >= epmaxmin[1] or \
        epminmax[2] >= epmaxmin[2]:
        eepiou = 0.00
    else:
        eepiou = 1.0 * ((epmaxmin[0]-epminmax[0]) * (epmaxmin[1]-epminmax[1]) * \
            (epmaxmin[1]-epminmax[1])) / ((epmaxmax[0]-epminmin[0]) * \
            (epmaxmax[1]-epminmin[1]) * (epmaxmax[2]-epminmin[2]))
    return eepiou

new_touch_merge_dic = {}
for sk in tqdm(touch_merge_dic.keys()):
    new_touch_merge_dic[sk] = [touch_merge_dic[sk][0]]
    merge_point_list = touch_merge_dic[sk][1:]
    for merge_i, merge_point in enumerate(merge_point_list):
        if merge_i == 0:
            new_touch_merge_dic[sk].append(merge_point)
        else:
            iou_flag = False
            for dist_i in range(1, len(new_touch_merge_dic[sk])):
                dist_point = new_touch_merge_dic[sk][dist_i]
                iou = _point_iou(merge_point, dist_point)
                if iou > 0.9:
                    iou_flag = True
                    break
            if not iou_flag:
                new_touch_merge_dic[sk].append(merge_point)

for sk in new_touch_merge_dic.keys():
    merge_point_list = new_touch_merge_dic[sk][1:]
    merge_score = 1.0
    for merge_i, merge_point in enumerate(merge_point_list):
        merge_score *= 1.0 - merge_point['score']
    new_touch_merge_dic[sk][0] = 1.0 - merge_score

pickle.dump(new_touch_merge_dic, open(join(root_dir, 'results/merge_error_1p_new.pkl'), 'wb'), protocol=4)
print('--- New merge error number:', len(new_touch_merge_dic.keys()))
touch_num = 0
for sk in new_touch_merge_dic.keys():
    touch_num += len(new_touch_merge_dic[sk]) - 1
print('--- Merge error real number:', touch_num)