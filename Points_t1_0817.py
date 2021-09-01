'''
t1: This gonna process whole split errors of wafer 14.
'''

from cloudvolume.skeleton import Skeleton
import fastremap
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


root_dir = Path('/dc1/SCN/wafer14/')
seg_dir = root_dir.joinpath('seg_after/')
map_dir = root_dir.joinpath('merge.dat')

anis = (20, 20, 40)
resz = 2

core = 24
spld = 800
spln = 3
spve = 16  # vector trace length
spra = 0.45  # 0.3216r=18.4d, 0.45r=25.8d

oanis = (5, 5, 40)
obias = (0, 8000, 0)  # should start from 01, but 03

arr_msg, arr_read = read_binary_dat(map_dir)
arr_fls = read_file_system(seg_dir)
print('Stack property:', arr_msg)

big_skels = pickle.load(open(root_dir.joinpath('big_skels.pkl'), 'rb'))  # all skels across stacks
stack_sk = pickle.load(open(root_dir.joinpath('stack_sk.pkl'), 'rb'))  # find all skels in each stack

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
print('Extracted all edges.')

endpoints = {}  # endpoint and its backtracking set
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
print('Extracted all endpoints and corresponding vectors.')
del big_edges

eps = []
eps_sk = []
eps_sv = []
eps_vec = []
for sk in big_skels.keys():
    svs = big_skels[sk].vertices
    ends = endpoints[sk]
    ends_ve = endpoints_vector[sk]
    for sv in ends.keys():
        eps.append(svs[sv])
        eps_sk.append(sk)
        eps_sv.append(sv)
        eps_vec.append(ends_ve[sv])
print('Endpoints number:', len(eps))

eps = np.vstack(eps).astype(np.float32)
eps_sk = np.array(eps_sk, dtype=np.uint32)
eps_sv = np.array(eps_sv, dtype=np.uint32)
eps_vec = np.array(eps_vec, dtype=np.float32)
eps_nbr = NearestNeighbors(n_neighbors=spln+1, algorithm='auto').fit(eps)
diss, inds = eps_nbr.kneighbors(eps)
print('Calculated K-Neighbors of endpoints.')

touch_split_dic = {}
touch_split_num = 0
for e in tqdm(range(eps.shape[0])):
    for en in range(1, spln+1):
        dis = diss[e, en]
        ind = inds[e, en]
        if dis > spld:
            continue
        if eps_sk[e] <= eps_sk[ind]:
            e1 = e
            e2 = ind
        else:
            e1 = ind
            e2 = e
        v1 = eps_vec[e1]
        v2 = eps_vec[e2]
        dot_product = - (v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2])
        cos_spra = math.cos(spra)
        # if (dot_product == 0) or (dot_product >= cos_spra):
        if dot_product >= cos_spra:
            sk1 = eps_sk[e1]; sk2 = eps_sk[e2]
            sv1 = eps_sv[e1]; sv2 = eps_sv[e2]
            sample1 = (eps[e1]/oanis+obias).astype(np.uint32).tolist()
            sample2 = (eps[e2]/oanis+obias).astype(np.uint32).tolist()
            epmean = (np.mean((eps[e1], eps[e2]), axis=0)/oanis+obias).astype(np.uint32).tolist()
            epset = []
            for sv in endpoints[sk1][sv1]:
                epset.append(big_skels[sk1].vertices[sv])
            for sv in endpoints[sk2][sv2]:
                epset.append(big_skels[sk2].vertices[sv])
            epset = np.vstack(epset).astype(np.float32)
            epmax = (np.max(epset, axis=0)/oanis+obias).astype(np.uint32).tolist()
            epmin = (np.min(epset, axis=0)/oanis+obias).astype(np.uint32).tolist()
            epscore = (dot_product - cos_spra) / (1 - cos_spra)
            epdic = {}
            epdic['pos'] = epmean
            epdic['min'] = epmin
            epdic['max'] = epmax
            epdic['sample1'] = sample1
            epdic['sample2'] = sample2
            epdic['score'] = epscore
            if (sk1, sk2) in touch_split_dic.keys():
                touch_split_dic[(sk1, sk2)].append(epdic)
            else:
                touch_split_dic[(sk1, sk2)] = [epdic]
            touch_split_num += 1

pickle.dump(touch_split_dic, open('split_error_2p.pkl', 'wb'), protocol=4)
print('--- Split error number:', touch_split_num)
o_time(s_time)