'''
t1: This gonna process whole shunt(branch/strange) errors of wafer 14.
'''

from cloudvolume.skeleton import Skeleton
import fastremap
import kimimaro
import imageio
import cv2
import numpy as np
from h5py import File
from sklearn.neighbors import NearestNeighbors
import networkx

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
brve = 32  # vector trace length
brp1 = 1.0
brp2 = 0.75
stra = 1.58  # 1.58r=90d
divd = 12  # near points-pair need to be delete one
margin = 0.025  # margin rate for point delete?

# obias, omin, omax should be calcuated before
oanis = (5, 5, 40)
obias = (0, 8000, 0)  # should start from 01, but 03
omin = (0 * oanis[0], 8000 * oanis[1], 0 * oanis[2])
omax = ((72000+400) * oanis[0], (136000+400) * oanis[1], 209 * oanis[2])

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


def _min_max_position(x, y, z, sk, zpad=1, xypad=50):
    svs = big_skels[sk].vertices
    sv_set = []
    for sv in range(big_skels[sk].vertices.shape[0]):
        for pz in range(z-zpad*oanis[2], z+(zpad+1)*oanis[2], oanis[2]):
            if svs[sv][2] == pz and \
                x-xypad*oanis[0]<= svs[sv][0] <= x+xypad*oanis[0] and \
                y-xypad*oanis[1]<= svs[sv][1] <= y+xypad*oanis[1]:
                sv_set.append(svs[sv])
    sv_set = np.vstack(sv_set).astype(np.float32)
    sv_min = np.min(sv_set, axis=0)
    sv_max = np.max(sv_set, axis=0)
    
    return sv_min, sv_max


def _process_branchpoints(sk, branch_dic):
    p = brve
    skves = big_skels[sk].vertices
    skeds = big_skels[sk].edges
    list0, list1 = skeds[:, 0].tolist(), skeds[:, 1].tolist()
    list_all = list0 + list1
    nodedegree = {}
    for key in list_all:
        nodedegree[key] = nodedegree.get(key, 0) + 1

    branchpoint = []
    for k, f in nodedegree.items():
        if f >= 3:
            dir = 0
            normal = 0
            for dire in range(f):
                points = []
                now_sv = k
                next_sv = big_edges[sk][k][dire]
                for num in range(int(p)):
                    points.append(next_sv)
                    next_sv_list = deepcopy(big_edges[sk][next_sv])
                    next_sv_list.remove(now_sv)
                    if len(next_sv_list) == 0:
                        # next_sv_list.append(now_sv)
                        break
                    if len(next_sv_list) >= 2:
                        # next_sv_list.append(now_sv)
                        break
                    now_sv = next_sv
                    next_sv = next_sv_list[0]
                    # next_sv_list.append(now_sv)
                if len(points) == int(p * brp1):
                    dir += 1
                if len(points) >= int(p * brp2):
                    normal += 1
            if dir >= 2 and normal >= 3:
                x = int(skves[k][0])
                y = int(skves[k][1])
                z = int(skves[k][2])
                if omin[0] + (omax[0]-omin[0]) * margin < x < omax[0] - (omax[0]-omin[0]) * margin:
                    if omin[1] + (omax[1]-omin[1]) * margin < y < omax[1] - (omax[1]-omin[1]) * margin:
                        smin, smax = _min_max_position(x, y, z, sk)
                        smin = (smin/oanis+obias).astype(np.uint32).tolist()
                        smax = (smax/oanis+obias).astype(np.uint32).tolist()
                        spos = np.array([x, y, z])
                        spos = (spos/oanis+obias).astype(np.uint32).tolist()
                        branchpoint.append({'pos': spos, 'min': smin, 'max': smax, 'k': k})
    
    # print(sk, ':', len(branchpoint))
    l = len(branchpoint)
    if l > 0:
        if sk in branch_dic.keys():
            branch_dic[sk] += branchpoint
        else:
            branch_dic[sk] = branchpoint


def _process_strangepoints(sk, strage_dic, lock):
    svs = big_skels[sk].vertices
    ses = big_skels[sk].edges
    list0, list1 = ses[:, 0].tolist(), ses[:, 1].tolist()
    list_all = list0 + list1
    nodedegree = {}
    for key in list_all:
        nodedegree[key] = nodedegree.get(key, 0) + 1

    strangepoint = []
    mindirection = math.cos(stra)  # min direction dot for strangepoint
    for k, f in nodedegree.items():
        if f == 2:
            d = _calculate_direction(k, sk, svs)
            if d < mindirection:
                mindirection = d  # only find the worst d
                x = int(svs[k][0])
                y = int(svs[k][1])
                z = int(svs[k][2])
                if omin[0] + (omax[0]-omin[0]) * margin < x < omax[0] - (omax[0]-omin[0]) * margin:
                    if omin[1] + (omax[1]-omin[1]) * margin < y < omax[1] - (omax[1]-omin[1]) * margin:
                        smin, smax = _min_max_position(x, y, z, sk)
                        smin = (smin/oanis+obias).astype(np.uint32).tolist()
                        smax = (smax/oanis+obias).astype(np.uint32).tolist()
                        spos = np.array([x, y, z])
                        spos = (spos/oanis+obias).astype(np.uint32).tolist()
                        if len(strangepoint) == 0:
                            strangepoint.append({'pos': spos, 'min': smin, 'max': smax, 'k': k})
                        else:
                            strangepoint[0] = {'pos': spos, 'min': smin, 'max': smax, 'k': k}

    # print(sk, ':', len(strangepoint))
    l = len(strangepoint)
    if l > 0:
        if sk in strage_dic.keys():
            lock.acquire()
            strage_dic[sk] += strangepoint
            lock.release()
        else:
            lock.acquire()
            strage_dic[sk] = strangepoint
            lock.release()


def _calculate_direction(k, sk, svs):
    p = brve
    points = []
    now_sv = k
    next_sv = big_edges[sk][k][0]
    for num in range(int(p)):
        points.append(next_sv)
        next_sv_list = deepcopy(big_edges[sk][next_sv])
        next_sv_list.remove(now_sv)
        if len(next_sv_list) == 0:
            # next_sv_list.append(now_sv)
            break
        if len(next_sv_list) >= 2:
            # next_sv_list.append(now_sv)
            break
        now_sv = next_sv
        next_sv = next_sv_list[0]
        # next_sv_list.append(now_sv)
    if len(points) <= int(p * 0.5):
        vector_a = [0.0, 0.0, 0.0]
    else:
        e = svs[k]
        s = 0
        for m in range(len(points)):
            s += svs[points[m]]
        s /= len(points)
        norm_a = math.sqrt((e[0] - s[0]) ** 2 + (e[1] - s[1]) ** 2 + (e[2] - s[2]) ** 2)
        vector_a = [(e[0] - s[0]) / norm_a, (e[1] - s[1]) / norm_a, (e[2] - s[2]) / norm_a]

    points = []
    now_sv = k
    next_sv = big_edges[sk][k][1]
    for num in range(int(p)):
        points.append(next_sv)
        next_sv_list = deepcopy(big_edges[sk][next_sv])
        next_sv_list.remove(now_sv)
        if len(next_sv_list) == 0:
            # next_sv_list.append(now_sv)
            break
        if len(next_sv_list) >= 2:
            # next_sv_list.append(now_sv)
            break
        now_sv = next_sv
        next_sv = next_sv_list[0]
        # next_sv_list.append(now_sv)
    if len(points) <= int(p * 0.5):
        vector_b = [0.0, 0.0, 0.0]
    else:
        e = svs[k]
        s = 0
        for m in range(len(points)):
            s += svs[points[m]]
        s /= len(points)
        norm_b = math.sqrt((e[0] - s[0]) ** 2 + (e[1] - s[1]) ** 2 + (e[2] - s[2]) ** 2)
        vector_b = [-(e[0] - s[0]) / norm_b, -(e[1] - s[1]) / norm_b, -(e[2] - s[2]) / norm_b]

    return np.dot(np.array(vector_a), np.array(vector_b))


def _branstran_post(bs, sk):
    # print(bs)
    l = len(bs)
    original = []
    for itemdict in bs:
        original.append(itemdict['k'])
    a = sorted(original)
    a_index = list(np.argsort(np.array(original)))

    # b = a + [a[l-1]+100]
    # c = [b[i+1]-b[i] for i in range(l)]
    # print(c)
    # c_array = np.array(c)
    # d = np.where(c_array < 5)

    retain = []
    retain_seq = []
    for i in range(l):
        if i == 0:
            retain.append(a[0])
            retain_seq.append(i)
        else:
            if a[i]-retain[-1] >= divd:
                retain.append(a[i])
                retain_seq.append(i)

    ed = big_skels[sk].edges
    G_edges = [tuple(i) for i in list(ed)]
    G = networkx.Graph()
    G.add_edges_from(G_edges)

    eccentricity = []
    for itemnum in retain_seq:
        p_dict = networkx.shortest_path_length(G, source=a[itemnum])
        maxkey = max(p_dict, key=p_dict.get)
        maxdistance = p_dict[maxkey]
        eccentricity.append(maxdistance)
    emax = max(eccentricity)
    emin = min(eccentricity)
    if emax == emin:
        score = [1.0] * len(eccentricity)
    else:
        score = [((emax - eccentricity[i]) / (emax - emin)) for i in range(len(eccentricity))]

    radiu = big_skels[sk].radius
    voxel_num = radiu.shape[0]
    maxradiu = np.partition(radiu, int(0.75*voxel_num))[int(0.75*voxel_num)]

    def _label_score(a, b):  # heuristic: cylinder
        return a * (b ** 2)
    
    post = [_label_score(voxel_num, maxradiu)]
    j = 0
    for i in retain_seq:
        dict_i = deepcopy(bs[a_index[i]])
        dict_i.pop('k')
        dict_i['score'] = score[j]
        j += 1
        post.append(dict_i)

    return post


branch_dic = {}
for sk in tqdm(big_skels.keys()):
    _process_branchpoints(sk, branch_dic)
branch_num = 0
for sk in branch_dic.keys():
    branch_num += len(branch_dic[sk])
o_time(s_time, 'branch errors')

strage_dic = Manager().dict()
ps_lock = Manager().Lock()
ps_partial = partial(_process_strangepoints, strage_dic=strage_dic, lock=ps_lock)
with Pool(processes=core) as pool:
    pool.map(ps_partial, big_skels.keys())
    pool.close(); pool.join()
strange_num = 0
for sk in strage_dic.keys():
    strange_num += len(strage_dic[sk])
o_time(s_time, 'strange errors')

# combine branch/strange as divide errors
branstran_dic = {}
for sk in branch_dic.keys():
    if sk in strage_dic.keys():
        bs = branch_dic[sk] + strage_dic[sk]
    else:
        bs = branch_dic[sk]
    bs_post = _branstran_post(bs, sk)
    branstran_dic[sk] = bs_post
for sk in strage_dic.keys():
    if sk not in branch_dic.keys():
        bs = strage_dic[sk]
        bs_post = _branstran_post(bs, sk)
        branstran_dic[sk] = bs_post
s = [i[1][0] for i in branstran_dic.items()]
slen = len(s)
# smax = max(s)
smax = np.partition(np.array(s), int(0.8 * slen))[int(0.8 * slen)]
# smin = min(s)
smin = np.partition(np.array(s), int(0.2 * slen))[int(0.2 * slen)]
branstran_num = 0
for i in branstran_dic.keys():
    branstran_dic[i][0] = max(min((1.0 * (branstran_dic[i][0] - smin) / (smax - smin)), 1.0), 0)
    branstran_num += len(branstran_dic[i]) - 1

pickle.dump(branstran_dic, open('divide_error_1p.pkl', 'wb'), protocol=4)
o_time(s_time, 'post processing divide errors')

print('--- Branch error number:', branch_num)
print('--- Strange error number:', strange_num)
print('--- Divide error number:', branstran_num)
o_time(s_time)