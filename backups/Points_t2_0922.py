'''
t2: This gonna process whole 4 different types at same time from configs.
'''

import kimimaro
from cloudvolume.skeleton import Skeleton
import fastremap
import imageio
import cv2
import numpy as np
from h5py import File
from sklearn.neighbors import NearestNeighbors
import networkx

from tqdm import tqdm
import time
import pickle
import math
from multiprocessing import Pool, Manager
from copy import deepcopy
from functools import partial
from os.path import dirname, abspath, join
from pathlib import Path

from configs.config import cfg
from utils.load_remap import read_binary_dat, read_file_system
from utils.compute_affinity import affinity_in_skel

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

            # Skeleton
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
                dust_threshold=int((cfg.SKEL.DUST**3)/(cfg.ANIS[0]*cfg.ANIS[1]*cfg.ANIS[2])/(cfg.RESZ**2)),
                    # skip connected components with fewer than this many voxels
                anisotropy=(cfg.ANIS[0]*cfg.RESZ, cfg.ANIS[1]*cfg.RESZ, cfg.ANIS[2]), # physical units
                fix_branching=True,  # default True
                fix_borders=True,  # default True
                fill_holes=False,  # default False
                fix_avocados=False,  # default False
                progress=False,  # default False, show progress bar
                parallel=cfg.CORE,  # <= 0 all cpu, 1 single process, 2+ multiprocess
                parallel_chunk_size=cfg.SKEL.CHUK,  # how many skeletons to process before updating progress bar
            )

            stack_pad = [remap_x * cfg.CUTS[0] * cfg.ANIS[0], \
                remap_y * cfg.CUTS[1] * cfg.ANIS[1], \
                remap_z * cfg.CUTS[2] * cfg.ANIS[2]]
            skelmap = arr_read[remap_z, remap_y, remap_x]
            for sk in skels.keys():
                skels[sk].vertices = skels[sk].vertices + stack_pad
                skels[sk].id = skelmap[sk]
                if skelmap[sk] in big_skels.keys():
                    big_skels[skelmap[sk]].append(deepcopy(skels[sk]))
                else:
                    big_skels[skelmap[sk]] = [deepcopy(skels[sk])]
                if stack_name in stack_sk.keys():
                    stack_sk[stack_name].append(skelmap[sk])
                else:
                    stack_sk[stack_name] = [skelmap[sk]]
            
            # Amputate
            amp_dir = join(seg_dir, stack_name, 'focus.pkl')
            amputate_mat = pickle.load(open(amp_dir, 'rb'))
            for label_pair in amputate_mat.keys():
                amp_list = amputate_mat[label_pair]
                ans_list = []
                for amp in amp_list:
                    ans_dic = {}
                    ans_dic['pos'] = (np.array(amp[4])*cfg.ANIS/cfg.OANIS+cfg.OBIAS)\
                        .astype(np.uint32).tolist()
                    ans_dic['min'] = (np.array(amp[1])*cfg.ANIS/cfg.OANIS+cfg.OBIAS)\
                        .astype(np.uint32).tolist()
                    ans_dic['max'] = (np.array(amp[0])*cfg.ANIS/cfg.OANIS+cfg.OBIAS)\
                        .astype(np.uint32).tolist()
                    ans_dic['sample1'] = (np.array(amp[2])*cfg.ANIS/cfg.OANIS+cfg.OBIAS)\
                        .astype(np.uint32).tolist()
                    ans_dic['sample2'] = (np.array(amp[3])*cfg.ANIS/cfg.OANIS+cfg.OBIAS)\
                        .astype(np.uint32).tolist()
                    ans_dic['score'] = amp[5]
                    ans_list.append(ans_dic)
                label1 = skelmap[label_pair[0]]
                label2 = skelmap[label_pair[1]]
                amputate_dic[(label1, label2)] = ans_list

for sk in big_skels.keys():
    if len(big_skels[sk]) == 1:
        big_skels[sk] = big_skels[sk][0]
    elif len(big_skels[sk]) > 1:
        merge_skel = kimimaro.join_close_components(big_skels[sk], radius=cfg.SKEL.JORA)
        merge_skel = kimimaro.postprocess(merge_skel, dust_threshold=0, tick_threshold=0)
        big_skels[sk] = merge_skel
    else:
        raise Exception('Error in merging skels.')
print('Extract \'all\' skels number:', len(big_skels))

pickle.dump(big_skels, open(join(root_dir, 'results/big_skels.pkl'), 'wb'), protocol=4)
pickle.dump(stack_sk, open(join(root_dir, 'results/stack_sk.pkl'), 'wb'), protocol=4)
pickle.dump(amputate_dic, open(join(root_dir, 'results/amputate_error_2p.pkl'), 'wb'), protocol=4)
print('--- Amputate error number:', len(amputate_dic.keys()))
o_time(s_time, 'skeletonizing & amputate error finding')

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
            for num in range(cfg.SPLIT.SPVE):
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

# Split
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
eps_nbr = NearestNeighbors(n_neighbors=cfg.SPLIT.SPLN+1, algorithm='auto').fit(eps)
diss, inds = eps_nbr.kneighbors(eps)
print('Calculated K-Neighbors of endpoints.')

touch_split_dic = {}
touch_split_num = 0
for e in tqdm(range(eps.shape[0])):
    for en in range(1, cfg.SPLIT.SPLN+1):
        dis = diss[e, en]
        ind = inds[e, en]
        if dis > cfg.SPLIT.SPLD:
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
        cos_spra = math.cos(cfg.SPLIT.SPRA)
        # if (dot_product == 0) or (dot_product >= cos_spra):
        if dot_product >= cos_spra:
            sk1 = eps_sk[e1]; sk2 = eps_sk[e2]
            sv1 = eps_sv[e1]; sv2 = eps_sv[e2]
            sample1 = (eps[e1]/cfg.OANIS+cfg.OBIAS).astype(np.uint32).tolist()
            sample2 = (eps[e2]/cfg.OANIS+cfg.OBIAS).astype(np.uint32).tolist()
            epmean = (np.mean((eps[e1], eps[e2]), axis=0)/cfg.OANIS+cfg.OBIAS).astype(np.uint32).tolist()
            epset = []
            for sv in endpoints[sk1][sv1]:
                epset.append(big_skels[sk1].vertices[sv])
            for sv in endpoints[sk2][sv2]:
                epset.append(big_skels[sk2].vertices[sv])
            epset = np.vstack(epset).astype(np.float32)
            epmax = (np.max(epset, axis=0)/cfg.OANIS+cfg.OBIAS).astype(np.uint32).tolist()
            epmin = (np.min(epset, axis=0)/cfg.OANIS+cfg.OBIAS).astype(np.uint32).tolist()
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

pickle.dump(touch_split_dic, open(join(root_dir, 'results/split_error_2p.pkl'), 'wb'), protocol=4)
print('--- Split error number:', touch_split_num)
o_time(s_time, 'split error finding')

# Divide
def _min_max_position(x, y, z, sk, zpad=1, xypad=50):
    svs = big_skels[sk].vertices
    sv_set = []
    for sv in range(big_skels[sk].vertices.shape[0]):
        for pz in range(z-zpad*cfg.OANIS[2], z+(zpad+1)*cfg.OANIS[2], cfg.OANIS[2]):
            if svs[sv][2] == pz and \
                x-xypad*cfg.OANIS[0]<= svs[sv][0] <= x+xypad*cfg.OANIS[0] and \
                y-xypad*cfg.OANIS[1]<= svs[sv][1] <= y+xypad*cfg.OANIS[1]:
                sv_set.append(svs[sv])
    sv_set = np.vstack(sv_set).astype(np.float32)
    sv_min = np.min(sv_set, axis=0)
    sv_max = np.max(sv_set, axis=0)
    
    return sv_min, sv_max

def _process_branchpoints(sk, branch_dic):
    p = cfg.DIVIDE.BRVE
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
                if len(points) == int(p * cfg.DIVIDE.BRP1):
                    dir += 1
                if len(points) >= int(p * cfg.DIVIDE.BRP2):
                    normal += 1
            if dir >= 2 and normal >= 3:
                x = int(skves[k][0])
                y = int(skves[k][1])
                z = int(skves[k][2])
                if cfg.OMIN[0] + (cfg.OMAX[0]-cfg.OMIN[0]) * cfg.DIVIDE.MARGIN < x < cfg.OMAX[0] \
                    - (cfg.OMAX[0]-cfg.OMIN[0]) * cfg.DIVIDE.MARGIN:
                    if cfg.OMIN[1] + (cfg.OMAX[1]-cfg.OMIN[1]) * cfg.DIVIDE.MARGIN < y < cfg.OMAX[1] \
                        - (cfg.OMAX[1]-cfg.OMIN[1]) * cfg.DIVIDE.MARGIN:
                        smin, smax = _min_max_position(x, y, z, sk)
                        smin = (smin/cfg.OANIS+cfg.OBIAS).astype(np.uint32).tolist()
                        smax = (smax/cfg.OANIS+cfg.OBIAS).astype(np.uint32).tolist()
                        spos = np.array([x, y, z])
                        spos = (spos/cfg.OANIS+cfg.OBIAS).astype(np.uint32).tolist()
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
    mindirection = math.cos(cfg.DIVIDE.STRA)  # min direction dot for strangepoint
    for k, f in nodedegree.items():
        if f == 2:
            d = _calculate_direction(k, sk, svs)
            if d < mindirection:
                mindirection = d  # only find the worst d
                x = int(svs[k][0])
                y = int(svs[k][1])
                z = int(svs[k][2])
                if cfg.OMIN[0] + (cfg.OMAX[0]-cfg.OMIN[0]) * cfg.DIVIDE.MARGIN < x < cfg.OMAX[0] \
                    - (cfg.OMAX[0]-cfg.OMIN[0]) * cfg.DIVIDE.MARGIN:
                    if cfg.OMIN[1] + (cfg.OMAX[1]-cfg.OMIN[1]) * cfg.DIVIDE.MARGIN < y < cfg.OMAX[1] \
                        - (cfg.OMAX[1]-cfg.OMIN[1]) * cfg.DIVIDE.MARGIN:
                        smin, smax = _min_max_position(x, y, z, sk)
                        smin = (smin/cfg.OANIS+cfg.OBIAS).astype(np.uint32).tolist()
                        smax = (smax/cfg.OANIS+cfg.OBIAS).astype(np.uint32).tolist()
                        spos = np.array([x, y, z])
                        spos = (spos/cfg.OANIS+cfg.OBIAS).astype(np.uint32).tolist()
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
    p = cfg.DIVIDE.BRVE
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
            if a[i]-retain[-1] >= cfg.DIVIDE.DIVD:
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
o_time(s_time, 'branch errors finding')

strage_dic = Manager().dict()
ps_lock = Manager().Lock()
ps_partial = partial(_process_strangepoints, strage_dic=strage_dic, lock=ps_lock)
with Pool(processes=cfg.CORE) as pool:
    pool.map(ps_partial, big_skels.keys())
    pool.close(); pool.join()
strange_num = 0
for sk in strage_dic.keys():
    strange_num += len(strage_dic[sk])
o_time(s_time, 'strange errors finding')

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
smax = np.partition(np.array(s), int(0.8 * slen))[int(0.8 * slen)]
smin = np.partition(np.array(s), int(0.2 * slen))[int(0.2 * slen)]
branstran_num = 0
for i in branstran_dic.keys():
    branstran_dic[i][0] = max(min((1.0 * (branstran_dic[i][0] - smin) / (smax - smin)), 1.0), 0)
    branstran_num += len(branstran_dic[i]) - 1

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
            
            for sk in stack_sk[stack_name]:
                svs = big_skels[sk].vertices
                edg = big_edges[sk]
                mesv = []
                mesa = []  # corresponding affinity
                mese = []
                for sv in range(svs.shape[0]):
                    sn = svs[sv] - stack_pad
                    psn = [int(sn[0]/(cfg.ANIS[0]*cfg.RESZ)), int(sn[1]/(cfg.ANIS[1]*cfg.RESZ)), int(sn[2]/cfg.ANIS[2])]
                    all_aff_sv = 0
                    for tv in edg[sv]:
                        tn = svs[tv] - stack_pad
                        ptn = [int(tn[0]/(cfg.ANIS[0]*cfg.RESZ)), int(tn[1]/(cfg.ANIS[1]*cfg.RESZ)), int(tn[2]/cfg.ANIS[2])]
                        aff_sz = affs.shape[1]//(cfg.CUTS[0]//cfg.RESZ)
                        aff_sv = affinity_in_skel(affs, psn, ptn, resz=aff_sz)
                        all_aff_sv += aff_sv
                    avg_aff_sv = all_aff_sv / len(edg[sv])

                    if 0 < avg_aff_sv < cfg.MERGE.AFFT:
                        mesv.append(svs[sv])
                        mesa.append(avg_aff_sv)
                        messz = len(mesv)
                        if big_merge_error is None:
                            bigsz = 0
                        else:
                            bigsz = len(big_merge_error.vertices)
                        if messz > 1 and \
                            (abs(mesv[messz-2][0]-mesv[messz-1][0]) ** 2 + \
                            abs(mesv[messz-2][1]-mesv[messz-1][1]) ** 2 + \
                            abs(mesv[messz-2][2]-mesv[messz-1][2]) ** 2) < (cfg.MERGE.JOSK ** 2):
                            mese.append(np.array([messz-2+bigsz, messz-1+bigsz]))
                if len(mesv) > 1 and len(mese) > 0:
                    mesv = np.vstack(mesv).astype(np.float32)
                    mese = np.vstack(mese).astype(np.uint32)
                    mesr = (np.max([cfg.ANIS[0]*cfg.RESZ, cfg.ANIS[1]*cfg.RESZ, cfg.ANIS[2]]) * \
                        np.ones(mesv.shape)).astype(np.float32)
                    mest = np.zeros(mesv.shape, dtype=np.uint8)
                    if big_merge_error is None:
                        big_merge_error = Skeleton(vertices=mesv, \
                        edges=mese, radii=mesr, vertex_types=mest, segid=1)  # default segid = 1 for merge error
                        big_merge_error_af = mesa
                        big_merge_error_sk = [sk for i in range(len(mesa))]
                    else:
                        big_merge_error.vertices = np.concatenate([big_merge_error.vertices, mesv])
                        big_merge_error.edges = np.concatenate([big_merge_error.edges, mese])
                        big_merge_error.radii = np.concatenate([big_merge_error.radii, mesr])  
                        big_merge_error.vertex_types = np.concatenate([big_merge_error.vertex_types, mest])
                        big_merge_error_af += mesa
                        big_merge_error_sk += [sk for i in range(len(mesa))]
print('Maped all affinities.')
del big_edges

big_merge_error_af = np.array(big_merge_error_af, dtype=np.float32)
big_merge_error_sk = np.array(big_merge_error_sk, dtype=np.uint32)

def _components(skels, af, sk):
    """
    * Modificated from skeleton.py in cloudvolume

    Extract connected components from graph. 
    Useful for ensuring that you're working with a single tree.

    Returns: [ Skeleton, Skeleton, ... ]
    """
    skel = skels.clone()
    forest = skels._compute_components(skel)
    
    if len(forest) == 0:
      return []
    elif len(forest) == 1:
      return [ skel ]

    skeletons = []
    afs = []
    sks = []
    for edge_list in forest:
      edge_list = np.array(edge_list, dtype=np.uint32)
      vert_idx = fastremap.unique(edge_list)

      vert_list = skel.vertices[vert_idx]
      radii = skel.radii[vert_idx]
      vtypes = skel.vertex_types[vert_idx]
      af_s = af[vert_idx]
      sk_s = sk[vert_idx]

      remap = { vid: i for i, vid in enumerate(vert_idx) }
      edge_list = fastremap.remap(edge_list, remap, in_place=True)

      skeletons.append(
        Skeleton(vert_list, edge_list, radii, vtypes, skel.id)
      )
      afs.append(af_s)
      sks.append(sk_s)

    return skeletons, afs, sks

pmes, pafs, psks = _components(big_merge_error, big_merge_error_af, big_merge_error_sk)
touch_merge_dic = {}
for p in tqdm(range(len(pmes))):
    pme = pmes[p]
    if len(pme.vertices) >= cfg.MERGE.AFFN:
        paf = pafs[p]
        psk = psks[p]
        pskt = np.unique(psk)[0]
        paft = (255 - np.mean(paf)) / 255
        pmin = np.min(pme.vertices, axis=0)
        pmax = np.max(pme.vertices, axis=0)
        pmean = np.mean(pme.vertices, axis=0)
        pdic = {}
        pdic['pos'] = (pmean/cfg.OANIS+cfg.OBIAS).astype(np.uint32).tolist()
        pdic['min'] = (pmin/cfg.OANIS+cfg.OBIAS).astype(np.uint32).tolist()
        pdic['max'] = (pmax/cfg.OANIS+cfg.OBIAS).astype(np.uint32).tolist()
        pdic['score'] = paft
        if pskt in touch_merge_dic.keys():
            touch_merge_dic[pskt].append(pdic)
        else:
            touch_merge_dic[pskt] = [-1, pdic]
p_dic_num_list = []
for t in touch_merge_dic.keys():
    p_dic_num_list.append(len(touch_merge_dic[t])-1)
p_dic_num_median = np.median(np.array(p_dic_num_list))
for t in touch_merge_dic.keys():
    p_dic_num = len(touch_merge_dic[t])-1
    if p_dic_num >= p_dic_num_median:
        touch_merge_dic[t][0] = 1.0
    else:
        touch_merge_dic[t][0] = 1.0 * p_dic_num / p_dic_num_median

pickle.dump(touch_merge_dic, open(join(root_dir, 'results/merge_error_1p.pkl'), 'wb'), protocol=4)
print('--- Merge error number:', len(touch_merge_dic.keys()))    
o_time(s_time, 'merge errors finding')
