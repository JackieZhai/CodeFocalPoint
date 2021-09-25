import networkx

import numpy as np
from copy import deepcopy
import math
from tqdm import tqdm
from multiprocessing import Manager, Pool
from functools import partial


def _min_max_position(big_skels, cfg, x, y, z, sk, zpad=1, xypad=50):
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

def process_branch_checking(big_skels, big_edges, cfg):
    branch_dic, branch_num = {}, 0

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
                            smin, smax = _min_max_position(big_skels, cfg, x, y, z, sk)
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
    
    for sk in tqdm(big_skels.keys()):
        _process_branchpoints(sk, branch_dic)
    for sk in branch_dic.keys():
        branch_num += len(branch_dic[sk])
    
    return branch_dic, branch_num


def _calculate_direction(k, sk, svs, big_edges, cfg):
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

def _process_strangepoints(sk, strange_dic, big_skels, big_edges, cfg, lock):
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
                d = _calculate_direction(k, sk, svs, big_edges, cfg)
                if d < mindirection:
                    mindirection = d  # only find the worst d
                    x = int(svs[k][0])
                    y = int(svs[k][1])
                    z = int(svs[k][2])
                    if cfg.OMIN[0] + (cfg.OMAX[0]-cfg.OMIN[0]) * cfg.DIVIDE.MARGIN < x < cfg.OMAX[0] \
                        - (cfg.OMAX[0]-cfg.OMIN[0]) * cfg.DIVIDE.MARGIN:
                        if cfg.OMIN[1] + (cfg.OMAX[1]-cfg.OMIN[1]) * cfg.DIVIDE.MARGIN < y < cfg.OMAX[1] \
                            - (cfg.OMAX[1]-cfg.OMIN[1]) * cfg.DIVIDE.MARGIN:
                            smin, smax = _min_max_position(big_skels, cfg, x, y, z, sk)
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
            if sk in strange_dic.keys():
                lock.acquire()
                strange_dic[sk] += strangepoint
                lock.release()
            else:
                lock.acquire()
                strange_dic[sk] = strangepoint
                lock.release()

def process_strange_checking(big_skels, big_edges, cfg):
    strange_dic, strange_num = {}, 0

    strange_dic = Manager().dict()
    ps_lock = Manager().Lock()
    ps_partial = partial(_process_strangepoints, strage_dic=strange_dic, \
        big_skels=big_skels, big_edges=big_edges, cfg=cfg, lock=ps_lock)
    with Pool(processes=cfg.CORE) as pool:
        pool.map(ps_partial, big_skels.keys())
        pool.close(); pool.join()
    for sk in strange_dic.keys():
        strange_num += len(strange_dic[sk])
    
    return strange_dic, strange_num


def process_branstran_post(branch_dic, strange_dic, big_skels, cfg):
    branstran_dic, branstran_num = {}, 0

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

    for sk in branch_dic.keys():
        if sk in strange_dic.keys():
            bs = branch_dic[sk] + strange_dic[sk]
        else:
            bs = branch_dic[sk]
        bs_post = _branstran_post(bs, sk)
        branstran_dic[sk] = bs_post
    for sk in strange_dic.keys():
        if sk not in branch_dic.keys():
            bs = strange_dic[sk]
            bs_post = _branstran_post(bs, sk)
            branstran_dic[sk] = bs_post
    s = [i[1][0] for i in branstran_dic.items()]
    slen = len(s)
    smax = np.partition(np.array(s), int(0.8 * slen))[int(0.8 * slen)]
    smin = np.partition(np.array(s), int(0.2 * slen))[int(0.2 * slen)]
    for i in branstran_dic.keys():
        branstran_dic[i][0] = max(min((1.0 * (branstran_dic[i][0] - smin) / (smax - smin)), 1.0), 0)
        branstran_num += len(branstran_dic[i]) - 1
    
    return branstran_dic, branstran_num
