from sklearn.neighbors import NearestNeighbors

from tqdm import tqdm
import math
import numpy as np


def skels_to_endpoints(big_skels, big_edges, cfg):
    endpoints, endpoints_vector = {}, {}
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
    return endpoints, endpoints_vector

def process_endpoint_split_checking(big_skels, endpoints, endpoints_vector, cfg):
    touch_split_dic, touch_split_num = {}, 0

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
            sk1 = eps_sk[e1]; sk2 = eps_sk[e2]
            sv1 = eps_sv[e1]; sv2 = eps_sv[e2]
            epset1, epset2 = [], []
            for sv in endpoints[sk1][sv1]:
                epset1.append(big_skels[sk1].vertices[sv])
            for sv in endpoints[sk2][sv2]:
                epset2.append(big_skels[sk2].vertices[sv])
            # zset1, zset2 = set(), set()
            # for pp in epset1:
            #     zset1.add(pp[2])
            # for pp in epset2:
            #     zset2.add(pp[2])
            # epset_z_iou = 1.0 * len(zset1 & zset2) / len(zset1 | zset2)
            # if epset_z_iou > cfg.SPLIT.XXX:
            #     continue
            v1 = eps_vec[e1]
            v2 = eps_vec[e2]
            dot_z_product = [v1[2], v2[2]]
            if dot_z_product[0] < math.cos(cfg.SPLIT.SPPA) and \
                dot_z_product[1] < math.cos(cfg.SPLIT.SPPA):
                continue
            dot_product = - (v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2])
            cos_spra = math.cos(cfg.SPLIT.SPRA)
            # if (dot_product == 0) or (dot_product >= cos_spra):
            if dot_product >= cos_spra:
                sample1 = (eps[e1]/cfg.OANIS+cfg.OBIAS).astype(np.uint32).tolist()
                sample2 = (eps[e2]/cfg.OANIS+cfg.OBIAS).astype(np.uint32).tolist()
                epmean = (np.mean((eps[e1], eps[e2]), axis=0)/cfg.OANIS+cfg.OBIAS).astype(np.uint32).tolist()
                epset = epset1 + epset2
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
                    eepiouf = False
                    for eepdic in touch_split_dic[(sk1, sk2)]:
                        epminmin = np.min(np.array([epdic['min'], eepdic['min']]), axis=0).tolist()
                        epminmax = np.max(np.array([epdic['min'], eepdic['min']]), axis=0).tolist()
                        epmaxmin = np.min(np.array([epdic['max'], eepdic['max']]), axis=0).tolist()
                        epmaxmax = np.max(np.array([epdic['max'], eepdic['max']]), axis=0).tolist()
                        if epminmax[0] >= epmaxmin[0] or \
                            epminmax[1] >= epmaxmin[1] or \
                            epminmax[2] >= epmaxmin[2]:
                            eepiou = 0.00
                        else:
                            eepiou = 1.0 * ((epmaxmin[0]-epminmax[0]) * (epmaxmin[1]-epminmax[1]) * \
                                (epmaxmin[1]-epminmax[1])) / ((epmaxmax[0]-epminmin[0]) * \
                                (epmaxmax[1]-epminmin[1]) * (epmaxmax[2]-epminmin[2]))
                        if eepiou > cfg.SPLIT.SPOV:
                            eepiouf = True
                            break
                    if eepiouf:
                        continue
                    touch_split_dic[(sk1, sk2)].append(epdic)
                else:
                    touch_split_dic[(sk1, sk2)] = [epdic]
                touch_split_num += 1
    
    return touch_split_dic, touch_split_num