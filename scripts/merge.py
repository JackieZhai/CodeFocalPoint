from cloudvolume.skeleton import Skeleton
import fastremap

import numpy as np
from tqdm import tqdm 

from utils.compute_affinity import affinity_in_skel

    
def process_affinity_merge_check(big_merge_error, big_merge_error_sk, big_merge_error_af, affs, \
    big_skels, big_edges, stack_sk_name, stack_pad, cfg):
    for sk in stack_sk_name:
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

def process_merge_post(touch_merge_dic, pmes, pafs, psks, cfg):
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
