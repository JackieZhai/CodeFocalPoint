from cloudvolume.skeleton import Skeleton
import fastremap

import numpy as np
from tqdm import tqdm 


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
      raise AssertionError
      # return []
    elif len(forest) == 1:
      raise AssertionError
      # return [ skel ]

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

def process_merge_post(pmes, pafs, psks, cfg):
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

    return touch_merge_dic
