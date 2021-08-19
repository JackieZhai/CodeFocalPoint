'''
t1: This gonna process whole merge errors of wafer 14.
'''

from cloudvolume.skeleton import Skeleton
import fastremap
import kimimaro
import imageio
import cv2
import numpy as np
from h5py import File

from tqdm import tqdm
import time
from pathlib import Path
import pickle
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
aff_dir = root_dir.joinpath('affnity/')
seg_dir = root_dir.joinpath('seg/')
map_dir = root_dir.joinpath('merge.dat')

anis = (10, 10, 40)
cuts = (2000, 2000, 209)
dust = 10000
ovlp = 200
resz = 4
core = 24
jora = 1600
josk = 80
afft = 100
affn = 4
oanis = (5, 5, 40)
obias = (0, 8000, 0)

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
            affs = File(aff_dir.joinpath(aff_name), 'r')['vol0'][:]
            affs = np.transpose(affs, (0, 3, 2, 1))
            stack_pad = [remap_x * cuts[0] * anis[0], \
                        remap_y * cuts[1] * anis[1], \
                        remap_z * cuts[2] * anis[2]]
            
            for sk in stack_sk[stack_name]:
                svs = big_skels[sk].vertices
                edg = big_edges[sk]
                mesv = []
                mesa = []  # corresponding affinity
                mese = []
                for sv in range(svs.shape[0]):
                    sn = svs[sv] - stack_pad
                    psn = [int(sn[0]/(anis[0]*resz)), int(sn[1]/(anis[1]*resz)), int(sn[2]/anis[2])]
                    all_aff_sv = 0
                    for tv in edg[sv]:
                        tn = svs[tv] - stack_pad
                        ptn = [int(tn[0]/(anis[0]*resz)), int(tn[1]/(anis[1]*resz)), int(tn[2]/anis[2])]
                        aff_sv = affinity_in_skel(affs, psn, ptn, resz=resz)
                        all_aff_sv += aff_sv
                    avg_aff_sv = all_aff_sv / len(edg[sv])

                    if 0 < avg_aff_sv < afft:
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
                            abs(mesv[messz-2][2]-mesv[messz-1][2]) ** 2) < (josk ** 2):
                            mese.append(np.array([messz-2+bigsz, messz-1+bigsz]))
                if len(mesv) > 1 and len(mese) > 0:
                    mesv = np.vstack(mesv).astype(np.float32)
                    mese = np.vstack(mese).astype(np.uint32)
                    mesr = (np.max([anis[0]*resz, anis[1]*resz, anis[2]]) * \
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
    if len(pme.vertices) >= affn:
        paf = pafs[p]
        psk = psks[p]
        pskt = np.unique(psk)[0]
        paft = (255 - np.mean(paf)) / 255
        pmin = np.min(pme.vertices, axis=0)
        pmax = np.max(pme.vertices, axis=0)
        pmean = np.mean(pme.vertices, axis=0)
        pdic = {}
        pdic['pos'] = (pmean/oanis+obias).astype(np.uint32).tolist()
        pdic['min'] = (pmin/oanis+obias).astype(np.uint32).tolist()
        pdic['max'] = (pmax/oanis+obias).astype(np.uint32).tolist()
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

pickle.dump(touch_merge_dic, open('merge_error_1p.pkl', 'wb'), protocol=4)

print('--- Merge error number:', len(touch_merge_dic.keys()))    
o_time(s_time)
