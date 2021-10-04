import kimimaro

from copy import deepcopy
from tqdm import tqdm


def process_skeletonize(big_skels, stack_sk, stack_name, labels, stack_pad, skelmap, cfg):
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
            
def process_skel_merging(big_skels, cfg):
    for sk in tqdm(big_skels.keys()):
        if len(big_skels[sk]) == 1:
            big_skels[sk] = big_skels[sk][0]
        elif len(big_skels[sk]) > 1:
            merge_skel = kimimaro.join_close_components(big_skels[sk], radius=cfg.SKEL.JORA)
            merge_skel = kimimaro.postprocess(merge_skel, dust_threshold=0, tick_threshold=0)
            big_skels[sk] = merge_skel
        else:
            raise Exception('Error in merging skels.')

def skels_to_edges(big_skels):
    big_edges = {}
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
    return big_edges
