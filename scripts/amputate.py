import numpy as np


# This is the operation for a single amputate_mat
def process_amputate_conversion(amputate_dic, amputate_mat, skelmap, cfg):
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