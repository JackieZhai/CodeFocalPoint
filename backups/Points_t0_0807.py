# From: https://github.com/seung-lab/kimimaro

import kimimaro
from networkx.linalg.algebraicconnectivity import fiedler_vector
import numpy as np
import time
import pickle
import math
from multiprocessing import Pool, Lock, Value, Manager
from functools import partial
from copy import deepcopy
import argparse
import imageio
from PIL import Image
import shutil
from os import listdir, path, mkdir
import cv2
from skimage import io, transform
import networkx as nx
from matplotlib import pyplot as plt
from tqdm import tqdm


# label是tif存的路径
# endsave是多进程中间结果暂存路径
# branchsave是分叉点存储路径
# strangesave是奇异点存储路径
# core为进程数，如果资源有的话可以设大一点
# distance、angle分别是搜索距离（nm）和角度（弧度值）
# path是求端点vector时候回溯的点数量
# overlap为False时输出会删去一些靠的比较近的接触点（还未测试过是否会误删）
# bias表示分块操作时产生的偏置
parser = argparse.ArgumentParser(description='find endpoint neighbor with kimimaro')
parser.add_argument('--label', '-l', type=str, default='/dc1/SCN/wafer14/seg/1901/', help='label file path')######### data path
parser.add_argument('--endsave', '-es', type=str, default='endpoints_save/', help='endpoints save files path')
parser.add_argument('--branchsave', '-bs', type=str, default='answer_branch/', help='branch points save files path')
parser.add_argument('--strangesave', '-ss', type=str, default='answer_strange/', help='strange points save files path')
parser.add_argument('--gliocytehighparasave', '-ghs', type=str, default='gliocytes-high/', help='gliocyte labels')
parser.add_argument('--gliocytelowparasave', '-gls', type=str, default='gliocytes-low/', help='gliocyte labels')
parser.add_argument('--resize', '-r', type=int, default=4, help='image resize scale')############ need to ajdust by the relationship between SCN & zibrafish
parser.add_argument('--dust', '-du', type=int, default=10000, help='kimimaro skeleton dust threshold')######### big for less, small for more
parser.add_argument('--core', '-c', type=int, default=24, help='parallel core number')
parser.add_argument('--randomsample', '-rs', type=int, default=1000, help='random sample number for convex features')
parser.add_argument('--path', '-p', type=int, default=64, help='point vector trace path length (vertex number)--touchpoint')#########
parser.add_argument('--mergepath', '-mp', type=int, default=128, help='point vector trace path length (vertex number)--branchpoint')######### big for less, small for more
parser.add_argument('--mindot', '-m', type=float, default=-0.01, help='min direction dot for strangepoint')
parser.add_argument('--near', '-n', type=int, default=48, help='near points-pair need to be delete one')######### big for less, small for more
parser.add_argument('--distance', '-di', type=list, default=[240, 80], help='endpoint neighbors maximum distance')
parser.add_argument('--angle', '-ag', type=float, default=0.3216, help='endpoint neighbors maximum angle')
parser.add_argument('--overlap', '-o', type=bool, default=False, help='is save as overlap list?')
parser.add_argument('--margin', '-mg', type=float, default=0.025, help='margin rate for point delete?')
parser.add_argument('--anis', '-ai', type=tuple, default=(5, 5, 40), help='anisotropy ratio')
parser.add_argument('--bias', '-b', type=tuple, default=(0, 0, 0), help='x,y,z bias for block operation')
args = parser.parse_args()


# 以端点为中心在圆锥范围内搜索的函数
def traverse_endpoint(segmentation, center, vector, radius, resolution, max_label, maximum_distance, maximum_radians, gliocyte, bias, backward):
    bias = [int(bias[0]/resize_para), int(bias[1]/resize_para), bias[2]]

    # the maximum degrees is a function of how the endpoint vectors are generated
    # the vectors have at best this resolution accuracy
    # maximum_radians = 0.3216
    # save computation time by calculating cos(theta) here
    cos_theta = math.cos(maximum_radians)

    # decompress important variables
    zpoint, ypoint, xpoint = center
    zradius, yradius, xradius = (
    int(maximum_distance / resolution[0]), int(maximum_distance / resolution[1]), int(maximum_distance / resolution[2]))

    zres, yres, xres = segmentation.shape
    label = segmentation[zpoint - bias[0], ypoint - bias[1], xpoint - bias[2]]

    # # create a set of labels to ignore
    labels_to_ignore = set(gliocyte)
    # start by ignoring all labels with the same value
    labels_to_ignore.add(segmentation[zpoint - bias[0], ypoint - bias[1], xpoint - bias[2]])

    # keep track of what is adjacent in this cube and which potential neighbors are already on the stack
    adjacency_matrix = set()
    potential_neighbors = set()

    zmeans = np.zeros(max_label + 1, dtype=np.float32)
    ymeans = np.zeros(max_label + 1, dtype=np.float32)
    xmeans = np.zeros(max_label + 1, dtype=np.float32)
    counts = np.zeros(max_label + 1, dtype=np.float32)
    zmaxs = np.zeros(max_label + 1, dtype=np.int32)
    ymaxs = np.zeros(max_label + 1, dtype=np.int32)
    xmaxs = np.zeros(max_label + 1, dtype=np.int32)
    zmins = np.ones(max_label + 1, dtype=np.int32) * zres + bias[0]
    ymins = np.ones(max_label + 1, dtype=np.int32) * yres + bias[1]
    xmins = np.ones(max_label + 1, dtype=np.int32) * xres + bias[2]
    issample = np.zeros(max_label + 1, dtype=np.int32)
    sample = np.zeros((max_label + 1, 3), dtype=np.int32)
    neighborsample = np.zeros((max_label + 1, 3), dtype=np.int32)

    # iterate through the window
    for iz in range(zpoint - zradius, zpoint + zradius + 1):
        if iz < bias[0] or iz > zres + bias[0] - 1: continue
        for iy in range(ypoint - yradius, ypoint + yradius + 1):
            if iy < bias[1] or iy > yres + bias[1] - 1: continue
            for ix in range(xpoint - xradius, xpoint + xradius + 1):
                if ix < bias[2] or ix > xres + bias[2] - 1: continue
                # get the label for this location
                voxel_label = segmentation[iz - bias[0], iy - bias[1], ix - bias[2]]

                # skip over extracellular/unlabeled material
                if not args.overlap:
                    if voxel_label < label: continue
                else:
                    if not voxel_label: continue

                # update the adjacency matrix
                if iz < zres + bias[0] - 1 and voxel_label != segmentation[
                    iz + 1 - bias[0], iy - bias[1], ix - bias[2]]:
                    adjacency_matrix.add((voxel_label, segmentation[iz + 1 - bias[0], iy - bias[1], ix - bias[2]]))

                    # update mean affinities
                    if voxel_label == label or segmentation[iz + 1 - bias[0], iy - bias[1], ix - bias[2]] == label:
                        if voxel_label == label:
                            index = segmentation[iz + 1 - bias[0], iy - bias[1], ix - bias[2]]
                            if issample[index] == 0:
                                sample[index] = [iz - bias[0], iy - bias[1], ix - bias[2]]
                                neighborsample[index] = [iz + 1 - bias[0], iy - bias[1], ix - bias[2]]
                            issample[index] = 1
                        else:
                            index = voxel_label
                            if issample[index] == 0:
                                neighborsample[index] = [iz - bias[0], iy - bias[1], ix - bias[2]]
                                sample[index] = [iz + 1 - bias[0], iy - bias[1], ix - bias[2]]
                            issample[index] = 1

                        zmeans[index] += (iz + 0.5)
                        ymeans[index] += iy
                        xmeans[index] += ix
                        counts[index] += 1
                        zmaxs[index] = max(zmaxs[index], (iz + 1))
                        ymaxs[index] = max(ymaxs[index], iy)
                        xmaxs[index] = max(xmaxs[index], ix)
                        zmins[index] = min(zmins[index], (iz))
                        ymins[index] = min(ymins[index], iy)
                        xmins[index] = min(xmins[index], ix)

                if iy < yres + bias[1] - 1 and voxel_label != segmentation[
                    iz - bias[0], iy + 1 - bias[1], ix - bias[2]]:
                    adjacency_matrix.add((voxel_label, segmentation[iz - bias[0], iy + 1 - bias[1], ix - bias[2]]))

                    # update mean affinities
                    if voxel_label == label or segmentation[iz - bias[0], iy + 1 - bias[1], ix - bias[2]] == label:
                        if voxel_label == label:
                            index = segmentation[iz - bias[0], iy + 1 - bias[1], ix - bias[2]]
                            if issample[index] == 0:
                                sample[index] = [iz - bias[0], iy - bias[1], ix - bias[2]]
                                neighborsample[index] = [iz - bias[0], iy + 1 - bias[1], ix - bias[2]]
                            issample[index] = 1
                        else:
                            index = voxel_label
                            if issample[index] == 0:
                                neighborsample[index] = [iz - bias[0], iy - bias[1], ix - bias[2]]
                                sample[index] = [iz - bias[0], iy + 1 - bias[1], ix - bias[2]]
                            issample[index] = 1

                        zmeans[index] += iz
                        ymeans[index] += (iy + 0.5)
                        xmeans[index] += ix
                        counts[index] += 1
                        zmaxs[index] = max(zmaxs[index], iz)
                        ymaxs[index] = max(ymaxs[index], (iy + 1))
                        xmaxs[index] = max(xmaxs[index], ix)
                        zmins[index] = min(zmins[index], iz)
                        ymins[index] = min(ymins[index], (iy))
                        xmins[index] = min(xmins[index], ix)

                if ix < xres + bias[2] - 1 and voxel_label != segmentation[
                    iz - bias[0], iy - bias[1], ix + 1 - bias[2]]:
                    adjacency_matrix.add((voxel_label, segmentation[iz - bias[0], iy - bias[1], ix + 1 - bias[2]]))

                    # update mean affinities
                    if voxel_label == label or segmentation[iz - bias[0], iy - bias[1], ix + 1 - bias[2]] == label:
                        if voxel_label == label:
                            index = segmentation[iz - bias[0], iy - bias[1], ix + 1 - bias[2]]
                            if issample[index] == 0:
                                sample[index] = [iz - bias[0], iy - bias[1], ix - bias[2]]
                                neighborsample[index] = [iz - bias[0], iy - bias[1], ix + 1 - bias[2]]
                            issample[index] = 1
                        else:
                            index = voxel_label
                            if issample[index] == 0:
                                neighborsample[index] = [iz - bias[0], iy - bias[1], ix - bias[2]]
                                sample[index] = [iz - bias[0], iy - bias[1], ix + 1 - bias[2]]
                            issample[index] = 1

                        zmeans[index] += iz
                        ymeans[index] += iy
                        xmeans[index] += (ix + 0.5)
                        counts[index] += 1
                        zmaxs[index] = max(zmaxs[index], iz)
                        ymaxs[index] = max(ymaxs[index], iy)
                        xmaxs[index] = max(xmaxs[index], (ix + 1))
                        zmins[index] = min(zmins[index], iz)
                        ymins[index] = min(ymins[index], iy)
                        xmins[index] = min(xmins[index], ix)

                # skip points that belong to the same label
                # needs to be after adjacency lookup
                if voxel_label in labels_to_ignore: continue

                # find the distance between the center location and this one and skip if it is too far
                if backward:
                    zdiff = resolution[0] * (iz - zpoint) + radius * vector[0]
                    ydiff = resolution[1] * (iy - ypoint) + radius * vector[1]
                    xdiff = resolution[2] * (ix - xpoint) + radius * vector[2]
                else:
                    zdiff = resolution[0] * (iz - zpoint)
                    ydiff = resolution[1] * (iy - ypoint)
                    xdiff = resolution[2] * (ix - xpoint)

                if (abs(zdiff) + abs(ydiff) + abs(xdiff)) == 0:
                    vector_to_point = (0.0, 0.0, 0.0)
                else:
                    distance = math.sqrt(zdiff * zdiff + ydiff * ydiff + xdiff * xdiff)
                    if distance > maximum_distance: continue
                    # get a normalized vector between this point and the center
                    vector_to_point = (zdiff / distance, ydiff / distance, xdiff / distance)

                # get the distance between the two vectors
                dot_product = vector[0] * vector_to_point[0] + vector[1] * vector_to_point[1] + vector[2] * vector_to_point[2]

                # get the angle from the dot product
                if (dot_product < cos_theta): continue

                # add this angle to the list to inspect further and ignore it every other time
                labels_to_ignore.add(voxel_label)
                potential_neighbors.add(voxel_label)

    # only include potential neighbor labels that are locally adjacent
    neighbors = []
    means = []
    maxs = []
    mins = []
    cs = []
    samples = []
    neighborsamples = []

    for neighbor_label in potential_neighbors:
        # do not include background predictions
        if not neighbor_label: continue
        # make sure that the neighbor is locally adjacent and add to the set of edges
        if not (neighbor_label, label) in adjacency_matrix and not (label, neighbor_label) in adjacency_matrix: continue

        # return the mean as integer values and continue
        xx = int(zmeans[neighbor_label] / counts[neighbor_label]) * resize_para
        yy = int(ymeans[neighbor_label] / counts[neighbor_label]) * resize_para
        if x_shape_original * margin < xx < x_shape_original * (1 - margin):
            if y_shape_original * margin < yy < y_shape_original * (1 - margin):
                neighbors.append(neighbor_label)

                sample = (sample + [bias[0], bias[1], bias[2]]) * [resize_para, resize_para, 1]
                neighborsample = (neighborsample + [bias[0], bias[1], bias[2]]) * [resize_para, resize_para, 1]

                means.append([xx, yy, int(xmeans[neighbor_label] / counts[neighbor_label]) + 2])
                maxs.append([int(zmaxs[neighbor_label]) * resize_para, int(ymaxs[neighbor_label]) * resize_para, int(xmaxs[neighbor_label]) + 2])
                mins.append([int(zmins[neighbor_label]) * resize_para, int(ymins[neighbor_label]) * resize_para, int(xmins[neighbor_label]) + 2])
                cs.append(int(counts[neighbor_label]))
                samples.append(list(sample[neighbor_label]))
                neighborsamples.append(list(neighborsample[neighbor_label]))

    return neighbors, means, maxs, mins, cs, samples, neighborsamples


def process_skels_endpoints(sk):
    slist = []
    svs = skels[sk].vertices
    svr = skels[sk].radius
    ends = endpoints_vector[sk]
    for en in ends.keys():
        center = [int(svs[en][0]/anis[0]), int(svs[en][1]/anis[1]), int(svs[en][2]/anis[2])]
        vector = ends[en]
        radius = svr[en]
        neighbors, means, maxs, mins, cs, samples1, samples2 = traverse_endpoint(fields, center, vector, radius, anis, max_label, maximum_distance[0], maximum_radians, gliocyte5, bias, True)
        neighbors_n, means_n, maxs_n, mins_n, cs_n, samples1_n, samples2_n = traverse_endpoint(labels, center, vector, radius, anis, max_label, maximum_distance[1], maximum_radians, gliocyte5, bias, False)
        neighbors_nw, means_nw, maxs_nw, mins_nw, cs_nw, samples1_nw, samples2_nw = [], [], [], [], [], [], []
        neighbors_set = set(neighbors)
        for ne in range(len(neighbors_n)):
            if neighbors_n[ne] not in neighbors_set:
                neighbors_nw.append(neighbors_n[ne])
                means_nw.append(means_n[ne])
                maxs_nw.append(maxs_n[ne])
                mins_nw.append(mins_n[ne])
                cs_nw.append(cs_n[ne])
                samples1_nw.append(samples1_n[ne])
                samples2_nw.append(samples2_n[ne])

        if len(neighbors) + len(neighbors_nw) > 0:
            sdic = {}
            # sdic['center'] = center
            # sdic['vector'] = vector
            sdic['neighbors'] = neighbors + neighbors_nw
            sdic['means'] = means + means_nw
            sdic['maxs'] = maxs + maxs_nw
            sdic['mins'] = mins + mins_nw
            sdic['cs'] = cs + cs_nw
            sdic['samples1'] = samples1 + samples1_nw
            sdic['samples2'] = samples2 + samples2_nw
            sdic['id'] = ([1] * len(neighbors)) + ([0] * len(neighbors_nw))  # 1 - precise, 2 - sketchy
            slist.append(sdic)
    if len(slist) > 0:
        pickle.dump(slist, open(args.endsave + '/{:d}.pkl'.format(sk), 'wb'), protocol=4)


def minmaxposition(x, y, z, sk):
    img = labels_original[:, :, z - bias[2]]
    binary = np.where(img == sk, 1, 0).astype(np.uint8)
    binary = binary.copy()
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    x0, y0, w, h = 0, 0, 0, 0
    for i in range(len(contours)):
        flag = cv2.pointPolygonTest(contours[i], (y - bias[1], x - bias[0]), False)
        if flag >= 0:
            x0, y0, w, h = cv2.boundingRect(contours[i])  # for np.array
            break
    # print(y0, bias[0], h, x0, bias[1], w, x, y, z)
    assert y0 + bias[0] <= x + resize_para
    assert y0 + bias[0] + h >= x - resize_para
    assert x0 + bias[1] <= y + resize_para
    assert x0 + bias[1] + w >= y - resize_para
    assert w > 0 or h > 0
    return [y0 + bias[0], x0 + bias[1], z], [y0 + bias[0] + h, x0 + bias[1] + w, z]


def process_skels_branchpoints(sk):  # , lock, num
    p = int(args.mergepath/resize_para)
    skves = skels[sk].vertices
    skeds = skels[sk].edges
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
                next_sv = edges[sk][k][dire]
                for num in range(int(p)):
                    points.append(next_sv)
                    next_sv_list = deepcopy(edges[sk][next_sv])
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
                if len(points) == int(p):
                    dir += 1
                if len(points) >= int(p * 0.75):
                    normal += 1
            if dir >= 2 and normal >= 3:
                x = int(skves[k][0] / anis_original[0])
                y = int(skves[k][1] / anis_original[1])
                z = int(skves[k][2] / anis_original[2])
                if x_shape_original * margin < x < x_shape_original * (1 - margin):
                    if y_shape_original * margin < y < y_shape_original * (1 - margin):
                        min, max = minmaxposition(x, y, z, sk)
                        branchpoint.append({'position': [x, y, z+2], 'min': min, 'max': max, 'k': k})

    # branchpoint = [{'position': [int(skves[k][0]/anis[0]), int(skves[k][1]/anis[1]), int(skves[k][2]/anis[2])]}
    #                for k, f in nodedegree.items() if f >= 3]
    # print(sk, ':', len(branchpoint))

    l = len(branchpoint)
    if l > 0:
        if path.exists(args.branchsave + '{:d}.pkl'.format(sk)):
            bp = pickle.load(open(args.branchsave + '{:d}.pkl'.format(sk), 'rb'))
            branchpoint.extend(bp)
            pickle.dump(branchpoint, open(args.branchsave + '{:d}.pkl'.format(sk), 'wb'), protocol=4)
        else:
            pickle.dump(branchpoint, open(args.branchsave + '{:d}.pkl'.format(sk), 'wb'), protocol=4)

    # lock.acquire()
    # num.value += l
    # lock.release()

    return l


def process_skels_strangepoints(sk, lock, num):  # , lock, num
    svs = skels[sk].vertices
    ses = skels[sk].edges
    list0, list1 = ses[:, 0].tolist(), ses[:, 1].tolist()
    list_all = list0 + list1
    nodedegree = {}
    for key in list_all:
        nodedegree[key] = nodedegree.get(key, 0) + 1

    # strangepoint = []
    # for k, f in nodedegree.items():
    #     if f == 2:
    #         if direction(k, sk, svs):
    #             strangepoint.append(
    #             {'position': [int(svs[k][0]/anis[0]), int(svs[k][1]/anis[1]), int(svs[k][2]/anis[2])]})

    strangepoint = []
    mindirection = args.mindot  # (-1)-to-(1)
    for k, f in nodedegree.items():
        if f == 2:
            d = direction(k, sk, svs)
            if d < mindirection:
                mindirection = d
                x = int(svs[k][0] / anis_original[0])
                y = int(svs[k][1] / anis_original[1])
                z = int(svs[k][2] / anis_original[2])
                if x_shape_original * margin < x < x_shape_original * (1 - margin):
                    if y_shape_original * margin < y < y_shape_original * (1 - margin):
                        min, max = minmaxposition(x, y, z, sk)
                        if len(strangepoint) == 0:
                            strangepoint.append({'position': [x, y, z+2], 'min': min, 'max': max, 'k': k})
                        else:
                            strangepoint[0] = {'position': [x, y, z+2], 'min': min, 'max': max, 'k': k}

    # print(sk, ':', len(strangepoint))
    l = len(strangepoint)
    if l > 0:
        if path.exists(args.strangesave + '{:d}.pkl'.format(sk)):
            sp = pickle.load(open(args.strangesave + '{:d}.pkl'.format(sk), 'rb'))
            strangepoint.extend(sp)
            pickle.dump(strangepoint, open(args.strangesave + '{:d}.pkl'.format(sk), 'wb'), protocol=4)
        else:
            pickle.dump(strangepoint, open(args.strangesave + '{:d}.pkl'.format(sk), 'wb'), protocol=4)

    lock.acquire()
    num.value += l
    lock.release()
    # return len(strangepoint)


def direction(k, sk, svs):
    # a = edges[sk][k][0]
    # b = edges[sk][k][1]
    # norm_a = math.sqrt((svs[a][0] - svs[k][0]) ** 2 + (svs[a][1] - svs[k][1]) ** 2 + (svs[a][2] - svs[k][2]) ** 2)
    # vector_a = [(svs[a][0] - svs[k][0]) / norm_a, (svs[a][1] - svs[k][1]) / norm_a, (svs[a][2] - svs[k][2]) / norm_a]
    # norm_b = math.sqrt((svs[k][0] - svs[b][0]) ** 2 + (svs[k][1] - svs[b][1]) ** 2 + (svs[k][2] - svs[b][2]) ** 2)
    # vector_b = [(svs[k][0] - svs[b][0]) / norm_b, (svs[k][1] - svs[b][1]) / norm_b, (svs[k][2] - svs[b][2]) / norm_b]
    # if np.dot(np.array(vector_a), np.array(vector_b)) < -0.5:  # cos(60') = 0.5  cos(80') = 0.17 cos(90') = 0
    #     return True
    # else:
    #     return False

    p = int(args.mergepath/resize_para)
    points = []
    now_sv = k
    next_sv = edges[sk][k][0]
    for num in range(int(p)):
        points.append(next_sv)
        next_sv_list = deepcopy(edges[sk][next_sv])
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
    next_sv = edges[sk][k][1]
    for num in range(int(p)):
        points.append(next_sv)
        next_sv_list = deepcopy(edges[sk][next_sv])
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


def branstran_post(bs, sk):
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
            if a[i]-retain[-1] >= int(args.near/resize_para):
                retain.append(a[i])
                retain_seq.append(i)

    ed = skels[sk].edges
    G_edges = [tuple(i) for i in list(ed)]
    G = nx.Graph()
    G.add_edges_from(G_edges)

    eccentricity = []
    for itemnum in retain_seq:
        p_dict = nx.shortest_path_length(G, source=a[itemnum])
        maxkey = max(p_dict, key=p_dict.get)
        maxdistance = p_dict[maxkey]
        eccentricity.append(maxdistance)
    emax = max(eccentricity)
    emin = min(eccentricity)
    if emax == emin:
        score = [100] * len(eccentricity)
    else:
        score = [int(100 * (emax - eccentricity[i]) / (emax - emin)) for i in range(len(eccentricity))]

    radiu = skels[sk].radius
    voxel_num = radiu.shape[0]
    maxradiu = np.partition(radiu, int(0.75*voxel_num))[int(0.75*voxel_num)]

    post = [label_score(voxel_num, maxradiu)]
    j = 0
    for i in retain_seq:
        dict_i = deepcopy(bs[a_index[i]])
        dict_i.pop('k')
        dict_i['score'] = score[j]
        j += 1
        post.append(dict_i)

    return post


def label_score(a, b):
    return a * (b ** 2)


def feature_extraction_prediction(sk):
    pos = np.where(labels == sk)
    posnum = pos[0].shape[0]
    size = posnum * int(anis[2]/anis[0])
    limitation = int(args.dust/(resize_para**2))
    if num > limitation:
        epoch = math.ceil(num/limitation)
        pairs1 = []
        for i in range(epoch-1):
            pairs0 = np.random.choice(posnum, limitation, replace=False).tolist()
            pairs1.extend(pairs0)
        pairs0 = np.random.choice(posnum, num-limitation*(epoch-1), replace=False).tolist()
        pairs1.extend(pairs0)
    else:
        pairs1 = np.random.choice(posnum, num, replace=False).tolist()
    # maxxyz = max(np.max(pos[0]) - np.min(pos[0]) + 1, np.max(pos[1]) - np.min(pos[1]) + 1, np.max(pos[2]) - np.min(pos[2]) + 1)
    maxdx = np.max(pos[0]) - np.min(pos[0]) + 1
    maxdy = np.max(pos[1]) - np.min(pos[1]) + 1
    maxdz = np.max(pos[2]) - np.min(pos[2]) + 1
    # maxvolume = maxdx * maxdy * maxdz
    diameter = int((maxdx ** 2 + maxdy ** 2 + (maxdz * int(anis[2]/anis[0])) ** 2) ** 0.5)
    mark = 0
    hetero = set()
    for pair1 in pairs1:
        x1 = pos[0][pair1]
        y1 = pos[1][pair1]
        z1 = pos[2][pair1]
        x1min, x1max = max(int(x1 - (maxdx / 6)), 0), min(int(x1 + (maxdx / 6)) + 1, labels.shape[0])
        y1min, y1max = max(int(y1 - (maxdy / 6)), 0), min(int(y1 + (maxdy / 6)) + 1, labels.shape[1])
        z1min, z1max = max(int(z1 - (maxdz / 6)), 0), min(int(z1 + (maxdz / 6)) + 1, labels.shape[2])
        smalllabels = labels[x1min:x1max, y1min:y1max, z1min:z1max]
        smallpos = np.where(smalllabels == sk)
        smallposnum = smallpos[0].shape[0]
        pair2 = np.random.choice(smallposnum, 1, replace=False).tolist()
        x2 = smallpos[0][pair2[0]] + x1min
        y2 = smallpos[1][pair2[0]] + y1min
        z2 = smallpos[2][pair2[0]] + z1min
        x_middle = int((x1 + x2) / 2)
        y_middle = int((y1 + y2) / 2)
        z_middle = int((z1 + z2) / 2)
        if labels[x_middle, y_middle, z_middle] == sk:
            mark += 1
        else:
            hetero.add(labels[x_middle, y_middle, z_middle])
    hetero = len(hetero)
    features = np.array([mark, size, diameter, hetero]).reshape(1, -1)
    if features[0, 0] <= int(0.85 * num) and clf5.predict(features)[0] == 1:
        pickle.dump(sk, open(args.gliocytehighparasave + '{:d}.pkl'.format(sk), 'wb'), protocol=4)
    elif features[0, 0] <= int(0.85 * num) and clf1.predict(features)[0] == 1:
        pickle.dump(sk, open(args.gliocytelowparasave + '{:d}.pkl'.format(sk), 'wb'), protocol=4)


def o_time(s_time):
    print('*** Time: ', time.time() - s_time)


if __name__ == "__main__":
    s_time = time.time()

    seg_list = listdir(args.label)
    seg_list.sort()
    # print(seg_list)

    resize_para = args.resize

    im1 = io.imread(args.label + seg_list[0])
    x_shape_original, y_shape_original = int(im1.shape[1]), int(im1.shape[0])
    x_shape, y_shape, z_shape = int(im1.shape[1]/resize_para), int(im1.shape[0]/resize_para), len(seg_list)

    neighbors_num_total = 0
    branchs_num_total = 0
    stranges_num_total = 0
    branstran_num_total = 0

    x_num = 1
    y_num = 1
    z_num = 1
    z_first = z_shape  # 80  # z_shape  # 50

    for x_part in range(x_num):
        for y_part in range(y_num):
            for z_part in range(z_num):
                # label with shape (x, y, z) of numpy array, axis z is for layers
                labels = None
                labels_original = None
                if z_part < z_num-1:
                    for seg_loc in seg_list[z_first * z_part: z_first * (z_part + 1)]:
                        im = io.imread(args.label + seg_loc).astype(np.int32)
                        im = im.T
                        if labels_original is None:
                            labels_original = im[int(x_part*(x_shape_original/x_num)): int((x_part+1)*(x_shape_original/x_num)), int(y_part*(y_shape_original/y_num)): int((y_part+1)*(y_shape_original/y_num))]
                        else:
                            labels_original = np.dstack((labels_original, im[int(x_part*(x_shape_original/x_num)): int((x_part+1)*(x_shape_original/x_num)), int(y_part*(y_shape_original/y_num)): int((y_part+1)*(y_shape_original/y_num))]))
                        # im = transform.resize(im, (x_shape, y_shape), order=0, preserve_range=True).astype(np.uint32)
                        # im = io.imread(args.labelresize + seg_loc).astype(np.uint32)
                        # im = im.T
                        im = cv2.resize(im, (y_shape, x_shape), interpolation=cv2.INTER_NEAREST)
                        if labels is None:
                            labels = im[int(x_part*(x_shape/x_num)): int((x_part+1)*(x_shape/x_num)), int(y_part*(y_shape/y_num)): int((y_part+1)*(y_shape/y_num))]
                        else:
                            labels = np.dstack((labels, im[int(x_part*(x_shape/x_num)): int((x_part+1)*(x_shape/x_num)), int(y_part*(y_shape/y_num)): int((y_part+1)*(y_shape/y_num))]))
                    print('Label shape: ', labels.shape)
                    print('Label type: ', type(labels))
                else:
                    for seg_loc in tqdm(seg_list[z_first * z_part:]):
                        im = io.imread(args.label + seg_loc).astype(np.int32)
                        im = im.T
                        if labels_original is None:
                            labels_original = im[int(x_part*(x_shape_original/x_num)): int((x_part+1)*(x_shape_original/x_num)), int(y_part*(y_shape_original/y_num)): int((y_part+1)*(y_shape_original/y_num))]
                        else:
                            labels_original = np.dstack((labels_original, im[int(x_part*(x_shape_original/x_num)): int((x_part+1)*(x_shape_original/x_num)), int(y_part*(y_shape_original/y_num)): int((y_part+1)*(y_shape_original/y_num))]))
                        # im = transform.resize(im, (x_shape, y_shape), order=0, preserve_range=True).astype(np.uint32)
                        # import matplotlib.pyplot as plt
                        # plt.imshow(im, cmap='gray')
                        # plt.show()
                        # im = io.imread(args.labelresize + seg_loc).astype(np.uint32)
                        # im = im.T
                        im = cv2.resize(im, (y_shape, x_shape), interpolation=cv2.INTER_NEAREST)
                        if labels is None:
                            labels = im[int(x_part*(x_shape/x_num)): int((x_part+1)*(x_shape/x_num)), int(y_part*(y_shape/y_num)): int((y_part+1)*(y_shape/y_num))]
                        else:
                            labels = np.dstack((labels, im[int(x_part*(x_shape/x_num)): int((x_part+1)*(x_shape/x_num)), int(y_part*(y_shape/y_num)): int((y_part+1)*(y_shape/y_num))]))
                    print('Label shape: ', labels.shape)
                    print('Label type: ', type(labels))
                # pickle.dump(labels, open('labels.pkl', 'wb'), protocol=4)

                # 设置超参数，其中anis分别为xyz尺度
                margin = args.margin
                process_para = args.core  # process cores
                endpoint_para = int(args.path/resize_para)  # trace path for endpoint vector
                maximum_distance = args.distance  # find distance for endpoint
                maximum_radians = args.angle  # find theta for endpoint
                anis_original = args.anis  # anisotropy ratio (original)
                anis = (anis_original[0] * resize_para, anis_original[1] * resize_para, anis_original[2])  # anisotropy ratio (actual)
                assert anis[0] == anis[1]
                assert anis[2] % anis[0] == 0

                bi = (x_part * (x_shape/x_num), y_part * (y_shape/y_num), z_part * z_first + int(seg_list[0].split(".")[0]))
                zipped = zip(args.bias, bi)
                mapped = map(sum, zipped)
                bias = list(mapped)
                bias = tuple([int(i) for i in bias])
                o_time(s_time)

                # 骨架化
                skels = kimimaro.skeletonize(
                    labels,
                    #   teasar_params={
                    #     'scale': 4,
                    #     'const': 500, # physical units
                    #     'pdrf_exponent': 4,
                    #     'pdrf_scale': 100000,
                    #     'soma_detection_threshold': 1100, # physical units
                    #     'soma_acceptance_threshold': 3500, # physical units
                    #     'soma_invalidation_scale': 1.0,
                    #     'soma_invalidation_const': 300, # physical units
                    #     'max_paths': 50, # default None
                    #   },
                    # object_ids=[10434, 10598, 10782, 10810, 11323, 14653, 14695, 14972, 15283, \
                    #     16532, 2268, 397, 6003, 6500, 6725, 9826], # process only the specified labels
                    # extra_targets_before=[ (27,33,100), (44,45,46) ], # target points in voxels
                    # extra_targets_after=[ (27,33,100), (44,45,46) ], # target points in voxels
                    dust_threshold=int(args.dust/(resize_para**2)),  # skip connected components with fewer than this many voxels
                    anisotropy=anis,  # default True
                    fix_branching=True,  # default True
                    fix_borders=True,  # default True
                    fill_holes=False,  # default False
                    fix_avocados=False,  # default False
                    progress=True,  # default False, show progress bar
                    parallel=process_para,  # <= 0 all cpu, 1 single process, 2+ multiprocess
                    parallel_chunk_size=500,  # how many skeletons to process before updating progress bar
                )

                # num = args.randomsample
                # clf1 = pickle.load(open('randomforest-1.pkl', 'rb'))
                # clf5 = pickle.load(open('randomforest-5.pkl', 'rb'))
                # if path.exists(args.gliocytehighparasave):
                #     shutil.rmtree(args.gliocytehighparasave)
                # mkdir(args.gliocytehighparasave)
                # if path.exists(args.gliocytelowparasave):
                #     shutil.rmtree(args.gliocytelowparasave)
                # mkdir(args.gliocytelowparasave)
                # with Pool(processes=process_para) as pool:  # process_para
                #     pool.map(feature_extraction_prediction, skels.keys())

                # gliocyte5 = []
                # gliocyte1 = []
                # high_list = listdir(args.gliocytehighparasave)
                # low_list = listdir(args.gliocytelowparasave)
                # for high in high_list:
                #     gliocyte5.append(int(high.split(".")[0]))
                # for low in low_list:
                #     gliocyte1.append(int(low.split(".")[0]))
                # shutil.rmtree(args.gliocytehighparasave)
                # shutil.rmtree(args.gliocytelowparasave)
                # print('gliocyte-5:', len(gliocyte5), gliocyte5)
                # print('gliocyte-1:', len(gliocyte1), gliocyte1)
                # pickle.dump(gliocyte5, open('gliocyte5.pkl', 'wb'), protocol=4)
                # pickle.dump(gliocyte1, open('gliocyte1.pkl', 'wb'), protocol=4)

                # for sk in gliocyte5:
                #     skels.pop(sk)
                gliocyte5 = []

                print('Extract skels number:', len(skels))
                o_time(s_time)

                for sk in skels.keys():
                    skels[sk].vertices = skels[sk].vertices + [anis[0] * bias[0], anis[1] * bias[1], anis[2] * bias[2]]

                # print('Total skels: ', len(skels.keys()))  # 2515
                # print('Max skels label: ', max(skels.keys()))  # 6075
                # print('Min skels label: ', min(skels.keys()))  # 1
                # print(skels[min(skels.keys())])
                # print(skels[min(skels.keys())].vertices)
                # print(skels[min(skels.keys())].edges)
                # print(skels[min(skels.keys())].radius)
                # print(skels[min(skels.keys())].vertices.shape)  # (361, 3)
                # print(skels[min(skels.keys())].edges.shape)  # (360, 2)
                # print(skels[min(skels.keys())].radius.shape)  # (361,)
                # skels[295].viewer(draw_vertices=True, draw_edges=True, units='nm', color_by='radius')
                # pickle.dump(skels, open('skeletons_dic.pkl', 'wb'), protocol=4)
                # skels = pickle.load(open('skeletons_dic.pkl', 'rb'))

                # edges便于调用骨架的连接性
                edges = {}
                # fields将骨架存为点云形式
                fields = np.zeros(labels.shape, dtype=np.int32)
                for sk in skels.keys():
                    svs = skels[sk].vertices
                    for sv in range(svs.shape[0]):
                        sn = svs[sv]
                        fields[int(sn[0]/anis[0]-bias[0]/resize_para), int(sn[1]/anis[1]-bias[1]/resize_para), int(sn[2]/anis[2]-bias[2])] = sk
                    ses = skels[sk].edges
                    edges[sk] = {}
                    for se in range(ses.shape[0]):
                        f = ses[se, 0]
                        t = ses[se, 1]
                        if f in edges[sk]:
                            edges[sk][f].append(t)
                        else:
                            edges[sk][f] = [t]
                        if t in edges[sk]:
                            edges[sk][t].append(f)
                        else:
                            edges[sk][t] = [f]
                o_time(s_time)
                # 这里存好之后可以调用，也可以不存
                imageio.volwrite('fields.tif', np.transpose(fields, (2, 1, 0)), bigtiff=True)
                # pickle.dump(edges, open('edges.pkl', 'wb'), protocol=4)
                # edges = pickle.load(open('edges.pkl', 'rb'))

                if path.exists(args.branchsave):
                    shutil.rmtree(args.branchsave)
                mkdir(args.branchsave)
                # 多进程实现分叉点
                # manager = Manager()
                # num = manager.Value('tmp', 0)
                # lock = manager.Lock()
                # partial_process_skels_branchpoints = partial(process_skels_branchpoints, lock=lock, num=num)
                # with Pool(processes=1) as pool:  # process_para
                #     pool.map(partial_process_skels_branchpoints, skels.keys())
                # print('Branching point number:', num.value)  # 269 from 144
                # branchs_num_total += num.value
                # o_time(s_time)

                # 单进程实现分叉点
                branchs_num = 0
                for l in skels.keys():
                    num = process_skels_branchpoints(l)
                    branchs_num += num
                print('Branching point number:', branchs_num)
                branchs_num_total += branchs_num
                o_time(s_time)

                if path.exists(args.strangesave):
                    shutil.rmtree(args.strangesave)
                mkdir(args.strangesave)
                # 多进程实现奇异点
                manager = Manager()
                num = manager.Value('tmp', 0)
                lock = manager.Lock()
                partial_process_skels_strangepoints = partial(process_skels_strangepoints, lock=lock, num=num)
                with Pool(processes=process_para) as pool:
                    pool.map(partial_process_skels_strangepoints, skels.keys())
                print('Strange point number:', num.value)
                stranges_num_total += num.value
                o_time(s_time)

                # 单进程实现奇异点
                # # process_skels_strangepoints(9089)
                # stranges_num = 0
                # for l in skels.keys():
                #     num = process_skels_strangepoints(l)
                #     stranges_num += num
                # print('Strange point number:', stranges_num)
                # stranges_num_total += stranges_num
                # o_time(s_time)

                # '''
                # 读取分叉点+奇异点，整合成一个pkl
                branstran = {}
                bran_list = listdir(args.branchsave)
                stran_list = listdir(args.strangesave)
                branstran_list = list(set(bran_list + stran_list))
                for branstran_item in branstran_list:
                    sk = int(branstran_item.split('.')[0])
                    if path.exists(args.branchsave + branstran_item):
                        bran = pickle.load(open(args.branchsave + branstran_item, 'rb'))
                    else:
                        bran = []
                    if path.exists(args.strangesave + branstran_item):
                        stran = pickle.load(open(args.strangesave + branstran_item, 'rb'))
                    else:
                        stran = []
                    bs = bran + stran
                    bs_post = branstran_post(bs, sk)
                    branstran[sk] = bs_post
                s = [i[1][0] for i in branstran.items()]
                slen = len(s)
                # smax = max(s)
                smax = np.partition(np.array(s), int(0.8 * slen))[int(0.8 * slen)]
                # smin = min(s)
                smin = np.partition(np.array(s), int(0.2 * slen))[int(0.2 * slen)]
                branstran_num = 0
                for i in branstran.keys():
                    branstran[i][0] = max(min(int(100 * (branstran[i][0] - smin) / (smax - smin)), 100), 0)
                    branstran_num += len(branstran[i]) - 1
                branstran_num_total += branstran_num
                print('Branching and strange point number:', branstran_num)
                pickle.dump(branstran, open('branstran_dic.pkl', 'wb'), protocol=4)
                # print(branstran[1316])
                # shutil.rmtree(args.branchsave)
                # shutil.rmtree(args.strangesave)
                o_time(s_time)
                # '''

                # 计算端点及端点处的vector
                endpoints = {}
                endpoints_vector = {}
                for sk in skels.keys():
                    endpoints[sk] = {}
                    endpoints_vector[sk] = {}
                    svs = skels[sk].vertices
                    for sv in range(svs.shape[0]):
                        if len(edges[sk][sv]) == 1:
                            endpoints[sk][sv] = [sv]
                            now_sv = sv
                            for num in range(endpoint_para):
                                next_sv = edges[sk][now_sv][0]
                                next_sv_list = edges[sk][next_sv]
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
                                # 此处修改为了：回溯点集的中心作为vector的起点
                                s = svs[endpoints[sk][sv][1]]
                                for m in range(2, len(endpoints[sk][sv])):
                                    s += svs[endpoints[sk][sv][m]]
                                s /= len(endpoints[sk][sv]) - 1
                                norm = math.sqrt((e[0] - s[0]) * (e[0] - s[0]) + (e[1] - s[1]) * (e[1] - s[1]) + (e[2] - s[2]) * (e[2] - s[2]))
                                endpoints_vector[sk][sv] = [(e[0] - s[0]) / norm, (e[1] - s[1]) / norm, (e[2] - s[2]) / norm]

                # 这里存好之后可以调用，也可以不存
                # pickle.dump(endpoints, open('endpoints.pkl', 'wb'), protocol=4)
                # pickle.dump(endpoints_vector, open('endpoints_vector.pkl', 'wb'), protocol=4)
                # endpoints_vector = pickle.load(open('endpoints_vector.pkl', 'rb'))
                o_time(s_time)

                # 多进程调用函数，并将中间结果分开存储至endsave路径下
                max_label = np.max(labels)
                if path.exists(args.endsave):
                    shutil.rmtree(args.endsave)
                mkdir(args.endsave)
                with Pool(processes=process_para) as pool:
                    pool.map(process_skels_endpoints, skels.keys())  # 'process_skels_endpoints' contain max_label
                o_time(s_time)

                # 读取所有结果，整合成一个pkl
                ans_dic = {}
                ans_list = listdir(args.endsave)
                ans_sk_list = []
                for ans_item in ans_list:
                    sk = int(ans_item.split('.')[0])
                    ans_sk_list.append(sk)
                    slist = pickle.load(open(args.endsave + ans_item, 'rb'))
                    ans_dic[sk] = slist
                shutil.rmtree(args.endsave)
                # pickle.dump(ans_dic, open('ans_dic.pkl', 'wb'), protocol=4)
                o_time(s_time)

                # 整理所得结果，输出为一个pkl
                neighbors_num = 0
                touch_dic = {}
                touch_dic_sketchy = {}
                means_already = []
                for sk in ans_sk_list:  # label
                    for ce in range(len(ans_dic[sk])):  # endpoint
                        # center = ans_dic[sk][ce]['center']  # endpoint location
                        # vector = ans_dic[sk][ce]['vector']  # endpoint direction
                        neis = ans_dic[sk][ce]['neighbors']  # endpoint's neighbors' label
                        means = ans_dic[sk][ce]['means']
                        maxs = ans_dic[sk][ce]['maxs']
                        mins = ans_dic[sk][ce]['mins']
                        cs = ans_dic[sk][ce]['cs']
                        samples1 = ans_dic[sk][ce]['samples1']
                        samples2 = ans_dic[sk][ce]['samples2']
                        touch_id = ans_dic[sk][ce]['id']
                        # print('len(neis):', len(neis))
                        for ne in range(len(neis)):  # endpoint's neighbors
                            if touch_id[ne] == 1:
                                if (sk, neis[ne]) in touch_dic.keys():
                                    touch_dic[(sk, neis[ne])].append([maxs[ne], mins[ne], samples1[ne], samples2[ne], means[ne], cs[ne]])
                                else:
                                    touch_dic[(sk, neis[ne])] = [[maxs[ne], mins[ne], samples1[ne], samples2[ne], means[ne], cs[ne]]]
                            elif touch_id[ne] == 0:
                                if (sk, neis[ne]) in touch_dic_sketchy.keys():
                                    touch_dic_sketchy[(sk, neis[ne])].append([maxs[ne], mins[ne], samples1[ne], samples2[ne], means[ne], cs[ne]])
                                else:
                                    touch_dic_sketchy[(sk, neis[ne])] = [[maxs[ne], mins[ne], samples1[ne], samples2[ne], means[ne], cs[ne]]]
                            else:
                                print('Warning: read and save touch_dic.pkl error!')
                            neighbors_num += 1
                neighbors_num_total += neighbors_num
                print('Total touch neighbors (part): ', neighbors_num)
                # dict[(label1, label2)] = [np.array[[接触max边界], [接触min边界], [label1采样点], [label2采样点], [接触中心], 接触数], ...]
                if path.exists('touch_dic.pkl'):
                    apd = pickle.load(open('touch_dic.pkl', 'rb'))  # <class 'dict'>
                    touch_dic.update(apd)
                    pickle.dump(touch_dic, open('touch_dic.pkl', 'wb'), protocol=4)
                else:
                    pickle.dump(touch_dic, open('touch_dic.pkl', 'wb'), protocol=4)
                if path.exists('touch_dic_sketchy.pkl'):
                    apd = pickle.load(open('touch_dic_sketchy.pkl', 'rb'))  # <class 'dict'>
                    touch_dic_sketchy.update(apd)
                    pickle.dump(touch_dic_sketchy, open('touch_dic_sketchy.pkl', 'wb'), protocol=4)
                else:
                    pickle.dump(touch_dic_sketchy, open('touch_dic_sketchy.pkl', 'wb'), protocol=4)
                o_time(s_time)

    print('Total branchpoints (all): ', branchs_num_total)
    print('Total strangepoints (all): ', stranges_num_total)
    print('Total branchs-stranges (all): ', branstran_num_total)
    print('Total touch neighbors (all): ', neighbors_num_total)
