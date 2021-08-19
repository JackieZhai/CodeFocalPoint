import numpy as np
import struct
import os


def read_binary_dat(merge):
    
    # 以二进制读取数组
    with open(merge, 'rb') as f:
        bite = struct.unpack('i', f.read(4))
        num_z = struct.unpack('i', f.read(4))
        num_y = struct.unpack('i', f.read(4))
        num_x = struct.unpack('i', f.read(4))
        overloap_z_pixel = struct.unpack('i', f.read(4))
        overloap_y_pixel = struct.unpack('i', f.read(4))
        overloap_x_pixel = struct.unpack('i', f.read(4))
        size_z = struct.unpack('i', f.read(4))
        size_y = struct.unpack('i', f.read(4))
        size_x = struct.unpack('i', f.read(4))
        height = struct.unpack('i', f.read(4))
        hang = struct.unpack('i', f.read(4))
        line = struct.unpack('i', f.read(4))

        label = np.zeros((int(num_z[0]), int(num_y[0]), int(num_x[0]), 2 ** int(bite[0])), dtype=np.uint32)

        arr_read = struct.unpack('{}i'.format((int(num_x[0]) * int(num_y[0]) * int(num_z[0])) * (2 ** int(bite[0]))), f.read())  # 这里unpack的第一个参数表示有多少个int类型的数据
        arr_read = np.array(list(arr_read))
        arr_read = arr_read.reshape(label.shape)

    # arr_message, arr_read
    return [bite[0], num_z[0], num_y[0], num_x[0], overloap_z_pixel[0], overloap_y_pixel[0], \
        overloap_x_pixel[0], size_z[0], size_y[0], size_x[0]], arr_read


def read_file_system(load_path):
    
    sub_data_path = os.path.join(load_path)
    sub_tifs = os.listdir(sub_data_path)

    total = int(len(sub_tifs))  # total 应该是整数才对

    sub_tifs.sort(key=lambda x: int(x[0:2]) * 100 + int(x[2:4]) * 1)
    sorted_file = sub_tifs

    # 计算 num_x, num_y, num_z
    height_all = np.zeros(total)
    row_all = np.zeros(total)
    line_all = np.zeros(total)

    for i, name in enumerate(sorted_file):
        height_all[i] = 0
    for i, name in enumerate(sorted_file):
        row_all[i] = int(name[0:2])
    for i, name in enumerate(sorted_file):
        line_all[i] = int(name[2:4])

    number_z = np.unique(height_all)
    number_y = np.unique(row_all)
    number_x = np.unique(line_all)

    num_z = len(number_z)
    num_y = len(number_y)
    num_x = len(number_x)

    all_z = number_z.tolist()
    FilenameAray_xy = []
    FilenameAray = []
    for j in range(num_z):
        FilenameAray_xy = []
        for i in range(num_y):
            zeroArray = ['0' for i in range(num_x)]
            FilenameAray_xy.append(zeroArray)
        FilenameAray.append(FilenameAray_xy)

    min_y = int(np.min(row_all))
    min_x = int(np.min(line_all))

    del row_all, line_all, FilenameAray_xy, number_y, number_x

    for name in (sorted_file):
        name_row = int(name[0:2])
        name_line = int(name[2:4])
        name_height = 0

        z = all_z.index(name_height)
        y = name_row - min_y
        x = name_line - min_x
        FilenameAray[z][y][x] = name
    
    return FilenameAray


###
# [0, 13, 0] = 1601
# [0, 14, 0] = 1701
###