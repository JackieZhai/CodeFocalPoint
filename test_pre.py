import imageio
import numpy as np
import h5py

from pathlib import Path

root_dir = Path('/dc1/SCN/wafer14/seg/')

# im1601 = imageio.volread('1601.tif')
# print(im1601.shape, im1601.dtype)
# im1601[im1601 == 255] = 0
# im1601[im1601 == 1241] = 255
# im1601[im1601 != 255] = 0
# im1601.astype(np.uint8)
# imageio.volwrite('1601_target.tif', im1601)

# im1701 = imageio.volread('1701.tif')
# print(im1701.shape, im1701.dtype)
# im1701[im1701 == 255] = 0
# im1701[im1701 == 181] = 255
# im1701[im1701 == 258] = 255
# im1701[im1701 == 1114] = 255
# im1701[im1701 != 255] = 0
# im1701.astype(np.uint8)
# imageio.volwrite('1701_target.tif', im1701)

# im1801 = imageio.volread('1801.tif')
# print(im1801.shape, im1801.dtype)
# im1801[im1801 == 255] = 0
# im1801[im1801 == 12213] = 255
# im1801[im1801 == 12665] = 255
# im1801[im1801 == 2515] = 255
# im1801[im1801 == 291] = 255
# im1801[im1801 == 6828] = 255
# im1801[im1801 == 8254] = 255
# im1801[im1801 == 8409] = 255
# im1801[im1801 == 9389] = 255
# im1801[im1801 != 255] = 0
# im1801.astype(np.uint8)
# imageio.volwrite('1801_target.tif', im1801)

# im1901 = imageio.volread('1901.tif')
# print(im1901.shape, im1901.dtype)
# im1901[im1901 == 255] = 0
# im1901[im1901 == 10434] = 255
# im1901[im1901 == 10598] = 255
# im1901[im1901 == 10782] = 255
# im1901[im1901 == 10810] = 255
# im1901[im1901 == 11323] = 255
# im1901[im1901 == 14653] = 255
# im1901[im1901 == 14695] = 255
# im1901[im1901 == 14972] = 255
# im1901[im1901 == 15283] = 255
# im1901[im1901 == 16532] = 255
# im1901[im1901 == 2268] = 255
# im1901[im1901 == 397] = 255
# im1901[im1901 == 6003] = 255
# im1901[im1901 == 6500] = 255
# im1901[im1901 == 6725] = 255
# im1901[im1901 == 9826] = 255
# im1901[im1901 != 255] = 0
# im1901.astype(np.uint8)
# imageio.volwrite('1901_target.tif', im1901)

# im = imageio.volread(root_dir.joinpath('test_fields.tif'))
# im[im > 0] = 255
# im = im.astype(np.uint8)
# im_e = 255 * np.ones((1, im.shape[1], im.shape[2]), dtype=im.dtype)
# im = np.vstack((im, im_e))
# print(im.shape)
# imageio.volwrite(root_dir.joinpath('test_fields_target.tif'), im)

# im = imageio.volread(root_dir.joinpath('1901_target_resize_4.tif'))
# im_e = 255 * np.ones((1, im.shape[1], im.shape[2]), dtype=im.dtype)
# im = np.vstack((im, im_e))
# print(im.shape)
# imageio.volwrite(root_dir.joinpath('1901_target_resize_4.tif'), im)

# im = imageio.volread(root_dir.joinpath('1901_resize_4.tif'))
# im[im > 0] = 255
# im = im.astype(np.uint8)
# im_e = 255 * np.ones((1, im.shape[1], im.shape[2]), dtype=im.dtype)
# im = np.vstack((im, im_e))
# print(im.shape)
# imageio.volwrite(root_dir.joinpath('1901_resize_4.tif'), im)

name = '2001'
f = h5py.File('/dc1/SCN/wafer14/affnity/elastic_' + name + '.h5', 'r')
fd = f['vol0'][:]
print(fd.shape, fd.dtype, type(fd))
# imageio.volwrite(root_dir.joinpath(name + '_aff_z.tif'), fd[0])
# imageio.volwrite(root_dir.joinpath(name + '_aff_y.tif'), fd[1])
# imageio.volwrite(root_dir.joinpath(name + '_aff_x.tif'), fd[2])
imageio.volwrite(root_dir.joinpath(name + '_aff.tif'), np.mean(fd, axis=0).astype(np.uint8))

