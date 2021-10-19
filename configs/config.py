from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from yacs.config import CfgNode as CN

__C = CN()

cfg = __C

__C.VERSION = 'Default version description'

# Core number used for kimimaro and other multiprocessing
__C.CORE = 24


# ------------------------------------------------------------------------ #
# Base options
# ------------------------------------------------------------------------ #

# Input scale options
__C.ANIS = [5, 5, 40]  # anisotropy
__C.CUTS = [4000, 4000, 209]  # pixel size w/o overlap
__C.RESZ = 8  # resize factor of projection in skel space

# Output scale options
__C.OANIS = [5, 5, 40]  # anisotropy
__C.OBIAS = [0, 0, 0]  # bias for ouput pixel position


# ------------------------------------------------------------------------ #
# Skeleton options (kimimaro from seung-lab)
# ------------------------------------------------------------------------ #
__C.SKEL = CN()

# Skeleton description
__C.SKEL.TYPE = 'Default description'

# Chunk size (how many skeletons to process before updating progress bar)
__C.SKEL.CHUK = 2400

# Edge length of the cubic 'physical' volume
# (skip connected components with fewer than sth.)
__C.SKEL.DUST = 342

# Maximum 'physical' radius when merge close compoents 
__C.SKEL.JORA = 1600

__C.SKEL.KWARGS = CN(new_allowed=True)


# ------------------------------------------------------------------------ #
# Amputate error options (2p, pre-processing from ZhangYC)
# ------------------------------------------------------------------------ #
__C.AMPUTATE = CN()

# Amputate description
__C.AMPUTATE.TYPE = 'Default description'

__C.AMPUTATE.KWARGS = CN(new_allowed=True)


# ------------------------------------------------------------------------ #
# Split error options (2p, from ZhaiH)
# ------------------------------------------------------------------------ #
__C.SPLIT = CN()

# Split description
__C.SPLIT.TYPE = 'Default description'

# K-neighborhood maximum '' distance
__C.SPLIT.SPLD = 800

# K-neighborhood the closest points maximum indice
__C.SPLIT.SPLN = 3

# Trace pixel length in skel space to calculate the vector
__C.SPLIT.SPVE = 32

# Two vectors should not parallel to x-y-plane too much  
# (1.5r=85.94d)
__C.SPLIT.SPPA = 1.5

# Radian of two vectors seems to be merged
# (0.3216r=18.4d, 0.45r=25.8d)
__C.SPLIT.SPRA = 0.3216

# Trace pixels overlap in z-axis should not too big
__C.SPLIT.SPOV = 0.5

__C.SPLIT.KWARGS = CN(new_allowed=True)


# ------------------------------------------------------------------------ #
# Divide error options (1p, branch/strange from LiZC)
# ------------------------------------------------------------------------ #
__C.DIVIDE = CN()

# Divide description
__C.DIVIDE.TYPE = 'Default description'

# Trace pixel length in skel space to calculate the vector
__C.DIVIDE.BRVE = 32

# Rate of the first / second longest branch
__C.DIVIDE.BRP1 = 1.0
__C.DIVIDE.BRP2 = 0.75

# Strage angles are above this degree
# (1.58r=90d)
__C.DIVIDE.STRA = 1.58

# Near points-pair need to be deleted one of them
__C.DIVIDE.DIVD = 12

# Margin rate for point to be deleted 
# (because skel at margin are not accurate)
__C.DIVIDE.MARGIN = 0.025

__C.DIVIDE.KWARGS = CN(new_allowed=True)


# ------------------------------------------------------------------------ #
# Merge error options (1p, from ZhaiH)
# ------------------------------------------------------------------------ #
__C.MERGE = CN()

# Merge description
__C.MERGE.TYPE = 'Default description'

# Near weak aff points-pair need to be joined
__C.MERGE.JOSK = 80

# The intensity threshold to be defined as weak aff points
__C.MERGE.AFFT = 100

# The member number of groups threshold to be defined as weak aff group
__C.MERGE.AFFN = 4

__C.MERGE.KWARGS = CN(new_allowed=True)
