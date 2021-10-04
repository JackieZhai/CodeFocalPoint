import numpy as np


def _make_range(a, b, n=None):
    if a == b:
        raise Exception
    if n is not None:
        if a < b:
            return range(a, a+n)
        else:
            return range(a, a-n, -1)
    else:
        if a < b:
            return range(a, b)
        else:
            return range(a, b, -1)


# aff, v1, v2 are axis from stack [0,0,0]
def affinity_in_skel(aff, v1, v2, resz=4):
    assert resz == 4, '--- Resize error: need to redesign this func'
    v1 = np.array(v1)
    v2 = np.array(v2)
    aff_list = []
    affsz = aff.shape[1:4]

    def affz(x, y, z):
        try:
            aff_list.append(aff[0][x, y, z])
        except:
            pass  # at 0 or affsz//resz-1
    def affy(x, y, z):
        try:
            aff_list.append(aff[1][x, y, z])
        except:
            pass  # at 0 or affsz//resz-1
    def affx(x, y, z):
        try:
            aff_list.append(aff[2][x, y, z])
        except:
            pass  # at 0 or affsz//resz-1

    dx, dy, dz = v2 - v1
    # 26-connected
    try:
        assert -1<=dx<=1 and -1<=dy<=1 and -1<=dz<=1 \
            and (not (dx==0 and dy==0 and dz==0))
    except:
        return 0
        # TODO: Maybe caused by 'kimimaro.join_close_components()'
        # Need to find solution for these step connections
    try:
        assert 0<=v1[0]<=affsz[0]//resz-1 and 0<=v1[1]<=affsz[1]//resz-1 \
            and 0<=v1[2]<=affsz[2]-1
        assert 0<=v2[0]<=affsz[0]//resz-1 and 0<=v2[1]<=affsz[1]//resz-1 \
            and 0<=v2[2]<=affsz[2]-1
    except:
        return 0
        # TODO: Caued by 'overlap' from 2000 to 2200
        # Need to find solution for these border pixels
        # Solution 1: on 0 and affsz//resz-1, it's indispensable to get full 12
        # connections, just minimize them (even < 12)
    
    v1[0:2] = v1[0:2] * resz + resz // 2
    v2[0:2] = v2[0:2] * resz + resz // 2



    if dz == 0:
        if dy!=0 and dx!=0:
            # dy,dx (4)
            # 0 0 0 0 0 0 0 0
            # 0 0 0 0 0 0 0 0
            # 0 0 2 2 1 0 0 0
            # 0 0 2 2 1 0 0 0
            # 0 0 1 1 0 0 0 0
            # 0 0 0 0 0 0 0 0
            # 0 0 0 0 0 0 0 0
            # 0 0 0 0 0 0 0 0
            if dx>0:
                for px in _make_range(v1[0], v2[0], 2):
                    affx(px, v1[1], v1[2])
                    affx(px, v1[1]+1, v1[2])
                    affx(px, v1[1]+2, v1[2])
            else:
                for px in _make_range(v1[0]-1, v2[0], 2):
                    affx(px, v1[1], v1[2])
                    affx(px, v1[1]+1, v1[2])
                    affx(px, v1[1]+2, v1[2])
            if dy>0:
                for py in _make_range(v1[1], v2[1], 2):
                    affy(v1[0], py, v1[2])
                    affy(v1[0]+1, py, v1[2])
                    affy(v1[0]+2, py, v1[2])
            else:
                for py in _make_range(v1[1]-1, v2[1], 2):
                    affy(v1[0], py, v1[2])
                    affy(v1[0]+1, py, v1[2])
                    affy(v1[0]+2, py, v1[2])
        elif dx!=0:
            # dy | dx (4)
            # 0 0 1 1 1 0 0 0
            # 0 0 1 1 1 0 0 0
            # 0 0 1 1 1 0 0 0
            # 0 0 1 1 1 0 0 0
            for px in _make_range(v1[0], v2[0], 3):
                affx(px, v1[1]-2, v1[2])
                affx(px, v1[1]-1, v1[2])
                affx(px, v1[1], v1[2])
                affx(px, v1[1]+1, v1[2])
        else:
            for py in _make_range(v1[1], v2[1], 3):
                affy(v1[0]-2, py, v1[2])
                affy(v1[0]-1, py, v1[2])
                affy(v1[0], py, v1[2])
                affy(v1[0]+1, py, v1[2])
    else:
        if dy!=0 and dx!=0:
            # dz,dy,dx (8)
            # 0 0 0 0
            # 0 0 0 0
            # 0 0 0 0
            # 0 0 0 4
            # 0 0 0 4 4 0 0 0
            # 0 0 0 0 0 0 0 0
            # 0 0 0 0 0 0 0 0
            # 0 0 0 0 0 0 0 0
            if dx>0 and dy>0:
                affx(v1[0]+1, v1[1]+1, v1[2])
                affy(v1[0]+1, v1[1]+1, v1[2])
                affz(v1[0]+1, v1[1]+1, v1[2])
                affy(v1[0]+2, v1[1]+1, v1[2])
                affz(v1[0]+2, v1[1]+1, v1[2])
                affx(v1[0]+1, v1[1]+2, v1[2])
                affz(v1[0]+1, v1[1]+2, v1[2])
                affz(v1[0]+2, v1[1]+2, v1[2])
                affx(v1[0]+1, v1[1]+1, v2[2])
                affy(v1[0]+1, v1[1]+1, v2[2])
                affy(v1[0]+2, v1[1]+1, v2[2])
                affx(v1[0]+1, v1[1]+2, v2[2])
            elif dx<0 and dy>0:
                affx(v1[0]-2, v1[1]+1, v1[2])
                affy(v1[0]-2, v1[1]+1, v1[2])
                affz(v1[0]-2, v1[1]+1, v1[2])
                affy(v1[0]-3, v1[1]+1, v1[2])
                affz(v1[0]-3, v1[1]+1, v1[2])
                affx(v1[0]-2, v1[1]+2, v1[2])
                affz(v1[0]-2, v1[1]+2, v1[2])
                affz(v1[0]-3, v1[1]+2, v1[2])
                affx(v1[0]-2, v1[1]+1, v2[2])
                affy(v1[0]-2, v1[1]+1, v2[2])
                affy(v1[0]-3, v1[1]+1, v2[2])
                affx(v1[0]-2, v1[1]+2, v2[2])
            elif dx>0 and dy<0:
                affx(v1[0]+1, v1[1]-2, v1[2])
                affy(v1[0]+1, v1[1]-2, v1[2])
                affz(v1[0]+1, v1[1]-2, v1[2])
                affy(v1[0]+2, v1[1]-2, v1[2])
                affz(v1[0]+2, v1[1]-2, v1[2])
                affx(v1[0]+1, v1[1]-3, v1[2])
                affz(v1[0]+1, v1[1]-3, v1[2])
                affz(v1[0]+2, v1[1]-3, v1[2])
                affx(v1[0]+1, v1[1]-2, v2[2])
                affy(v1[0]+1, v1[1]-2, v2[2])
                affy(v1[0]+2, v1[1]-2, v2[2])
                affx(v1[0]+1, v1[1]-3, v2[2])
            elif dx<0 and dy<0:
                affx(v1[0]-2, v1[1]-2, v1[2])
                affy(v1[0]-2, v1[1]-2, v1[2])
                affz(v1[0]-2, v1[1]-2, v1[2])
                affy(v1[0]-3, v1[1]-2, v1[2])
                affz(v1[0]-3, v1[1]-2, v1[2])
                affx(v1[0]-2, v1[1]-3, v1[2])
                affz(v1[0]-2, v1[1]-3, v1[2])
                affz(v1[0]-3, v1[1]-3, v1[2])
                affx(v1[0]-2, v1[1]-2, v2[2])
                affy(v1[0]-2, v1[1]-2, v2[2])
                affy(v1[0]-3, v1[1]-2, v2[2])
                affx(v1[0]-2, v1[1]-3, v2[2])
            else:
                raise Exception
        elif dy==0 and dx==0:
            # dz (2)
            # 0 1 1 0
            # 1 1 1 1
            # 1 1 1 1
            # 0 1 1 0
            if dz>0:
                affz(v1[0], v1[1], v1[2])
                affz(v1[0]+1, v1[1], v1[2])
                affz(v1[0], v1[1]+1, v1[2])
                affz(v1[0]-1, v1[1], v1[2])
                affz(v1[0]-1, v1[1]+1, v1[2])
                affz(v1[0], v1[1]-1, v1[2])
                affz(v1[0]+1, v1[1]-1, v1[2])
                affz(v1[0]-1, v1[1]-1, v1[2])
                affz(v1[0]-2, v1[1], v1[2])
                affz(v1[0], v1[1]-2, v1[2])
                affz(v1[0]-1, v1[1]-2, v1[2])
                affz(v1[0]-2, v1[1]-1, v1[2])
            else:
                affz(v1[0], v1[1], v2[2])
                affz(v1[0]+1, v1[1], v2[2])
                affz(v1[0], v1[1]+1, v2[2])
                affz(v1[0]-1, v1[1], v2[2])
                affz(v1[0]-1, v1[1]+1, v2[2])
                affz(v1[0], v1[1]-1, v2[2])
                affz(v1[0]+1, v1[1]-1, v2[2])
                affz(v1[0]-1, v1[1]-1, v2[2])
                affz(v1[0]-2, v1[1], v2[2])
                affz(v1[0], v1[1]-2, v2[2])
                affz(v1[0]-1, v1[1]-2, v2[2])
                affz(v1[0]-2, v1[1]-1, v2[2])
        else:
            # dz,dy | dz,dx (8)
            # 0
            # 0
            # 2
            # 4 2
            # 2 2
            # 0 0
            # 0 0
            # 0 0
            if dy>0:
                if dz>0:
                    affz(v1[0], v1[1]+1, v1[2])
                    affz(v1[0]-1, v1[1]+1, v1[2])
                    affz(v1[0], v1[1]+2, v1[2])
                    affz(v1[0]-1, v1[1]+2, v1[2])
                else:
                    affz(v1[0], v1[1]+1, v2[2])
                    affz(v1[0]-1, v1[1]+1, v2[2])
                    affz(v1[0], v1[1]+2, v2[2])
                    affz(v1[0]-1, v1[1]+2, v2[2])
                affy(v1[0], v1[1], v1[2])
                affy(v1[0]-1, v1[1], v1[2])
                affy(v1[0], v1[1]+1, v1[2])
                affy(v1[0]-1, v1[1]+1, v1[2])
                affy(v1[0], v1[1]+1, v2[2])
                affy(v1[0]-1, v1[1]+1, v2[2])
                affy(v1[0], v1[1]+2, v2[2])
                affy(v1[0]-1, v1[1]+2, v2[2])
            elif dy<0:
                if dz>0:
                    affz(v1[0], v1[1]-2, v1[2])
                    affz(v1[0]-1, v1[1]-2, v1[2])
                    affz(v1[0], v1[1]-3, v1[2])
                    affz(v1[0]-1, v1[1]-3, v1[2])
                else:
                    affz(v1[0], v1[1]-2, v2[2])
                    affz(v1[0]-1, v1[1]-2, v2[2])
                    affz(v1[0], v1[1]-3, v2[2])
                    affz(v1[0]-1, v1[1]-3, v2[2])
                affy(v1[0], v1[1]-1, v1[2])
                affy(v1[0]-1, v1[1]-1, v1[2])
                affy(v1[0], v1[1]-2, v1[2])
                affy(v1[0]-1, v1[1]-2, v1[2])
                affy(v1[0], v1[1]-2, v2[2])
                affy(v1[0]-1, v1[1]-2, v2[2])
                affy(v1[0], v1[1]-3, v2[2])
                affy(v1[0]-1, v1[1]-3, v2[2])
            elif dx>0:
                if dz>0:
                    affz(v1[0]+1, v1[1], v1[2])
                    affz(v1[0]+1, v1[1]-1, v1[2])
                    affz(v1[0]+2, v1[1], v1[2])
                    affz(v1[0]+2, v1[1]-1, v1[2])
                else:
                    affz(v1[0]+1, v1[1], v2[2])
                    affz(v1[0]+1, v1[1]-1, v2[2])
                    affz(v1[0]+2, v1[1], v2[2])
                    affz(v1[0]+2, v1[1]-1, v2[2])
                affy(v1[0], v1[1], v1[2])
                affy(v1[0], v1[1]-1, v1[2])
                affy(v1[0]+1, v1[1], v1[2])
                affy(v1[0]+1, v1[1]-1, v1[2])
                affy(v1[0]+1, v1[1], v2[2])
                affy(v1[0]+1, v1[1]-1, v2[2])
                affy(v1[0]+2, v1[1], v2[2])
                affy(v1[0]+2, v1[1]-1, v2[2])
            elif dx<0:
                if dz>0:
                    affz(v1[0]-2, v1[1], v1[2])
                    affz(v1[0]-2, v1[1]-1, v1[2])
                    affz(v1[0]-3, v1[1], v1[2])
                    affz(v1[0]-3, v1[1]-1, v1[2])
                else:
                    affz(v1[0]-2, v1[1], v2[2])
                    affz(v1[0]-2, v1[1]-1, v2[2])
                    affz(v1[0]-3, v1[1], v2[2])
                    affz(v1[0]-3, v1[1]-1, v2[2])
                affy(v1[0]-1, v1[1], v1[2])
                affy(v1[0]-1, v1[1]-1, v1[2])
                affy(v1[0]-2, v1[1], v1[2])
                affy(v1[0]-2, v1[1]-1, v1[2])
                affy(v1[0]-2, v1[1], v2[2])
                affy(v1[0]-2, v1[1]-1, v2[2])
                affy(v1[0]-3, v1[1], v2[2])
                affy(v1[0]-3, v1[1]-1, v2[2])
            else:
                raise Exception
    
    if 1<=v1[0]<affsz[0]//resz-1 and 1<=v1[1]<affsz[1]//resz-1 \
        and 1<=v1[2]<affsz[2]-1 and \
        1<=v2[0]<affsz[0]//resz-1 and 1<=v2[1]<affsz[1]//resz-1 \
        and 1<=v2[2]<affsz[2]-1:
        assert len(aff_list) == 12

    try:
        return np.min(np.array(aff_list))
    except:
        print(aff_list, v1, v2)
        raise Exception







