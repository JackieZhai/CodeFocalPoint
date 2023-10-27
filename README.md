# FocalPoint

now testing on 'wafer 14'. (by Jingbin Yuan)

also testing on 'scn stacks' and 'zebrafish lc area'.

## data formats

XXX_error_2p.pkl:
```
dict[(label1, label2)] = [
    {'pos':[x,y,z], 'min':[x,y,z], 'max':[x,y,z], 'sample1':[x,y,z], 'sample2':[x,y,z], 'score':float}, 
    {'pos':[x,y,z], 'min':[x,y,z], 'max':[x,y,z], 'sample1':[x,y,z], 'sample2':[x,y,z], 'score':float}, 
    ...
]
```

XXX_error_1p.pkl:
```
dict[label] = [
    label_score:float, 
    {'pos':[x,y,z], 'min':[x,y,z], 'max':[x,y,z], 'score':float}, 
    {'pos':[x,y,z], 'min':[x,y,z], 'max':[x,y,z], 'score':float}, 
    ...
]
```

## data descriptions

amputate_error_2p.pkl: (from Yanchao Zhang)
1. inconsistency (...1 1 1 2 1 1 1...)
2. isolated piece (single layer and small enough)

divide_error_1p.pkl: (from Zhenchen Li)
1. branch point (must be long enough)
2. strange point (> 90 degree)

split_error_2p.pkl:
1. skeleton endpoints k-neighborhood
2. check the ending vectors

merge_error_1p.pkl:
1. skeleton points in affinity map
2. check the dark consist areas

## config descriptions

see configs/*.yaml

## contributions

* [Hao Zhai](https://github.com/JackieZhai)
* Zhenchen Li
* [Yanchao Zhang](https://github.com/Cristand)
* Jingbin Yuan
* Jing Liu
* Bei Hong