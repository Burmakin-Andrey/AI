import numpy as np


def iou(bbox1: list, bbox2: list) -> float:
    
    (x1, w1, y1, h1) = bbox1
    (x2, w2, y2, h2) = bbox2
    ps1 = [[x1, y1], [x1, y1+w1], [x1+h1, y1+w1], [x1+h1, y1]] # |0 1|
    ps2 = [[x2, y2], [x2, y2+w2], [x2+h2, y2+w2], [x2+h2, y2]] # |3 2|
    shape = max(max(x1 + h1, y1 + w1), max(x2 + h2, y2 + w2))
    feald = np.zeros((shape,shape))
    for i in range(ps2[0][0], ps2[3][0]):
        feald[i][ps2[0][1]:ps2[1][1]] += 1
    origS = feald.sum()
    for i in range(ps1[0][0], ps1[3][0]):
        feald[i][ps1[0][1]:ps1[1][1]] += 1
    feald[feald==1] = 0
    S = feald.sum()
    if ps1 == ps2:
        return 1.0
    else:
        return round((S / 2) / origS, 2)
    

bbox1 = [0, 10, 0, 10]
bbox2 = [0, 10, 1, 10]
bbox3 = [20, 30, 20, 30]
bbox4 = [5, 15, 5, 15]
assert iou(bbox1, bbox1) == 1.0
assert iou(bbox1, bbox2) == 0.9
assert iou(bbox1, bbox3) == 0.0
assert round(iou(bbox1, bbox4), 2) == 0.11