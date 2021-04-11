import numpy as np

def closest_point_on_line(a, b, p):
    # line: a,b
    # point:
    ap = p-a
    ab = b-a
    result = a + np.dot(ap,ab)/np.dot(ab,ab) * ab
    return result

def is_point_on_and_between_line(a, b, p):
    # line: a,b
    # point:
    x1, x2, x3 = a[0], b[0], p[0]
    y1, y2, y3 = a[1], b[1], p[1]
    slope = (y2 - y1) / (x2 - x1)
    p_on = (y3 - y1) == slope * (x3 - x1)
    p_between = (min(x1, x2) <= x3 <= max(x1, x2)) and (min(y1, y2) <= y3 <= max(y1, y2))
    on_and_between = p_on and p_between

    return on_and_between

def is_point_between_line(a, b, p):
    # line: a,b
    # point:
    x1, x2, x3 = a[0], b[0], p[0]
    y1, y2, y3 = a[1], b[1], p[1]
    p_between = (min(x1, x2) <= x3 <= max(x1, x2)) and (min(y1, y2) <= y3 <= max(y1, y2))
    return p_between

def direction_of_point(a, b, p):
    ap = p-a
    ab = b-a
    # if point on line, return 0, otherwise return 1 or -1
    return np.clip(np.cross(ap, ab),-1,1)

def object_cross_line(a, b, xyxy):
    a,b = np.array(a), np.array(b)
    x1,y1,x2,y2 = xyxy
    p1,p2,p3,p4 = np.array([x1,y1]),np.array([x2,y1]),np.array([x1,y2]),np.array([x2,y2])
    
    def point_of_line(a,b,p):
        near_point = closest_point_on_line(a,b,p)
        is_between = is_point_between_line(a,b,near_point)
        direction = None
        if is_between:
            direction = direction_of_point(a, b, p)
        return is_between, direction

    btw_dirs = [point_of_line(a,b,p) for p in [p1,p2,p3,p4]]
    is_betweens = [item[0] for item in btw_dirs]
    dirs = [item[1] for item in btw_dirs]
    # at least two points of an object is between line
    # and in each side at least one point.
    return sum(is_betweens)>1 and (dirs.count(1)>0 and dirs.count(-1)>0)