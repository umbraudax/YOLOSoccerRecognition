def get_center_of_bbox(bbox):
    x1, y1, x2, y2 = bbox
    return int((x1 + x2) / 2), int((y1 + y2) / 2)

def get_width_of_bbox(bbox):
    x1, _, x2, _ = bbox
    return int(x2 - x1)

def distance(x1, y1, x2, y2):
    return ((x1 - x2)**2 + (y1 - y2)**2)**0.5

def p_distance(p1, p2):
    return ((p1[0] - p1[0])**2 + (p1[1] - p2[1])**2)**0.5

def xy_distance(p1, p2):
    return p1[0]-p2[0], p1[1]-p2[1]

def get_foot_position(bbox):
    x1, _, x2, y2 = bbox
    return int((x1 + x2)/2), y2