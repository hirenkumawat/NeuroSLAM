import numpy as np
import cmath

# used by visual_odometry, view_cells
def compare_segments(seg1, seg2, length):
    """
    find contiguous subsegments, one from seg1 and one from seg2, 
    with smallest l1 distance. return their offset.
    """
    # brute force
    # best_dist = np.sum(np.abs(seg1[:length]-seg2[:length]))
    best_dist=99999999
    best_offset = -1

    for i in range(0,len(seg1)-length):
        # for j in range(0,len(seg2)-length):
        j= len(seg2)-length-2-i
        dist = np.sum(np.abs(seg1[i:][:length]-seg2[j:][:length]))
        if dist < best_dist:
            best_dist = dist
            best_offset = j-i
    
    return best_offset, best_dist

def wrapped_avg_idx(arr):
    n = len(arr)
    z = 0+0j
    for idx,val in enumerate(arr):
        z += val * np.e ** (1j * 2 * np.pi * idx / n)
    return (int(cmath.phase(z) / np.pi * n) + n) % n

def min_delta(d1, d2, max_):
    delta = np.min([np.abs(d1-d2), max_-np.abs(d1-d2)])
    return delta

def clip_rad_180(angle):
    while angle > np.pi:
        angle -= 2*np.pi
    while angle <= -np.pi:
        angle += 2*np.pi
    return angle

def clip_rad_360(angle):
    while angle < 0:
        angle += 2*np.pi
    while angle >= 2*np.pi:
        angle -= 2*np.pi
    return angle

def signed_delta_rad(angle1, angle2):
    dir = clip_rad_180(angle2 - angle1)
    
    delta_angle = abs(clip_rad_360(angle1) - clip_rad_360(angle2))
    
    if (delta_angle < (2*np.pi-delta_angle)):
        if (dir>0):
            angle = delta_angle
        else:
            angle = -delta_angle
    else: 
        if (dir>0):
            angle = 2*np.pi - delta_angle
        else:
            angle = -(2*np.pi-delta_angle)
    return angle
