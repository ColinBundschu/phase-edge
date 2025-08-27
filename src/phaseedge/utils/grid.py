import math

ANCHOR: float = 0.0

def snap_floor(x: float, bin_width: float, anchor: float = ANCHOR) -> float:
    k = math.floor((x - anchor) / bin_width)
    return anchor + k * bin_width

def snap_ceil(x: float, bin_width: float, anchor: float = ANCHOR) -> float:
    k = math.ceil((x - anchor) / bin_width)
    return anchor + k * bin_width
