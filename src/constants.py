from enum import Enum

class BeamKeypoints(Enum):
    X1 = 0
    Y1 = 1
    X2 = 2
    Y2 = 3
    X3 = 4
    Y3 = 5
    X4 = 6
    Y4 = 7

class ProbeType(Enum):
    LINEAR = 0
    CURVILINEAR = 1
    PHASED = 2