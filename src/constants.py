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

class Probe(Enum):
    LINEAR = 0
    CURVED_LINEAR = 1
    PHASED_ARRAY = 2

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]