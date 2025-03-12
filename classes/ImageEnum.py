from enum import Enum

class ImageSource(Enum):
    IMAGE_1 = 1
    IMAGE_2 = 2
    
class Channel(Enum):
    RED = 0
    GREEN = 1
    BLUE = 2

class DistributionCurve(Enum):
    PDF = "PDF"
    CDF = "CDF"
