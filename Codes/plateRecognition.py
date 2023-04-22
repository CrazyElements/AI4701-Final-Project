from plate_locate import PlateLocate
from char_segment import CharSegment
from char_recognize import CharRecognize
# 基于 HyperLPR 的车牌识别系统


class PlateRecognition(object):
    def __init__(self) -> None:
        pass

    def plateLocate(self):
        """Locate the plate in the image
        """
        pass

    # 通过图像处理技术和 SVM 模型得到车牌
    def plateDetect(self):
        """Detect the plate in the image
        """
        pass

    def charSegment(self):
        """Segment the characters in the plate
        """
        pass

    # 通过图像处理技术和 CNN 模型得到字符串
    def charRecognize(self):
        """Recognize the characters in the plate
        """
        pass


if __name__ == "__main__":
    pass