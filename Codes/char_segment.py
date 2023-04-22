import cv2
import numpy as np
from enum import Enum
import os


# Segment the characters in the plate
class CharSegment(object):
    def __init__(self, img_, out_dir_, out_fpre_) -> None:
        self.img = img_
        self.out_dir = out_dir_
        self.out_fpre = out_fpre_

    def charSegment(self):
        """Segment the characters in the plate
        """

        pass

    def showImg(self, img, name="image", size=(800, 600)):
        """Show the image
        """
        img_ = cv2.resize(img, size, interpolation=cv2.INTER_CUBIC)
        cv2.imshow(name, img_)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def saveImg(self, img, filename, size=(800, 600)):
        """Save the image
        """
        img_ = cv2.resize(img, size, interpolation=cv2.INTER_CUBIC)
        path = os.path.join(self.out_dir, filename)
        cv2.imwrite(path, img_)


if __name__ == "__main__":
    in_filename = "images_res_tradition/difficult/plate_detect/1_plate.jpg"
    img = cv2.imread(in_filename)
    out_dir = "images_res_tradition/difficult/char_segment"
    out_fpre = "1_"
    cs = CharSegment(img, out_dir, out_fpre)