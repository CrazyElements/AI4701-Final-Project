import cv2
import numpy as np
import os
import re


# Segment the characters in the plate
class CharSegment(object):
    def __init__(self, img_, out_dir_, out_fpre_, color) -> None:
        self.img = img_
        self.out_dir = out_dir_
        self.out_fpre = out_fpre_
        self.colorType = color
        self.n_char = 0

    def preprocess(self):
        """Preprocess the image (binarized, remove rivets)

        Returns:
            ndarray: binary image without rivets
        """
        # convert to gray image
        img_blur = cv2.GaussianBlur(self.img, (5, 5), 0)
        img_gray = cv2.cvtColor(img_blur, cv2.COLOR_BGR2GRAY)
        print(self.colorType)
        if self.colorType == "Color.GREEN":
            _, img_bin = cv2.threshold(img_gray, 140, 255,
                                       cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            self.n_char = 8
        else:
            _, img_bin = cv2.threshold(img_gray, 100, 255,
                                       cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            self.n_char = 7

        def removeRivets(img_bin_):
            """remove the rivets in the plate

            Args:
                img_bin_ (ndarray): binary image

            Returns:
                ndarray: binary image without rivets
            """
            h, w = img_bin_.shape[:2]
            for y in range(h):
                n_white = np.sum(img_bin_[y, :] == 255)
                if n_white < w // 5 or n_white > 0.8 * w:
                    img_bin_[y, :] = 0
                else:
                    break
            for y in range(h - 1, -1, -1):
                n_white = np.sum(img_bin_[y, :] == 255)
                if n_white < w // 5 or n_white > 0.8 * w:
                    img_bin_[y, :] = 0
                else:
                    break
            return img_bin_

        self.saveImg(img_bin, self.out_fpre + "bin.jpg")
        img_rivetsRemoved = removeRivets(img_bin)
        self.saveImg(img_rivetsRemoved, self.out_fpre + "rivetsRemoved.jpg")
        # self.showImg(img_rivetsRemoved)
        return img_rivetsRemoved

    def findContours(self,
                     img_bin_,
                     min_area_per_of_total_area=20,
                     whRatio_range=(2, 4),
                     max_bias_wrt_mline_per_of_h=7):
        """Find the contours of the characters in the plate

        Args:
            img_bin_ (ndarray): binary image
            min_area_per_of_total_area (int, optional): min area percentage of total area. Defaults to 20.
            whRatio_range (tuple, optional): range(min, max) of width / height of the image. Defaults to (2, 4).
            max_bias_wrt_mline_per_of_h (int, optional): max distance from the center of the character to the line in the image. Defaults to 7.

        Returns:
            ndarray: the position of the four points of contour of the English and numeric characters in the image, sorted by x
        """
        # 首先得到数字和英文字符的轮廓
        contours, _ = cv2.findContours(img_bin_, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        img_ = self.img.copy()
        # sort the contours
        numAndEnChars = []
        plate_area = img_.shape[0] * img_.shape[1]
        print("h:", img_.shape[0], "w:", img_.shape[1], "area:", plate_area)
        for contour in contours:
            minRect = cv2.minAreaRect(contour)
            scale = 1.1  # 扩大
            minRect_ = (minRect[0], (minRect[1][0] * scale,
                                     minRect[1][1] * scale), minRect[2])
            box = cv2.boxPoints(minRect_)  # 获取旋转矩形的四个顶点坐标
            box = np.int0(box)
            y1, y2, x1, x2 = min(box[:,
                                     1]), max(box[:,
                                                  1]), min(box[:,
                                                               0]), max(box[:,
                                                                            0])
            if y1 < 0:
                y1 = 0
            if y2 > img_.shape[0]:
                y2 = img_.shape[0]
            if x1 < 0:
                x1 = 0
            if x2 > img_.shape[1]:
                x2 = img_.shape[1]
            roi_region = [y1, y2, x1, x2]

            def isChar(roi_region):
                """Determine whether the region is a character

                Args:
                    roi_region (ndarray): the region of the character(position of four points)

                Returns:
                    bool: True if the region is a character
                """
                y1, y2, x1, x2 = roi_region
                h, w = y2 - y1, x2 - x1
                # print(h, w, plate_area)
                area = h * w  # height * width
                whRatio = w / h  # 宽高比
                bias_wrt_mline = abs(y1 + y2 // 2 - h // 2)  # 相对中线的偏移量

                if whRatio < 1:
                    whRatio = 1 / whRatio

                print("area: ", area, "whRatio: ", whRatio, "bias_wrt_mline: ",
                      bias_wrt_mline)

                if area < plate_area // min_area_per_of_total_area or whRatio < whRatio_range[
                        0] or whRatio > whRatio_range[
                            1] or bias_wrt_mline > h // max_bias_wrt_mline_per_of_h:
                    return False
                return True

            if isChar(roi_region):
                numAndEnChars.append(roi_region)
        numAndEnChars_sorted = sorted(numAndEnChars, key=lambda x: x[2])
        if len(numAndEnChars) > self.n_char - 1:
            numAndEnChars_sorted = numAndEnChars_sorted[1:]
        elif len(numAndEnChars) < self.n_char - 1:
            return self.findContours(
                img_bin_,
                max_bias_wrt_mline_per_of_h=max_bias_wrt_mline_per_of_h - 1)
        return np.array(numAndEnChars_sorted)

    def charSegment(self):
        """Segment the characters in the plate

        Returns:
            ndarray: the position of the four points of contour of the English and numeric characters in the image, sorted by x
        """
        img_bin = self.preprocess()
        numAndEnChars_sorted = self.findContours(img_bin)

        def findCNChar():
            """find the Chinese character in the plate and merge it into English and numeric characters' positions

            Returns:
                ndarray: the position of the four points of contour of all characters in the image, sorted by x
            """
            yu_total = np.mean(numAndEnChars_sorted[:, 0], dtype=np.int32)
            yb_total = np.mean(numAndEnChars_sorted[:, 1], dtype=np.int32)
            w_avg = np.mean(numAndEnChars_sorted[:, 3] -
                            numAndEnChars_sorted[:, 2],
                            dtype=np.int32)
            w_diff_avg = np.mean(numAndEnChars_sorted[2:, 2] -
                                 numAndEnChars_sorted[1:-1, 2],
                                 dtype=np.int32)
            x1 = numAndEnChars_sorted[0, 2] - w_diff_avg
            cn_char = np.array([[yu_total, yb_total, x1, x1 + w_avg]])
            chars_sorted = np.concatenate([cn_char, numAndEnChars_sorted],
                                          axis=0)
            return chars_sorted

        chars_sorted = findCNChar()
        print(chars_sorted)
        print("found all chars?", len(chars_sorted) == self.n_char)
        self.drawContours(chars_sorted)
        for i, char_region in enumerate(chars_sorted):
            y1, y2, x1, x2 = char_region
            if y1 < 0:
                y1 = 0
            if x1 < 0:
                x1 = 0
            char = img_bin[y1:y2, x1:x2]
            filename = self.out_fpre + f"char_{i}" + ".jpg"
            self.saveImg(char, filename, size=(20, 20))
            # self.showImg(char, size=(100, 150))

    def drawContours(self, chars_sorted):
        """Draw the contours of the characters according to the position of the four points of contours

        Args:
            chars_sorted (ndarray): the position of the four points of contour of the all characters in the image, sorted by x
        """
        img_ = self.img
        for i, char_region in enumerate(chars_sorted):
            y1, y2, x1, x2 = char_region
            if y1 < 0:
                y1 = 0
            if x1 < 0:
                x1 = 0
            cv2.rectangle(img_, (x1, y1), (x2, y2), (0, 255, 0), 2)
        self.saveImg(img_, self.out_fpre + "contours.jpg")

    def showImg(self, img, name="image", size=(640, 200)):
        """Show the image
        """
        img_ = cv2.resize(img, size, interpolation=cv2.INTER_CUBIC)
        cv2.imshow(name, img_)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def saveImg(self, img, filename, size=(640, 200)):
        """Save the image
        """
        img_ = cv2.resize(img, size, interpolation=cv2.INTER_CUBIC)
        path = os.path.join(self.out_dir, filename)
        cv2.imwrite(path, img_)


def searchColor(s):
    """Search the color of the plate according to the filename

    Args:
        s (string): the filename of the plate image

    Returns:
        string: the color of the plate
    """
    pattern = r"_\w+\.(\w+)_"
    match = re.search(pattern, s)
    if match:
        color = match.group(1)
        return color
    else:
        print("No match")


def splitChars4Onelevel(level, number, color):
    """Segment the characters in the plate for a specific level and number

    Args:
        level (string): the difficulty
        number (string): the number of the plate
        color (string): the color of the plate
    """
    in_filename = "images_res/" + level + "/plate_detect/" + number + "_" + color + "_warp.jpg"
    # print(in_filename)
    if not os.path.exists(in_filename):
        return
    img = cv2.imread(in_filename)
    out_dir = "images_res/" + level + "/char_segment/" + number
    out_fpre = number + "_"
    os.makedirs(out_dir, exist_ok=True)
    cv2.resize(img, (800, 600), interpolation=cv2.INTER_CUBIC)
    cs = CharSegment(img, out_dir, out_fpre, color)
    cs.charSegment()


def splitChats4AllLevels():
    """Segment the characters in the plate for all levels and numbers
    """
    levels = ["easy", "medium", "difficult"]
    numbers = [1, 2, 3]
    for level in levels:
        for number in numbers:
            for color in ["Color.BLUE", "Color.GREEN"]:
                splitChars4Onelevel(level, str(number), color)


if __name__ == "__main__":
    splitChats4AllLevels()
    # splitChars4Onelevel("difficult", "2", "Color.GREEN")
