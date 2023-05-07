import cv2
import numpy as np
from enum import Enum
import os
# from getPlate import getPlate4SpecificLevel, getPlate4AllLevels

Color = Enum('Color', ('BLUE', "GREEN"))
lower_blue = np.array([100, 150, 60])  # 深蓝色
upper_blue = np.array([120, 255, 255])
lower_green = np.array([50, 40, 100])  # 真：131, 37%, 76%
upper_green = np.array([90, 100, 255])  # 假


# detect the place in the image
class ColorMatchPlateDetect(object):
    def __init__(self, img_, out_dir_, out_fpre_):
        self.img = img_
        self.out_dir = out_dir_
        self.out_fpre = out_fpre_
        self.imgSize = (640, 200)
        self.hsvLB = np.array([0, 0, 0])
        self.hsvUB = np.array([0, 0, 0])
        self.error = 0.5  # 车牌偏差量
        self.aspect = 3.2  # 车牌标准宽高比
        self.areaDpi = 1000
        self.areaMin = 50
        self.areaMax = 1000
        self.colorType = Color.BLUE
        self.mask = 0
        self.mask_b_roi = 0
        self.mask_g_roi = 0
        self.roi = []
        self.wh = (0, 0)

    # 颜色定位，在色彩空间中处理，使用H分量的蓝色和黄色匹配
    def RGB2HSV(self, img):
        """Convert the image from RGB to HSV

        Args:
            img (ndarray): the image to be converted

        Returns:
            ndarray: hsv image after gaussian blur
        """
        # 1. convert the image to HSV
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # print(type(hsv))
        # 2. split the image into 3 channels
        h, s, v = cv2.split(hsv)
        # 3. equalize the value channel
        v = cv2.equalizeHist(v)
        # 4. merge the 3 channels
        hsv = cv2.merge([h, s, v])
        hsv_blur = cv2.GaussianBlur(hsv, (7, 7), 0)
        return hsv_blur

    def preProcess(self, img, y_size, x_size, mode=1):
        """Preprocess the image (erode and dilate)

        Returns:
            ndarray: the preprocessed image
        """
        kernel_y = cv2.getStructuringElement(cv2.MORPH_RECT, y_size)
        kernel_x = cv2.getStructuringElement(cv2.MORPH_RECT, x_size)
        if mode:
            eroded1 = cv2.erode(img, kernel_x)
            dilated1 = cv2.dilate(eroded1, kernel_y)
            eroded2 = cv2.erode(dilated1, kernel_y)
            dilated2 = cv2.dilate(eroded2, kernel_x)
            return dilated2
        else:
            dilated1 = cv2.dilate(img, kernel_y)
            erode1 = cv2.erode(dilated1, kernel_x)
            dilated2 = cv2.dilate(erode1, kernel_x)
            erode2 = cv2.erode(dilated2, kernel_y)
            return erode2

    def getHsvUnderMask(self, img):
        """Get the hsv image under the mask

        Args:
            img (ndarray): the image to be processed

        Returns:
            ndarrat: the hsv image under the mask
        """
        hsv = self.RGB2HSV(img)
        self.mask_b_roi = cv2.inRange(hsv, lower_blue, upper_blue)
        self.mask_g_roi = cv2.inRange(hsv, lower_green, upper_green)
        self.mask = self.mask_b_roi + self.mask_g_roi
        # 只有在掩模图像中对应位置的像素点为 1 时，HSV 彩色图像对应位置的像素点才会保留下来，否则就会被掩盖
        hsv_mask = cv2.bitwise_and(hsv, hsv, mask=self.mask)
        return hsv_mask

    def getCandidatePlates(self, img):
        """Get the candidate plates

        Args:
            img (ndarray): the image to be processed

        Returns:
            ndarray: roi image of the plates
            ndarray: roi image of the plates under the mask
        """
        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            minRect = cv2.minAreaRect(contour)  # 返回最小外接矩形
            _, wh, _ = minRect
            scale = 1.4  # 扩大
            minRect_ = (minRect[0], (minRect[1][0] * scale,
                                     minRect[1][1] * scale), minRect[2])
            box = cv2.boxPoints(minRect_)  # 获取旋转矩形的四个顶点坐标
            box = np.int0(box)
            y1, y2, x1, x2 = min(box[:,
                                     1]), max(box[:,
                                                  1]), min(box[:,
                                                               0]), max(box[:,
                                                                            0])
            roi = [y1, y2, x1, x2]
            if self.isPlate(roi, wh):
                self.roi = roi
                self.wh = wh
                img_plate_roi = self.img[y1:y2, x1:x2]
                img_plate_mask_roi = self.mask[y1:y2, x1:x2]
                self.mask_b_roi = self.mask_b_roi[y1:y2, x1:x2]
                self.mask_g_roi = self.mask_g_roi[y1:y2, x1:x2]
                return img_plate_roi, img_plate_mask_roi

    def isPlate(self, roi, wh):
        """Judge whether the rectangle is a plate

        Args:
            roi (ndarray): region of interest(plate)
            wh (tuple): the width and height of the plate

        Returns:
            bool: whether the rectangle is a plate
        """
        y1, y2, x1, x2 = roi
        w, h = wh
        min_area = self.areaDpi * self.areaMin  # 面积最小值
        max_area = self.areaDpi * self.areaMax  # 面积最大值

        ratio_min = self.aspect * (1 - self.error)  # 宽高比最小值
        ratio_max = self.aspect * (1 + self.error)  # 宽高比最大值

        area = h * w  # height * width

        if area == 0:
            return False

        whRatio = w / h  # 宽高比

        if whRatio < 1:
            whRatio = 1 / whRatio

        rate = np.sum(self.mask[y1:y2, x1:x2] == 255) / area
        # 被舍弃的矩形框
        if (area < min_area
                or area > max_area) or (whRatio < ratio_min
                                        or whRatio > ratio_max) or rate < 0.4:
            print(rate, area)
            return False
        else:
            print("rate:", rate)
            return True

    def colorMatch(self):
        """Determine the color of the plate by the probability of the three templates
        """
        area = self.mask_b_roi.shape[0] * self.mask_b_roi.shape[1]
        r_b = np.sum(self.mask_b_roi == 255) / area
        r_g = np.sum(self.mask_g_roi == 255) / area
        print(f"blue rate is {r_b}, green rate is {r_g}")
        self.colorType = [Color.BLUE, Color.GREEN][np.argmax([r_b, r_g])]

    def getSrcAndDstPts(self, img_roi, mask_roi, offset):
        """Get the source points and destination points for perspective transform

        Args:
            img_roi (ndarray): roi image of the plates
            mask_roi (ndarray): roi image of the plates under the mask
            offset (int): the offset of the line

        Returns:
            ndarray: source points
            ndarray: destination points
        """
        h, w = img_roi.shape[:2]
        h1, w1 = h // 2, w // 12
        mask_up = mask_roi[:h1, 5 * w1:7 * w1]
        mask_down = mask_roi[h1:, 5 * w1:7 * w1]

        # print(mask_up.shape)

        def findLine(mask, ob):
            """find the upper and lower boundary of the mask

            Args:
                mask (ndarray): the roi mask
                ob (string): determine which to be processed, the upper or lower boundary

            Returns:
                tuple: point of the line
                int: the slope of the line
            """
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)

            def deleteCorner(contours):
                """Filter out corner points

                Args:
                    contours (ndarray): contours of the outline of the mask

                Returns:
                    ndarray: contours of the outline of the mask without corner points
                """
                h_, w_ = mask.shape[:2]
                contour = contours[0]
                right_top, right_bottom, left_top, left_bottom = np.array([
                    w_ - 1, 0
                ]), np.array([w_ - 1,
                              h_ - 1]), np.array([0, 0]), np.array([0, h_ - 1])
                delete_idx = np.array([False] * len(contour))
                for i, point in enumerate(contour):
                    point = point[0]

                    def isCorner():
                        """Determine whether the point is a corner point

                        Returns:
                            bool: whether the point is a corner point
                        """
                        return (ob == "down" and
                                (np.array_equal(point, right_top)
                                 or np.array_equal(point, left_top))) or (
                                     ob == "up" and
                                     (np.array_equal(point, right_bottom)
                                      or np.array_equal(point, left_bottom)))

                    if isCorner():
                        delete_idx[i] = True
                contour = np.delete(contour, np.where(delete_idx), axis=0)
                # for point in contour:
                #     cv2.circle(mask, tuple(point[0]), 3, 0, 2)
                return contour

            contour = deleteCorner(contours)
            vx, vy, x, y = cv2.fitLine(contour, cv2.DIST_L2, 0, 0.01, 0.01)
            k = vy / vx
            if ob == "up":
                x1, y1 = int(x - 1000) + 5 * w1, int(y - 1000 * k)
                x2, y2 = int(x + 1000) + 5 * w1, int(y + 1000 * k)
            else:
                x1, y1 = int(x - 1000) + 5 * w1, int(y - 1000 * k) + h1
                x2, y2 = int(x + 1000) + 5 * w1, int(y + 1000 * k) + h1
            # cv2.line(img_roi, (x1, y1), (x2, y2), (0, 0, 255), 2)
            # self.showImg(mask, "mask")
            pt = (0, int(y1 - k * x1))
            return pt, k

        pt2, k2 = findLine(mask_down, "down")
        if self.colorType == Color.BLUE:
            pt1, k1 = findLine(mask_up, "up")
        else:
            k1 = k2
            pt1 = (0, pt2[1] - self.wh[1] * 1.15)
            x1, y1 = int(pt1[0]), int(pt1[1])
            x2, y2 = x1 + 1000, y1 + int(1000 * k1)
            # print(x1, y1, x2, y2)
            # cv2.line(img_roi, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # self.showImg(img_roi, 'img_roi')
        pt1 = (pt1[0], pt1[1] + offset)
        pt2 = (pt2[0], pt2[1] - offset)

        # print(pt1, pt2)

        def findPoint(pt, left, right, k):
            """Find the point on the line using binary search

            Args:
                pt (tuple): the point on the line
                left (int): the left boundary of the search range(x)
                right (int): the right boundary of the search range(x)
                k (float): the slope of the line
            """
            def get_y(x):
                """Get the y coordinate of the point on the line

                Args:
                    x (int): the x coordinate of the point on the line

                Returns:
                    int: the y coordinate of the point on the line
                """
                return int(k * (x - pt[0]) + pt[1])

            left_, _ = left, right
            while left + 1 < right:  # 区间不为空
                mid_x = int(left + (right - left) // 2)
                mid_y = get_y(mid_x)
                # print(mid_x, left, right)
                if mask_roi[mid_y,
                            mid_x] and mask_roi[get_y(mid_x + 4), mid_x +
                                                4] or mask_roi[get_y(mid_x -
                                                                     4),
                                                               mid_x - 4]:
                    if left_ == -1:
                        right = mid_x  # 范围缩小到 (mid, right)
                    else:
                        left = mid_x
                else:
                    if left_ == -1:
                        left = mid_x  # 范围缩小到 (left, mid)
                    else:
                        right = mid_x
            idx_y = get_y(left)
            # print("\n")
            return (left, idx_y)

        p3 = findPoint(pt2, -1, w // 3, k2)
        p4 = findPoint(pt2, 2 * w // 3, w, k2)
        if self.colorType == Color.BLUE:
            p1 = findPoint(pt1, -1, w // 3, k1)
            p2 = findPoint(pt1, 2 * w // 3, w, k1)
        else:
            p1 = (p3[0], int(p3[1] - self.wh[1] * 1.15) + offset)
            p2 = (p4[0], int(p4[1] - self.wh[1] * 1.15) + offset)

        src_pts = np.float32([p1, p2, p3, p4])
        dst_pts = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
        # print(src_pts)
        # cv2.circle(img_roi, p1, 5, (0, 0, 255), -1)
        # cv2.circle(img_roi, p2, 5, (0, 0, 255), -1)
        # cv2.circle(img_roi, p3, 5, (0, 0, 255), -1)
        # cv2.circle(img_roi, p4, 5, (0, 0, 255), -1)
        # self.showImg(img_roi, 'img_roi')
        return src_pts, dst_pts

    def perspectiveTrans(self, img_plate_roi, mask_roi):
        """Perspective transform

        Args:
            img_plate_roi (np.ndarray): the image of the plate
            mask_roi (np.ndarray): the mask of the plate
        """
        # 得到透视变换的源点和目标点
        src_pts, dst_pts = self.getSrcAndDstPts(img_plate_roi,
                                                mask_roi,
                                                offset=20)
        # 计算透视变换矩阵
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        h, w = img_plate_roi.shape[:2]
        img_warp = cv2.warpPerspective(img_plate_roi, M, (w, h))
        # self.showImg(img_warp, "img_warp", (640, 200))
        self.saveImg(img_warp,
                     self.out_fpre + f"{self.colorType}_warp.jpg",
                     size=(640, 200))

    def plateLocate(self):
        """Locate the plate in the image
        """
        hsv_mask = self.getHsvUnderMask(self.img)
        hsv_mask_gray = cv2.cvtColor(hsv_mask, cv2.COLOR_BGR2GRAY)
        self.saveImg(hsv_mask_gray, self.out_fpre + "hsv_mask_gray.jpg")
        # 腐蚀和膨胀
        img_preprocessed1 = self.preProcess(hsv_mask_gray, (50, 1), (1, 15), 1)
        img_preprocessed2 = self.preProcess(img_preprocessed1, (50, 1),
                                            (1, 15), 0)
        img_preprocessed = cv2.bitwise_and(img_preprocessed1,
                                           img_preprocessed2)
        self.saveImg(img_preprocessed, self.out_fpre + "preprocessed.jpg")
        img_plate_roi, img_mask_roi = self.getCandidatePlates(img_preprocessed)
        self.saveImg(img_plate_roi, self.out_fpre + "plate_roi.jpg")
        self.saveImg(img_mask_roi, self.out_fpre + "mask_roi.jpg")
        self.colorMatch()
        mask_roi = self.preProcess(img_mask_roi, (135, 1), (1, 30), mode=1)
        self.saveImg(mask_roi, self.out_fpre + "mask_roi_processed.jpg")
        # src_pts, dst_pts = self.getSrcAndDstPts(img_plate_roi, mask_roi)
        self.perspectiveTrans(img_plate_roi, mask_roi)

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


def getPlate4Onelevel(level, number):
    """Get the plate for one level

    Args:
        level (string): the difficulty
        number (string): the number of the plate
    """
    in_filename = "images/" + level + "/" + number + ".jpg"
    img = cv2.imread(in_filename)
    out_dir = "images_res/" + level + "/plate_detect"
    out_fpre = number + "_"
    os.makedirs(out_dir, exist_ok=True)
    cv2.resize(img, (800, 600), interpolation=cv2.INTER_CUBIC)
    pd = ColorMatchPlateDetect(img, out_dir, out_fpre)
    if level == "easy":
        pd.getHsvUnderMask(pd.img)
        pd.colorMatch()
        filename = out_fpre + f"{pd.colorType}_warp.jpg"
        pd.saveImg(pd.img, filename, size=(640, 200))
    else:
        pd.plateLocate()


def getPlate4AllLevels():
    """Get the plate for all levels
    """
    levels = ["easy", "medium", "difficult"]
    numbers = [1, 2, 3]
    for level in levels:
        for number in numbers:
            getPlate4Onelevel(level, str(number))


def getPlate4SpecificLevel():
    """Get the plate for specific level
    """
    level = ["easy", "medium", "difficult"][int(
        input(
            "please select the level of difficulty(1: easy, 2: medium, 3: difficult):"
        )) - 1]
    number = input("please select the number of the image(1-3):")
    getPlate4Onelevel(level, str(number))


if __name__ == "__main__":
    getPlate4AllLevels()
    # getPlate4SpecificLevel()
    # getPlate4Onelevel("medium", "1")
    # getPlate4Onelevel("difficult", "3")