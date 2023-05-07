import cv2
import numpy as np
from enum import Enum
import os
# from getPlate import getPlate4SpecificLevel, getPlate4AllLevels

Color = Enum('Color', ('BLUE', "YELLOW", "GREEN"))
lower_blue = np.array([100, 150, 60])  # 深蓝色
upper_blue = np.array([120, 255, 255])
lower_yellow = np.array([10, 160, 150])  # 真： 43, 75%, 83%
upper_yellow = np.array([35, 255, 255])
lower_green = np.array([50, 40, 150])  # 真：131, 37%, 76%
upper_green = np.array([80, 100, 255])  # 假
lower_white = np.array([20, 0, 150])  # 真：45, 2%, 70%
upper_white = np.array([40, 30, 245])  # 假: 60, 4%, 95%, 也是真


# detect the place in the image
class ColorMatchPlateDetect(object):
    def __init__(self, img_, out_dir_, out_fpre_):
        self.img = img_
        self.out_dir = out_dir_
        self.out_fpre = out_fpre_
        self.imgSize = (640, 200)
        self.hsvLB = np.array([0, 0, 0])
        self.hsvUB = np.array([0, 0, 0])
        self.error = 0.2  # 车牌偏差量
        self.aspect = 3.2  # 车牌标准宽高比
        self.areaDpi = 1000
        self.areaMin = 70
        self.areaMax = 800
        self.colorType = Color.BLUE
        self.mask = 0

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
        hsv_blur = cv2.GaussianBlur(hsv, (5, 5), 0)
        return hsv_blur

    def getCandidatePlates(self, img):
        """Get the candidate plates

        Args:
            img (ndarray): the image to be processed

        Returns:
            ndarray: the candidate plates
        """
        # find contours
        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        validMinRects = []
        for contour in contours:
            minRect = cv2.minAreaRect(contour)  # 返回最小外接矩形
            # if color == Color.GREEN:
            # self.addGreenPlate(minRect, validMinRects)
            if self.isProperSize(minRect):
                validMinRects.append(minRect)
        # print(validMinRects)
        return validMinRects

    def plateLocate(self):
        """Locate the plate in the image
        """
        color = self.colorMatch()
        print("get plate color:", color)
        hsv_mask = self.getHsvUnderMask(self.img, color)
        hsv_mask_gray = cv2.cvtColor(hsv_mask, cv2.COLOR_BGR2GRAY)
        # self.showImg(hsv_mask_gray, "hsv_mask_gray")
        self.saveImg(hsv_mask_gray, self.out_fpre + "hsv_mask_gray.jpg")
        # 开闭运算
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (35, 35))
        closed = cv2.morphologyEx(hsv_mask_gray,
                                  cv2.MORPH_CLOSE,
                                  kernel,
                                  iterations=1)
        self.saveImg(closed, self.out_fpre + "closed.jpg")
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=2)
        self.saveImg(opened, self.out_fpre + "opened.jpg")
        validMinRects = self.getCandidatePlates(opened)
        self.drawMinRects(validMinRects)
        rotated_plate = self.cutMinRectsAndRotate(validMinRects)
        self.perspectiveTrans(rotated_plate, validMinRects)

    def colorMatch(self):
        """Match the color of the plate according to the probability of the three masks

        Returns:
            Color: the color of the plate
        """
        probs = []
        for color in Color:
            p = self.colorJudge(color)
            probs.append(p)
            print(f"the probability of {color} is {p}")
        self.colorType = [Color.BLUE, Color.YELLOW,
                          Color.GREEN][np.array(probs).argmax()]
        return self.colorType

    def getHsvUnderMask(self, img, color):
        """Get the hsv image under the specific color mask

        Args:
            img (ndarray): the image to be processed
            color (Color): the color of the plate

        Returns:
            ndarray: hsv image under the specific color mask
        """
        hsv = self.RGB2HSV(img)
        if color == Color.BLUE:
            self.hsvLB = lower_blue
            self.hsvUB = upper_blue
            # 只有在蓝色区域的像素点才会被设置为白色，其余区域的像素点都被设置为黑色
            mask_w = cv2.inRange(hsv, lower_white, upper_white)  # 蓝色车牌的文字是白色的
            mask_b = cv2.inRange(hsv, lower_blue, upper_blue)
            mask = mask_w | mask_b
        elif color == Color.YELLOW:
            self.hsvLB = lower_yellow
            self.hsvUB = upper_yellow
            mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        else:
            self.hsvLB = lower_green
            self.hsvUB = upper_green
            mask_g = cv2.inRange(hsv, lower_green, upper_green)
            mask_w = cv2.inRange(hsv, lower_white, upper_white)
            mask = mask_g | mask_w
        # 只有在掩模图像中对应位置的像素点为 1 时，HSV 彩色图像对应位置的像素点才会保留下来，否则就会被掩盖
        self.mask = mask
        hsv_mask = cv2.bitwise_and(hsv, hsv, mask=mask)
        return hsv_mask

    def colorJudge(self, color):
        """Judge the probability of the plate containing the specific color

        Args:
            color (Color): the possible color of the plate

        Returns:
            float: the probability of the plate containing the specific color
        """
        hsv_mask = self.getHsvUnderMask(self.img, color)
        _, _, v = cv2.split(hsv_mask)  # 返回灰度信息
        p = np.count_nonzero(v) / (v.shape[0] * v.shape[1])
        return p

    # 对外接矩形进行判断，排除不可能是车牌的矩形
    def isProperSize(self, minRect):
        """Judge whether the region is a proper plate

        Args:
            minRect (ndarray): the minAreaRect of the region

        Returns:
            bool: whether the region is a proper plate
        """
        _, hw, _ = minRect

        min_area = self.areaDpi * self.areaMin  # 面积最小值
        max_area = self.areaDpi * self.areaMax  # 面积最大值

        ratio_min = self.aspect * (1 - self.error)  # 宽高比最小值
        ratio_max = self.aspect * (1 + self.error)  # 宽高比最大值

        area = hw[0] * hw[1]  # height * width
        whRatio = hw[1] / hw[0]  # 宽高比

        if whRatio < 1:
            whRatio = 1 / whRatio

        # 被舍弃的矩形框
        if (area < min_area or area > max_area) or (whRatio < ratio_min
                                                    or whRatio > ratio_max):
            # print(area, whRatio)
            return False
        else:
            # print(area, whRatio)
            return True

    # 绘制出可能的车牌
    def drawMinRects(self, validMinRects):
        """Draw the possible plates

        Args:
            validMinRects (ndarray): the minAreaRects of the possible plates
        """
        img_ = self.img.copy()
        for minRect in validMinRects:
            scale = 1.05
            minRect = (minRect[0], (minRect[1][0] * scale,
                                    minRect[1][1] * scale), minRect[2])
            box = cv2.boxPoints(minRect)  # 获取旋转矩形的四个顶点坐标
            box = np.int0(box)
            cv2.drawContours(img_, [box], 0, (0, 0, 255), 5)  # 绘制轮廓图像
        self.saveImg(img_, self.out_fpre + "minRects.jpg")

    def cutMinRectsAndRotate(self, validMinRects):
        """Cut the possible plates from the original image and rotate them

        Args:
            validMinRects (ndarray): the minAreaRects of the possible plates

        Returns:
            ndarray: rotated images of the possible plates
        """
        for minRect in validMinRects:  # 这里只考虑了一个车牌的情况，如果有多个车牌，还需要用SVM进行判断
            scale = 1.05  # 扩大化旋转
            minRect = (minRect[0], (minRect[1][0] * scale,
                                    minRect[1][1] * scale), minRect[2])
            box = cv2.boxPoints(minRect)  # 获取旋转矩形的四个顶点坐标
            box = np.int0(box)
            # 将四个顶点的坐标转换为齐次坐标
            src_pts = np.concatenate(
                [box, np.ones([4, 1], dtype=np.float32)], axis=1)
            center, _, angle = minRect
            print("angle=", angle)
            y1, y2, x1, x2 = min(box[:,
                                     1]), max(box[:,
                                                  1]), min(box[:,
                                                               0]), max(box[:,
                                                                            0])
            if y1 < 0:
                y1 = 0
            if x1 < 0:
                x1 = 0
            plate_roi = self.img[y1:y2, x1:x2]
            self.saveImg(plate_roi, self.out_fpre + "plate_roi.jpg")
            # print(src_pts)
            # 将倾斜的车牌(最小外接矩形)旋转回来
            rotation_Mat = cv2.getRotationMatrix2D(
                center, angle if angle < 45 else angle - 90, scale=1)
            rotation_Mat = np.concatenate(
                [rotation_Mat, np.array([[0, 0, 1]])], axis=0)
            dst_pts = np.int0(
                src_pts
                @ rotation_Mat.T[:, :2])  # 这里应该扩大化旋转，因为旋转后可能会有一部分车牌未显示在图片中
            x1, x2, y1, y2 = min(dst_pts[:, 0]), max(dst_pts[:, 0]), min(
                dst_pts[:, 1]), max(dst_pts[:, 1])
            # print(dst_pts)
            rotated_img = cv2.warpAffine(
                self.img, rotation_Mat[:2, :],
                (self.img.shape[1], self.img.shape[0]))
            rotated_plate = rotated_img[y1:y2, x1:x2]
            filename = self.out_fpre + "rotatedPlate.jpg"
            self.saveImg(rotated_plate, filename=filename, size=(640, 200))
        return rotated_plate

    def calSlope(self, rotated_plate):
        """Calculate the slope of the bevel edge of rotated plate

        Args:
            rotated_plate (ndarray): rotated image of the plate

        Returns:
            float: the slope of the bevel edge
        """
        rp_blur = cv2.GaussianBlur(rotated_plate, (15, 15), 0)
        rp_mask = self.getHsvUnderMask(rp_blur, self.colorType)
        rp_gray = cv2.cvtColor(rp_mask, cv2.COLOR_BGR2GRAY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 30))
        closed = cv2.morphologyEx(rp_gray,
                                  cv2.MORPH_CLOSE,
                                  kernel,
                                  iterations=1)
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=2)
        h, w = opened.shape[:2]
        print(h, w)
        w_thresh = w // 10
        pts = []
        for y in range(1 * h // 3, 2 * h // 3, h // 30):
            for x in range(20, w_thresh):
                if rp_gray[y, x] > 100:
                    pts.append([y, x])
                    cv2.circle(closed, (x, y), 5, (255, 255, 255), -1)
                    break
        pts = np.array(pts)
        # self.showImg(closed, "closed", (640, 200))
        # 拟合直线
        vx, vy, _, _ = cv2.fitLine(points=pts,
                                   distType=cv2.DIST_L2,
                                   param=0.001,
                                   reps=0.001,
                                   aeps=0.01)
        slope = vy / vx
        return slope

    def getSrcAndDstPts(self, rotated_plate):
        """Get the source points and destination points for perspective transform

        Args:
            rotated_plate (ndarray): rotated image of the plate

        Returns:
            ndarray: source points
            ndarray: destination points
        """
        h, w = rotated_plate.shape[:2]
        slope = self.calSlope(rotated_plate)[0]
        print("slope=:", slope)
        xd = np.int0(h * abs(slope))
        if slope < 0:
            src_pts = np.float32([[xd, 0], [w - 1, 0], [w - 1 - xd, h - 1],
                                  [0, h - 1]])
            dst_pts = np.float32([[xd // 2, 0], [w - 1 - xd // 2, 0],
                                  [w - 1 - xd // 2, h - 1], [xd // 2, h - 1]])
        else:
            src_pts = np.float32([[0, 0], [w - 1 - xd, 0], [w - 1, h - 1],
                                  [xd, h - 1]])
            dst_pts = np.float32([[xd // 2, 0], [w - 1 - xd // 2, 0],
                                  [w - 1 - xd // 2, h - 1], [xd // 2, h - 1]])
        return src_pts, dst_pts

    def perspectiveTrans(self, rotated_plate, validMinRects):
        """Perspective transform

        Args:
            rotated_plate (ndarray): rotated image of the plate
            validMinRects (list): valid min rectangles
        """
        for minRect in validMinRects:
            src_pts, dst_pts = self.getSrcAndDstPts(rotated_plate)
            M = cv2.getPerspectiveTransform(src_pts, dst_pts)
            h, w = rotated_plate.shape[:2]
            img_warp = cv2.warpPerspective(rotated_plate, M, (w, h))
            self.saveImg(img_warp, self.out_fpre + "warp.jpg", size=(640, 200))

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
    out_dir = "images_res/" + level + "/plate_detect1"
    out_fpre = number + "_"
    os.makedirs(out_dir, exist_ok=True)
    cv2.resize(img, (800, 600), interpolation=cv2.INTER_CUBIC)
    pd = ColorMatchPlateDetect(img, out_dir, out_fpre)
    if level == "easy":
        filename = out_fpre + "warp.jpg"
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
    # getPlate4Onelevel("medium", "2")