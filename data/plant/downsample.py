import cv2
# import def_Gaussian as dg
# import time
import os.path


# import glob

#####################################################################################################################
# 读取文件夹里面的图像数量 并返回filenum
def countFile(dir):
    # 输入文件夹
    tmp = 0
    for item in os.listdir(dir):
        if os.path.isfile(os.path.join(dir, item)):
            tmp += 1
        else:
            tmp += countFile(os.path.join(dir, item))
    return tmp


filenum = countFile("M:\\ColmapData\\NerfPlant1\\images")  # 返回的是图片的张数
print(filenum)

# filenum
n = 2
index = 1  # 保存图片编号
num = 0  # 处理图片计数
for i in range(1, filenum + 1):
    ########################################################
    # 1.读取原始图片
    if index < 10:
        filename = "M:\\ColmapData\\NerfPlant1\\images\\" + str(
            i) + ".jpg"
    elif index < 100:
        filename = "M:\\ColmapData\\NerfPlant1\\images\\" + str(
            i) + ".jpg"
    else:
        filename = "M:\\ColmapData\\NerfPlant1\\images\\" + str(
            i) + ".jpg"
    print(filename)
    original_image = cv2.imread(filename)
    # 2.下采样
    if n == 2:
        img_1 = cv2.pyrDown(original_image)
        img_1 = cv2.pyrDown(img_1)
    if n == 4:
        img_1 = cv2.pyrDown(original_image)
        img_1 = cv2.pyrDown(img_1)
    if n == 8:
        img_1 = cv2.pyrDown(original_image)
        img_1 = cv2.pyrDown(img_1)
        img_1 = cv2.pyrDown(img_1)
    # 3.将下采样图片保存到指定路径当中
    if index < 10:
        cv2.imwrite("M:\\ColmapData\\NerfPlant1\\images_2\\" + str(
            index) + ".jpg", img_1)
    elif index < 100:
        cv2.imwrite("M:\\ColmapData\\NerfPlant1\\images_2\\" + str(
            index) + ".jpg", img_1)
    else:
        cv2.imwrite("M:\\ColmapData\\NerfPlant1\\images_2\\" + str(
            index) + ".jpg", img_1)

    num = num + 1
    print("正在为第" + str(num) + "图片采样......")
    index = index + 1