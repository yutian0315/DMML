import os
import cv2
import pickle
import numpy as np
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from config import DEFAULT_PICTURE_SIZE, DATASET_DIR, LABEL_TRANSFORM_PATH


def pictureToArray(dir):#将图片转换成array格式
    try:
        picture = cv2.imread(dir)
        if picture is not None:
            picture = cv2.resize(picture, DEFAULT_PICTURE_SIZE)
            return img_to_array(picture)
        else:
            return np.array([])
    except Exception as e:
        print(f"Error : {e}")
        return None


def loadData():#加载图片
    pictureList, labelList = [], []#存放图片及图片标签的列表
    rootDir = os.listdir(DATASET_DIR)#图片路径
    for pictureFolder in rootDir:#对路径里的所有图片文件夹
        picture_files = os.listdir(f"{DATASET_DIR}/{pictureFolder}")
        for picture in picture_files:#对图片文件夹的所有图片文件
            pictureName = f"{DATASET_DIR}/{pictureFolder}/{picture}"#精确到每张图片的路径
            if pictureName.endswith((".jpg", ".JPG")):#检查图片是否以“.jpg”或“.JPG”作为文件扩展名结尾
                pictureList.append(pictureToArray(pictureName))#将该当前图片添加到图片列表
                labelList.append(pictureFolder)#将该当前图片标签到标签列表

    labelBinarizer = LabelBinarizer()#创建了一个 LabelBinarizer 实例
    pictureLabels = labelBinarizer.fit_transform(labelList)#对labelList进行编码，转换为 One-Hot 编码形式
    pickle.dump(labelBinarizer, open(LABEL_TRANSFORM_PATH, 'wb'))#保存标签编码器

    pictureList = np.asarray(pictureList)#将 pictureList转换为 numpy数组格式，以便后续进行数据操作。
    #将数据集按6：4进行划分，训练集占60%
    xTrain, xVal, yTrain, yVal = train_test_split(pictureList, pictureLabels, test_size=0.6, random_state=42)
    # 将上述得到的40%数据集再划分，验证集和测试集各占20%
    Xtest, xVal, Ytest, yVal = train_test_split(xVal, yVal, test_size=0.5, random_state=42)
    return xTrain, xVal, Xtest, yTrain, yVal, Ytest, len(labelBinarizer.classes_)
