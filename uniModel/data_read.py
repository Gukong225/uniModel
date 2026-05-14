import io
import os

import scipy.io as sio
import numpy as np
import torch
from sklearn.decomposition import PCA
from tifffile import tifffile

# from build_EMP import build_emp
igbp2hunan = np.array([255, 0, 1, 2, 1, 3, 4, 6, 6, 5, 6, 7, 255])

import numpy as np

EPS = 1e-6

import numpy as np
def lidar_zscore(x, eps=1e-6):
    m = x.mean()
    s = x.std()
    return (x - m) / (s + eps)
def hsi_zscore(img, eps=1e-8):
    # img: H x W x C
    mean = img.reshape(-1, img.shape[-1]).mean(axis=0)
    std  = img.reshape(-1, img.shape[-1]).std(axis=0)
    std[std < eps] = eps
    return (img - mean) / std
def safe_sar_to_db(sar, eps=1e-6):
    sar = np.asarray(sar).astype(np.float32)

    # 1) 负值全部拉到 eps（SAR 强度本来不该是负的）
    sar[sar < eps] = eps

    # 2) 替换 NaN 和 inf
    sar = np.nan_to_num(sar, nan=eps, posinf=eps, neginf=eps)

    # 3) 转 dB
    sar_db = 10.0 * np.log10(sar)

    return sar_db

def safe_zscore(x, eps=1e-6):
    x = np.asarray(x).astype(np.float32)

    # 替换 NaN 或 inf（容错）
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    mean = x.mean()
    std  = x.std()

    # 如果 std=0（常数图像），避免除零
    if std < eps:
        return np.zeros_like(x)

    return (x - mean) / (std + eps)

# 组合流程
def preprocess_sar(sar):
    sar_db   = safe_sar_to_db(sar)
    sar_norm = safe_zscore(sar_db)
    return sar_norm
def load_lc(lc):
    lc[lc == 255] = 12
    lc = igbp2hunan[lc]
    return lc
def pca_whitening(image, number_of_pc):
    shape = image.shape
    image = np.reshape(image, [shape[0]*shape[1], shape[2]])
    number_of_rows = shape[0]
    number_of_columns = shape[1]
    pca = PCA(n_components = number_of_pc)
    image = pca.fit_transform(image)
    pc_images = np.zeros(shape=(number_of_rows, number_of_columns, number_of_pc),dtype=np.float32)
    for i in range(number_of_pc):
        pc_images[:, :, i] = np.reshape(image[:, i], (number_of_rows, number_of_columns))
    return pc_images


def load_data(dataset,baolius=False):
    if dataset == 'Trento':
        image_file_HSI = r'/media/xd132/USER_new/jjh/dataset/Trento/HSI_Trento.mat'
        image_file_LiDAR = r'/media/xd132/USER_new/jjh/dataset/Trento/Lidar_Trento.mat'
        label_file_tr = r'/media/xd132/USER_new/jjh/dataset/Trento/GT_Trento.mat'
        label_file_ts = r'/media/xd132/USER_new/jjh/dataset/Trento/TrLabel.mat'
        image_data_HSI = sio.loadmat(image_file_HSI)
        image_data_LiDAR = sio.loadmat(image_file_LiDAR)
        label_data_tr = sio.loadmat(label_file_tr) 
        label_data_ts = sio.loadmat(label_file_ts)
        image_HSI = image_data_HSI['HSI_Trento']
        image_LiDAR = image_data_LiDAR['Lidar_Trento']
        label = label_data_tr['GT_Trento']
    elif dataset == '2013houston':
        houston_good_bands = [
            1, 2, 3, 4, 6, 7, 9, 10, 11, 12, 13,
            14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
            25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
            36, 37, 39, 40, 41, 42, 43, 44, 45, 46, 47,
            48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58,
            59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69,
            70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80,
            81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 92,
            93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103,
            104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114,
            116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126,
            127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137,
            138, 139, 140, 141, 142, 143
        ]
        houston_good_bands_0based = [b - 1 for b in houston_good_bands]
        image_file_HSI = r'/media/xidian/xd132/JJH/dataset/Houston/HSI_data.mat'
        image_file_LiDAR = r'/media/xidian/xd132/JJH/dataset/Houston/LiDAR_data.mat'
        label_file_tr = r'/media/xidian/xd132/JJH/dataset/Houston/All_Label.mat'
        # label_file_ts = r'./Houston2013/TSLabel.mat'

        image_data_HSI = sio.loadmat(image_file_HSI)
        image_data_LiDAR = sio.loadmat(image_file_LiDAR)
        label_data_tr = sio.loadmat(label_file_tr)

        # label_data_ts = sio.loadmat(label_file_ts)
        image_HSI = image_data_HSI['HSI_data']
        image_HSI = hsi_zscore(image_HSI[:, :, houston_good_bands_0based])
        image_LiDAR = image_data_LiDAR['LiDAR_data']

        label = label_data_tr['All_Label']
    elif dataset == 'Muufl':
        image_file_HSI = r'/media/xd132/USER_new/jjh/dataset/Muufl/HSI_data.mat'
        image_file_LiDAR = r'/media/xd132/USER_new/jjh/dataset/Muufl/LiDAR_data.mat'
        label_file = r'/media/xd132/USER_new/jjh/dataset/Muufl/All_Label.mat'
        image_data_HSI = sio.loadmat(image_file_HSI)
        image_data_LiDAR = sio.loadmat(image_file_LiDAR)
        label_data = sio.loadmat(label_file)
        image_HSI = image_data_HSI['HSI_data']
        image_HSI = hsi_zscore(image_HSI)
        image_LiDAR = image_data_LiDAR['LiDAR_data']
        image_LiDAR = lidar_zscore(image_LiDAR)
        label = label_data['All_Label']
    elif dataset == 'Augsburg':
        image_file_HSI = r'/media/xd132/USER_new/jjh/dataset/Augsburg/HSI_data.mat'
        image_file_LiDAR = r'/media/xd132/USER_new/jjh/dataset/Augsburg/SAR_data.mat'
        label_file = r'/media/xd132/USER_new/jjh/dataset/Augsburg/All_Label.mat'
        image_data_HSI = sio.loadmat(image_file_HSI)
        image_data_LiDAR = sio.loadmat(image_file_LiDAR)
        label_data = sio.loadmat(label_file)
        image_HSI = image_data_HSI['HSI_data']
        image_HSI = hsi_zscore(image_HSI)
        image_LiDAR = image_data_LiDAR['SAR_data']
        image_LiDAR = preprocess_sar(image_LiDAR)
        label = label_data['All_Label']
    elif dataset == 'Berlin':
        image_file_HSI = r'/media/xd132/USER_new/jjh/dataset/Berlin/HSI_data.mat'
        image_file_LiDAR = r'/media/xd132/USER_new/jjh/dataset/Berlin/SAR_data.mat'
        label_file = r'/media/xd132/USER_new/jjh/dataset/Berlin/All_Label.mat'
        image_data_HSI = sio.loadmat(image_file_HSI)
        image_data_LiDAR = sio.loadmat(image_file_LiDAR)
        label_data = sio.loadmat(label_file)
        image_HSI = image_data_HSI['HSI_data']
        image_LiDAR = image_data_LiDAR['SAR_data']
        label = label_data['All_Label']
    elif dataset == 'Hunan' :
        image_file_HSI = r'/media/xd132/USER_new/jjh/dataset/Hunan/s2/s2_28965.tif'
        image_file_LiDAR = r'/media/xd132/USER_new/jjh/dataset/Hunan/s1/s1_28965.tif'
        label_file = r'/media/xd132/USER_new/jjh/dataset/Hunan/lc_converted/lc_28965.tif'

        image_data_HSI = tifffile.imread(image_file_HSI)
        image_data_LiDAR = tifffile.imread(image_file_LiDAR)
        label_data = tifffile.imread(label_file)
        image_HSI = image_data_HSI
        image_LiDAR = image_data_LiDAR

        label = label_data
    else:
        raise Exception('dataset does not find')

    if baolius == False:
        image_HSI = image_HSI.astype(np.float32)
        image_LiDAR = image_LiDAR.astype(np.float32)
        label = label.astype(np.int64)
    return image_HSI, image_LiDAR, label
 
def read_numbers_from_file(file_path):
    # 读取文件并提取数字
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            # 去掉行首尾的空白字符，并按空格分割
            parts = line.strip().split()
            if len(parts) == 2:  # 确保行中有两个数字
                value = int(parts[1])  # 提取第二列的数字
                data.append(value)
    data_array = np.array(data)
    # 打印结果
    return data_array

# 使用示例


def prepare_padded_scene_for_inference(
    dataset: str,
    feature_type: str,
    windowsize: int,
    baolius: bool = False,
):
    """
    与 readdata 前半段一致：load_data -> 边缘 padding ->（Hunan 特殊处理）-> PCA/none，
    供整图滑窗推理，避免与训练 patch 预处理不一致。

    Returns:
        image_hsi:    (Hp, Wp, C) float32，已 padding
        image_lidar:  (Hp, Wp, C2) float32，已 padding
        label_orig:   (H, W) int64，原始尺寸 GT（0 为背景，1..K 为类别）
        halfsize:     int，(windowsize - 1) // 2
    """
    or_image_HSI, or_image_LiDAR, or_label = load_data(dataset, baolius)
    halfsize = int((windowsize - 1) / 2)
    if or_image_LiDAR.ndim < 3:
        or_image_LiDAR = np.expand_dims(or_image_LiDAR, 2)

    image = np.pad(
        or_image_HSI,
        ((halfsize, halfsize), (halfsize, halfsize), (0, 0)),
        "edge",
    )
    image_LiDAR = np.pad(
        or_image_LiDAR,
        ((halfsize, halfsize), (halfsize, halfsize), (0, 0)),
        "edge",
    )

    if dataset == "Hunan":
        image_LiDAR = image_LiDAR.repeat(72, axis=2)
        image_LiDAR = pca_whitening(image_LiDAR, number_of_pc=3)

    if feature_type == "PCA":
        image_hsi_out = pca_whitening(image, number_of_pc=10)
        image_lidar_out = np.copy(image_LiDAR)
    elif feature_type == "none":
        image_hsi_out = np.copy(image)
        image_lidar_out = np.copy(image_LiDAR)
    else:
        raise Exception("feature_type does not find (use PCA or none)")

    image_hsi_out = image_hsi_out.astype(np.float32)
    image_lidar_out = image_lidar_out.astype(np.float32)
    or_label = or_label.astype(np.int64)

    return image_hsi_out, image_lidar_out, or_label, halfsize


def readdata(type, dataset, windowsize, train_num, val_num, num,proportion,baolius = False):

    or_image_HSI, or_image_LiDAR, or_label = load_data(dataset,baolius)
    halfsize = int((windowsize-1)/2)
    number_class = np.max(or_label).astype(np.int64)
    if or_image_LiDAR.ndim < 3:
        or_image_LiDAR = np.expand_dims(or_image_LiDAR, 2)

    image = np.pad(or_image_HSI, ((halfsize, halfsize), (halfsize, halfsize), (0, 0)), 'edge')

    image_LiDAR = np.pad(or_image_LiDAR, ((halfsize, halfsize), (halfsize, halfsize), (0, 0)), 'edge')

    label = np.pad(or_label, ((halfsize, halfsize), (halfsize, halfsize)), 'constant',constant_values=0)
    if dataset == 'Hunan':
        image_LiDAR = image_LiDAR.repeat(72, axis=2)
        image_LiDAR = pca_whitening(image_LiDAR, number_of_pc = 3)
    if type == 'PCA':
        image1 = pca_whitening(image, number_of_pc = 10)
        image_LiDAR1 = np.copy(image_LiDAR)
    elif type == 'none':
        image1 = np.copy(image)
        image_LiDAR1 = np.copy(image_LiDAR)
    else:
        raise Exception('type does not find')
        
    
    n = np.zeros(number_class,dtype=np.int64)
    for i in range(number_class):
        temprow, tempcol = np.where(label == i + 1)
        n[i] = len(temprow)    #每一类的的样本数目
    total_num = np.sum(n)
    if train_num > 0:
        nTrain_perClass = np.ones(number_class,dtype=np.int64) * train_num
        for i in range(number_class):
            if n[i] <=  nTrain_perClass[i]:
                nTrain_perClass[i] = 15
        nvalid_perClass = val_num * n
        # nvalid_perClass = n - nTrain_perClass
        nvalid_perClass = nvalid_perClass.astype(np.int32)

    elif train_num == -1:

        file_path = '/media/xd132/USER_new/jjh/MutilCLIP/classname/num_' + dataset + '.txt'
        nTrain_perClass = read_numbers_from_file(file_path)
        nvalid_perClass = val_num * n
        nvalid_perClass = n - nTrain_perClass
        nvalid_perClass = nvalid_perClass.astype(np.int32)

    elif train_num == 0:
        nTrain_perClass = np.ones(number_class, dtype=np.int64)

        for i in range(number_class):
            nTrain_perClass[i] = proportion * n[i]
        nvalid_perClass = val_num * n
        nvalid_perClass = n - nTrain_perClass
        nvalid_perClass = nvalid_perClass.astype(np.int32)
    index = []
    flag = 0
    fl = 0
    bands = np.size(image,2)
    bands_LIDAR = np.size(image_LiDAR,2)
    validation_image = np.zeros([np.sum(nvalid_perClass), windowsize, windowsize, bands], dtype=np.float32)
    validation_image_LIDAR = np.zeros([np.sum(nvalid_perClass), windowsize, windowsize, bands_LIDAR], dtype=np.float32)
    validation_label = np.zeros(np.sum(nvalid_perClass), dtype=np.int64)
    train_image = np.zeros([np.sum(nTrain_perClass), windowsize, windowsize, bands], dtype=np.float32)
    train_image_LIDAR = np.zeros([np.sum(nTrain_perClass), windowsize, windowsize, bands_LIDAR], dtype=np.float32)
    train_label = np.zeros(np.sum(nTrain_perClass),dtype=np.int64)
    train_index = np.zeros([np.sum(nTrain_perClass), 2], dtype = np.int32)              
    val_index =  np.zeros([np.sum(nvalid_perClass), 2], dtype = np.int32)
    for i in range(number_class):
        temprow, tempcol = np.where(label == i + 1)
        matrix = np.zeros([len(temprow),2], dtype=np.int64)
        matrix[:,0] = temprow
        matrix[:,1] = tempcol
        np.random.seed(num)
        np.random.shuffle(matrix)

        temprow = matrix[:,0]
        tempcol = matrix[:,1]
        index.append(matrix)

        for j in range(nTrain_perClass[i]):
            train_image[flag + j, :, :, :] = image[(temprow[j] - halfsize):(temprow[j] + halfsize + 1),
                                            (tempcol[j] - halfsize):(tempcol[j] + halfsize + 1)]
            train_image_LIDAR[flag + j, :, :, :] = image_LiDAR[(temprow[j] - halfsize):(temprow[j] + halfsize + 1),
                                            (tempcol[j] - halfsize):(tempcol[j] + halfsize + 1)]
            train_label[flag + j] = i
            train_index[flag + j] = matrix[j,:]
        flag = flag + nTrain_perClass[i]
        for j in range(nTrain_perClass[i], nTrain_perClass[i] + nvalid_perClass[i]):
            validation_image[fl + j-nTrain_perClass[i], :, :,:] = image[(temprow[j] - halfsize):(temprow[j] + halfsize + 1),
                                                   (tempcol[j] - halfsize):(tempcol[j] + halfsize + 1)]
            validation_image_LIDAR[fl + j-nTrain_perClass[i], :, :,:] = image_LiDAR[(temprow[j] - halfsize):(temprow[j] + halfsize + 1),
                                                   (tempcol[j] - halfsize):(tempcol[j] + halfsize + 1)]
            validation_label[fl + j-nTrain_perClass[i] ] = i
            val_index[fl + j-nTrain_perClass[i]] = matrix[j,:]
        fl =fl + nvalid_perClass[i]

    return train_image, train_image_LIDAR, train_label, validation_image, validation_image_LIDAR, validation_label,\
           nTrain_perClass, nvalid_perClass,train_index, val_index, index, image, image_LiDAR, label,total_num
