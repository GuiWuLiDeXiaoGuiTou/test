#coding:utf-8
#该程序实现的功能是对图像提取CNN 特征

#############################################
#   Written By Gege Song                       #
#   2018-06-03                              #
#############################################
import numpy as np
import os
import pickle
import struct


import sys
sys.path.append('/home/dell/caffe/python')
import caffe


#---------------------------------------------------------------------   程序入口处，加载模型，输入图片   ----------------------------------------------------

def extraction( imagefile_abs):
    net = LoadModel()
    extractFeature( net,imagefile_abs)



def LoadModel():
    # 运行模型的prototxt
    deployPrototxt =  '/devdata/SGG_experiments/Codes/Classifying_Patent_Images/Extraction_Features/CNN/931_Random_filling/deploy.prototxt'
    # 相应载入的modelfile
    modelFile = '/devdata/SGG_experiments/Codes/Classifying_Patent_Images/Extraction_Features/CNN/931_Random_filling/caffe_alexnet_train_iter_37000.caffemodel'

    caffe.set_mode_gpu()  #采用gpu运算
    net = caffe.Net(deployPrototxt, modelFile,caffe.TEST)  # 加载model和network

    return net

# 提取特征并保存为相应地文件
def extractFeature( net,imagefile_abs):
    # meanfile 也可以用自己生成的
    meanFile = '/devdata/SGG_experiments/Codes/Classifying_Patent_Images/Extraction_Features/CNN/931_Random_filling/imagenet_mean.npy'
    # 图片预处理设置
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})  # 设定图片的shape格式(1,3,28,28)
    transformer.set_transpose('data', (2, 0, 1))  # 改变维度的顺序，由原始图片(28,28,3)变为(3,28,28)
    transformer.set_mean('data', np.load(meanFile).mean(1).mean(1))  # 减去均值，前面训练模型时没有减均值，这儿就不用
    transformer.set_raw_scale('data', 255)  # 缩放到【0，255】之间
    transformer.set_channel_swap('data', (2, 1, 0))  # 交换通道，将图片由RGB变为BGR

    net.blobs['data'].data[...] = transformer.preprocess('data', caffe.io.load_image(imagefile_abs))
    out = net.forward()

    fc7Data = net.blobs['fc7'].data

    path_part = imagefile_abs.split('/')
    fileName = path_part[len(path_part) - 1]

    fea_file = fileName.replace('.jpg', '_AHDH_FeatureData.pkl')

    print ' extract feature ', fc7Data

    if not os.path.exists("/devdata/SGG_experiments/Codes/Classifying_Patent_Images/save_models/"):
        os.makedirs("/devdata/SGG_experiments/Codes/Classifying_Patent_Images/save_models/")

    output = open("/devdata/SGG_experiments/Codes/Classifying_Patent_Images/save_models/" + fea_file, 'wb')
    pickle.dump(fc7Data, output, -1)
    output.close()
    a = 1
    t = 0


extraction('/devdata/SGG_experiments/Images/experience_data_oringal/test/code/407853.jpg')

