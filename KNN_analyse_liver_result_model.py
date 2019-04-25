# !/usr/bin/env python
# -*- coding: utf8 -*-
import os,re,sys
from numpy import *
import operator
# for the error name 'operator' is not defined
#将文本记录转换为 NumPy 的解析程序
trainfilename='train_sampledataset.txt'
testfilename='test_sampledataset.txt'
def file2matrix(filename):
   """
   Desc:
       导入训练数据
   parameters:
       filename: 数据文件路径
   return:
       数据矩阵 returnMat 和对应的类别 classLabelVector
   """
   fr = open(filename,'r', encoding='UTF-8')
   # 获得文件中的数据行的行数
   lines=fr.readlines()
   numberOfLinesuse = len(lines)-1
   #print("行数",numberOfLinesuse)
   # 对应数据的列数
   numberOfcluomsuse = len(lines[0].strip('\n').split('\t'))-3
   #print("列数",numberOfcluomsuse)
   # 生成对应的空矩阵
   # 例如：zeros(2，3)就是生成一个 2*3的矩阵，各个位置上全是 0
   returnMat = zeros((numberOfLinesuse, numberOfcluomsuse))  # prepare matrix to return
   classLabelVector = []  # prepare labels return
   fr = open(filename,'r', encoding='UTF-8')
   lines = fr.readlines()
   index = 0
   for line in lines[1:]:
       # str.strip([chars]) --返回已移除字符串头尾指定字符所生成的新字符串
       line = line.strip()
       # 以 '\t' 切割字符串
       listFromLine = line.split('\t')
       # 每列的属性数据
       #print(listFromLine[3:])
       returnMat[index, :] = listFromLine[3:]
       # 每列的类别数据，就是 label 标签数据
       classLabelVector.append(int(listFromLine[2])) #标签，0，1 0阴性，1阳性
       index += 1
   # 返回数据矩阵returnMat和对应的类别classLabelVector
   return returnMat, classLabelVector

#test
#trainMat,trainLabel=file2matrix(trainfilename)
#print(trainMat,trainLabel)

#分析数据
def plotanalysefun(trainMat,trainLabel):
    import matplotlib
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(trainMat[:, 4], trainMat[:, 9], 15.0*array(trainLabel), 15.0*array(trainLabel))  #查看第一列，第二列数据PARP2和CHEK1基因的
    plt.show()

def sigmoidNorm(dataSet):
    """
    Desc:
        归一化特征值，消除特征之间量级不同导致的影响
    parameter:
        dataSet: 数据集
    return:
        归一化后的数据集 normDataSet. ranges和minVals即最小值与范围，并没有用到
    归一化公式：
        Y = (X-Xmin)/(Xmax-Xmin)
        其中的 min 和 max 分别是数据集中的最小特征值和最大特征值。该函数可以自动将数字特征值转化为0到1的区间。
        # 计算每种属性的最大值、最小值、范围
        minVals = dataSet.min(0)
        maxVals = dataSet.max(0)
        # 极差
        ranges = maxVals - minVals
        normDataSet = zeros(shape(dataSet))
        m = dataSet.shape[0]
        # 生成与最小值之差组成的矩阵
        normDataSet = dataSet - tile(minVals, (m, 1))
        # 将最小值之差除以范围组成矩阵
        normDataSet = normDataSet / tile(ranges, (m, 1))  # element wise divide
        但不适用于染色体数据
        使用sigmod进行归一化
        使用zscore归一化
    """
    normDataSet = zeros(shape(dataSet))
    normDataSet = 1.0/1+exp(-dataSet)
    return normDataSet

def zscoreNorm(dataSet):
    #数值太小了
    from sklearn import preprocessing
    normDataSet = zeros(shape(dataSet))
    normDataSet=preprocessing.scale(dataSet)
    return normDataSet

#test2
#normtrainMat = sigmoidNorm(trainMat)
#print(normtrainMat) #做激活函数不错，但这里归一化效果比较诡异，不过没有其他各适合的


def classify0(inX, dataSet, labels, k):
    """
    对于每一个在数据集中的数据点：
    计算目标的数据点（需要分类的数据点）与该数据点的距离
    将距离排序：从小到大
    选取前K个最短距离
    选取这K个中最多的分类类别
    返回该类别来作为目标数据点的预测值
    """
    dataSetSize = dataSet.shape[0]
    # 距离度量 度量公式为欧氏距离
    diffMat = tile(inX, (dataSetSize, 1))- dataSet #inX是目标数据待测试样本的数据，与训练集中的其他数据进行对比。
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5
    #print(distances)
    # 将距离排序：从小到大
    sortedDistIndicies = distances.argsort()  #返回的是这个数组的大小的索引，从小到大
    # 选取前K个最短距离， 选取这K个中最多的分类类别
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]] #按这个索引返回的是相应大小排序的标签
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1 #每个标签的出现次数
    #python3中用items()替换python2中的iteritems()
    #key=operator.itemgetter(1)根据字典的值进行排序
    #key=operator.itemgetter(0)根据字典的键进行排序
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    #print(sortedClassCount) # [(标签，次数)，(另一个标签，次数)]
    #print(sortedClassCount[0])
    return sortedClassCount[0][0]

#result = classify0(trainMat[0], normtrainMat, trainLabel, 5)
#print(result)


def testKNNresult():
    #获得训练数据构建矩阵
    trainMat, trainLabel = file2matrix(trainfilename)
    #数据归一化
    normtrainMat = sigmoidNorm(trainMat)
    fr = open(testfilename, 'r', encoding='UTF-8')
    # 获得文件中的数据行的行数
    lines = fr.readlines()
    numberOfLinesuse = len(lines) - 1
    numberOfcluomsuse = len(lines[0].strip('\n').split('\t')) - 3
    testMat = zeros((1, numberOfcluomsuse))
    #print(testMat)
    rightnum=0
    errornum=0
    fr = open(testfilename, 'r', encoding='UTF-8')
    lines = fr.readlines()
    for line in lines[1:]:
        line = line.strip()
        listFromLine = line.split('\t')
        #要保证矩阵类型一致
        testMat[0:,]=listFromLine[3:]
        inX=sigmoidNorm(testMat[0:,]) #同样归一化
        tranresult = classify0(inX, normtrainMat, trainLabel, 10)
        print("这个样本",listFromLine[1],"预测结果是",tranresult,"实际结果是",listFromLine[2])
        #比较训练模型结果与测试结果标签是否相同
        if int(tranresult) == int(listFromLine[2]):
            rightnum+=1
        else:
            errornum+=1
    rightratio=float(rightnum/numberOfLinesuse)*100
    errorratio=float(errornum/numberOfLinesuse)*100
    print("该模型此次预测结果正确率为%.2f %%，错误率为%.2f %%" %(rightratio,errorratio))


testKNNresult()