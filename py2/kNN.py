#encoding:utf-8
from numpy import *
import operator

def createDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group,labels

def classify0(inX,dataSet,labels,k):
    # return array rows  if shape[1] return array columns
    dataSetSize = dataSet.shape[0]
    # Two arrays subtract,get new array
    diffMat = tile(inX,(dataSetSize,1)) - dataSet
    # squared
    sqDiffMat = diffMat ** 2
    # Sum,return one dimension array
    sqDistances = sqDiffMat.sum(axis=1)
    # distance between test_point and other point
    distances = sqDistances ** 0.5
    # Sort,return index of arrays that sorted from small to large
    sortedDistIndicies = distances.argsort()
    # Define a null Dict
    classCount = {}
    for i in range(k):
        # return labels of k points nearest
        voteIlabel = labels[sortedDistIndicies[i]]
        # stored in dict
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    # Sort classCount.iteritems() output key-value pairs,True means Desc
    sortedClassCount = sorted(classCount.iteritems(),
                              key=operator.itemgetter(1),reverse = True)
    return sortedClassCount[0][0]


if __name__ ==  '__main__':
    group, labels = createDataSet()
    label = classify0([0.1,0.3],group,labels,3)
    print(label)
