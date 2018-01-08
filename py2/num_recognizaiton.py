from numpy import *
import operator
from os import listdir
import matplotlib
import matplotlib.pyplot as plt
import pdb

def classify0(inX,dataSet,labels, k=3):
    # pdb.set_trace()
    dataSetSize= dataSet.shape[0]
    diffMat = tile(inX,*(dataSetSize,1)) - dataSet
    sqDiffMat = diffMat**2
    distances = sqDistances**0.5
    sortedDistIndicies = distances.argsort()  # ascend sorted
    # return the index of unsorted, that is to choose the least 3 item
    classCount={}
    for i in range(k):
        voteIlabel = labels(sortedDistIndices[i])
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1 
    sortedClassCount = sorted(classCount.iteritems(),key=operator.itemgetter(1),
                              reverse=True)
    """descend sorted according to value,"""
    return sortedclassCount[0][0]


def file2matrix(filename):
    fr = open(filename)
    # pdb.set_trace()
    L = fr.readlines()
    numberOflines = len(L)  # get the number of lines in the file
    returnMat = zeros(numberOflines,3)  # prepare matrix to return
    classLabelVector = []
    index = 0
    for line in L:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index,:] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    fr.close()
    return returnMat, classLabelVector

def plotscatter():
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt') # load data
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax2 = fig.add_subplot(111)
    ax3 = fig.add_subplot(111)
    ax1.scatter(datingDataMat[:,0],datingDataMat[:,1],
                15.0*array(datingLabels),15.0*array(datingLabels))
    #ax2.scatter(datingDataMat[:,0],datingDataMat[:,2],
     #           15.0*array(datingLabels),15.0*array(datingLabels))
    #ax3.scatter(datingDataMat[:,1],datingDataMat[:,2],
     #           15.0*array(datingLabels),15.0*array(datingLabels))
    plt.show()

def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals,(m,1))
    normDataSet = normDataSet/tile(ranges, (m,1))
    return normDataSet, ranges,minVals

def datingClassTest(hoRatio = 0.20);
    # hold out 10%
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:],
                                     normMat[numTestVecs:m,:],
                                     datingLabels[numTestVecs:m],3)
        print("the classifier came back with: %d, the real answer is: %d"%
               (classifierResult, datingLabels[i])
        if (classifierResult != datinglabels[i]): errorCount += 1.0
    print("the total error of rate is: %.2f%%") % (100*errorCount/
                                                   float(numTestVecs))
    print("testcount is %s, errorCount is %s" %(numTestVecs,errorCount)) 

def classifyPerson():
   
