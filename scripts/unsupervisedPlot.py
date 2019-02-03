#!python
import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as mpatches


def even(number):
    return ((number % 2) == 0)  

def visualize(inputFilename,outputFilename, metricIndex):

    #read file
    points = pd.read_csv(inputFilename)
    #get Ks for X axis
    Xs = points.iloc[:,0]
    #get metric values for Y axis
    Ys = points.iloc[:,1]

    ''' metrics which seek maximum as an optimal value are required to be with an odd integer index
    and the ones with even index seek minimum. The first type are graphed red, and the 2nd type in blue'''
    
    last_index = outputFilename.rfind('_')
    title_ = outputFilename[:last_index]
    last_index = title_.rfind('_')
    title_ = title_[last_index+1:]
    
    _color = 'red'
    if even(metricIndex):
        _color = 'blue'
    
    plt.title(title_)
    plt.xlabel('K')
    plt.ylabel('metric value')
    
    plt.scatter(Xs,Ys,color=_color)
    plt.savefig(outputFilename)
    plt.close()
    
'''  
    PCAcentroidsData = pd.read_csv(PCAcentroidsFilename,index_col='id')
    columnsCount = len(PCAcentroidsData.columns)
    CutCentroidData = PCAcentroidsData.drop(PCAcentroidsData.columns[range(2,columnsCount)], axis=1)    
    
    hardClusterData = pd.read_csv(hardClustersFilename,index_col='id')

    CutInputData['Cluster'] =  hardClusterData.iloc[:,0]   
    
    CutInputData['Cluster'] = pd.Categorical(CutInputData['Cluster'])
    clusters = CutInputData['Cluster'].cat.categories
    
    clustersCount = (len(clusters))
    colors = cm.rainbow(np.linspace(0, 1, clustersCount))
    info = []
    for _label,col in zip(clusters,range(0,clustersCount)):
        plt.scatter(CutInputData[CutInputData['Cluster']==_label].iloc[:,0], CutInputData[CutInputData['Cluster']==_label].iloc[:,1],color = colors[col])
        info.append(mpatches.Patch(color=colors[col], label= ('Cluster'+str(_label))))
    
    info.append(mpatches.Patch(color='black', label='Centroid'))
    plt.scatter(CutCentroidData.iloc[:,0],CutCentroidData.iloc[:,1] ,color = 'black')
    
    plt.legend(handles=info)
    plt.savefig(outputFilename)
    plt.close()
'''

if __name__ == '__main__':

    inputFilename = "" 
    outputFilename = ""
    metricIndex = -1
    
    argc = len(sys.argv)
    if  argc < 2:
        print ("Usage: unsupervisedPlot.py fullPath-inputFile.csv fullPath-outputFilename.png")
        exit (0)

    else:
        inputFilename = sys.argv[1]
        outputFilename = sys.argv[2]
        metricIndex = int(sys.argv[3])

        visualize(inputFilename ,outputFilename, metricIndex)