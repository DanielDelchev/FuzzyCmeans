#!python
import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as mpatches


def visualize(PCAinputFilename,PCAcentroidsFilename,hardClustersFilename,outputFilename):

    PCAinputData = pd.read_csv(PCAinputFilename,index_col='id')
    columnsCount = len(PCAinputData.columns)
    CutInputData = PCAinputData.drop(PCAinputData.columns[range(2,columnsCount)], axis=1)
    
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
    
    plt.title('PCA')
    plt.legend(handles=info)
    plt.savefig(outputFilename)
    plt.close()


if __name__ == '__main__':

    PCAinputFilename = ""
    PCAcentroidsFilename = ""
    hardClustersFilename = "" 
    outputFilename = ""

    argc = len(sys.argv)
    if  argc < 5:
        print ("Usage: visualise.py fullPath-PCAinputFile.csv fullPath-PCAnormalCentroidsFile.csv fullPath-hardClustersFilename.csv fullPath-outputFilename.png")
        exit (0)

    else:
        PCAinputFilename = sys.argv[1]
        PCAcentroidsFilename = sys.argv[2]
        hardClustersFilename = sys.argv[3] 
        outputFilename = sys.argv[4]
 

        visualize(PCAinputFilename,PCAcentroidsFilename,hardClustersFilename,outputFilename)