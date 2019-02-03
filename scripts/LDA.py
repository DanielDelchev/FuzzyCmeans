#!python
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.datasets.samples_generator import make_blobs

from pandas.tools.plotting import parallel_coordinates
import sys
import os
import numpy as np
import matplotlib.cm as cm
import matplotlib.patches as mpatches


def LDAvisualize(inputFilename,centroidsFilename,hardClustersFilename,outputFilename):
    
    inputData = pd.read_csv(inputFilename,index_col='id')
    
    centroidsData = pd.read_csv(centroidsFilename,index_col='id')  
    
    hardClusterData = pd.read_csv(hardClustersFilename,index_col='id')

    inputData['Cluster'] =  hardClusterData.iloc[:,0]   
   
    X = inputData.drop(['Cluster'],axis=1)
    y = inputData['Cluster']
    
    lda = LDA(n_components=2)
    lda_transformed = pd.DataFrame(lda.fit_transform(X, y),columns=['Coordinate 1','Coordinate 2'],)
    lda_transformed.index += 1
    
    lda_transformed['Cluster'] = hardClusterData.iloc[:,0]
    
    inputData['Cluster'] = pd.Categorical(inputData['Cluster'])
    clusters = inputData['Cluster'].cat.categories
    clustersCount = (len(clusters))
    colors = cm.rainbow(np.linspace(0, 1, clustersCount))
    info = []
    for _label,col in zip(clusters,range(0,clustersCount)):
        plt.scatter( lda_transformed[y==_label]['Coordinate 1'] , lda_transformed[y==_label]['Coordinate 2'] ,color = colors[col])
        info.append(mpatches.Patch(color=colors[col], label= ('Cluster'+str(_label))))
    
    
    centroids = lda.transform(centroidsData)
    info.append(mpatches.Patch(color='black', label='Centroid'))
    xs = list(zip(*centroids))[0]
    ys = list(zip(*centroids))[1]

    plt.title('LDA')
    plt.scatter(xs, ys ,color = 'black')
    
    plt.legend(handles=info)
    plt.savefig(outputFilename)
    plt.close()
    
    
if __name__ == '__main__':

    inputFilename = ""
    centroidsFilename = ""
    hardClustersFilename = "" 
    outputFilename = ""

    argc = len(sys.argv)
    if  argc < 5:
        print ("Usage: visualise.py fullPath-inputFile.csv fullPath-standartizedCentroidsFile.csv fullPath-hardClustersFilename.csv fullPath-outputFilename.png")
        exit (0)

    else:
        inputFilename = sys.argv[1]
        centroidsFilename = sys.argv[2]
        hardClustersFilename = sys.argv[3] 
        outputFilename = sys.argv[4]

        LDAvisualize(inputFilename,centroidsFilename,hardClustersFilename,outputFilename)