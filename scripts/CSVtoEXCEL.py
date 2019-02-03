#!python
import os
import sys
import pandas as pd
import numpy as np

def convert(inputFile="./../data/myPrefix_similarityRelation.csv", outputFile="./../data/myPrefix_similarityRelation.xlsx"):

    inputData = pd.read_csv(inputFile,index_col='SAMPLES')
    writer = pd.ExcelWriter(outputFile)
    inputData.to_excel(writer, index = True ,header=True)
    writer.save()


if __name__ == '__main__':
    
    inputFilename = ""
    outputFilename = ""

    argc = len(sys.argv)
    if  argc < 3:
        print ("Usage: CSVtoEXCEL.py fullPath-inputFilename.csv fullPath-outputFilename.xlsx")
        exit (0)

    else:
        inputFilename = sys.argv[1]
        outputFilename = sys.argv[2]
    
    convert(inputFilename, outputFilename)