#include <iostream>
#include <fstream>
#include <string>
#include <cstring>
#include <sstream>
#include <set>
#include <map>
#include <assert.h>
#include <vector>
#include <functional>
#include <tuple>
#include <utility>
#include <algorithm>
#include <cmath>

#include <stdio.h>
#include <stdlib.h>

#include "tnt_array1d.h"
#include "tnt_array1d_utils.h"
#include "tnt_array2d.h"
#include "tnt_array2d_utils.h"
#include "jama_eig.h"


using namespace std;
using namespace TNT;
using namespace JAMA;

//substract mean of column from each row in that column
	void centreMatrix (Array2D<double>& dataMatrix, Array1D<double>& means) {
	   size_t columns = dataMatrix.dim2();
	   size_t rows = dataMatrix.dim1();
	   for (size_t column=0; column<columns; column++) {
		   double mean = 0;
		   for (size_t row=0; row<rows; row++) {
			   mean += dataMatrix[row][column];
		   }

		   mean /= rows;

		   // store the mean
		   means[column] = mean;

		   // subtract the mean
		   for (size_t row=0; row<rows; row++) {
			   dataMatrix[row][column] -= mean;
		   }
	   }
	}

	double computeCovariance(const Array2D<double>& dataMatrix, size_t featureXindex, size_t featureYindex) {
	   //SUM((Xi-mean(X))*(Yi-mean(Y)))/n-1
	   //but mean(X) = mean(Y) = 0 (dataMatrix has been centred) (should have been)
	   double cov = 0;
	   //n
	   size_t dim = dataMatrix.dim1();

	   for (size_t index=0; index<dim; index++) {
		   cov += dataMatrix[index][featureXindex] * dataMatrix[index][featureYindex];
	   }


	   return (cov / (dim-1));//n-1
	}

	Array2D<double> computeCovarianceMatrix(const Array2D<double> & dataMatrix) {
		size_t dim = dataMatrix.dim2();
		Array2D<double> result (dim,dim);

		for (size_t row=0; row<dim; row++) {
			for (size_t column = row; column<dim; column++) {
				result[row][column] = computeCovariance(dataMatrix, row, column);
			}
		}

		// matrix is symetrical below and above the main diagonal
		for (size_t row=1; row<dim; row++) {
			for (size_t column=0; column<row; column++) {
				result[row][column] = result[column][row];
			}
		}
		return result;
	}

	void eigen(const Array2D<double> & covarianceMatrix, Array2D<double>& eigenVectorMatrix, Array1D<double>& eigenValues) {
		Eigenvalue<double> EIG(covarianceMatrix);
        EIG.getV(eigenVectorMatrix);
		EIG.getRealEigenvalues(eigenValues);
	}

	const Array2D<double> transpose(const Array2D<double>& src) {
		size_t rows = src.dim1();
		size_t columns = src.dim2();
		Array2D<double> transposed(columns,rows);

		for (size_t row=0; row<rows; row++) {
			for (size_t column=0; column<columns; column++) {
				transposed[column][row] = src[row][column];
			}
		}
		return transposed;
	}

	Array2D<double> multiply(const Array2D<double>& leftM, const Array2D<double>& rightM) {
		if(leftM.dim2() != rightM.dim1()){
            perror("PCA Invalid matrices passed for multiplication!\n");
            exit (1);
		}
		size_t leftMrows = leftM.dim1();
        size_t rightMrows = rightM.dim1();
		size_t rightMcolumns = rightM.dim2();

		Array2D<double> result(leftMrows,rightMcolumns);

		for (size_t row=0; row<leftMrows; row++) {
			for (size_t column=0; column<rightMcolumns; column++) {
				double sum = 0;
				for (size_t index=0; index<rightMrows; index++) {
					sum += leftM[row][index] * rightM[index][column];
				}
				result [row][column] = sum;
			}
		}
		return result;
	}


Array2D<double> readData(const std::string& CSVfilename,std::vector<int>* labels = nullptr, bool skipLabels = false,bool skipHeadings=false,char sep = ','){
    const char* separator = &sep;

    size_t featuresCount = 0;

    std::fstream fileStream(CSVfilename.c_str(),std::ios::in);
    if (!fileStream.is_open()){
        perror("PCA Could not open file!\n");
        exit (1);
    }

    std::string line;

    getline(fileStream,line);

    for (const auto& ch : line){
        if (ch==separator[0]){
            featuresCount++;
        }
    }
    if (skipLabels == false){
        featuresCount++;
    }

    std::vector<std::vector<double> > buffer;
    std::vector<double> subbuffer;
    subbuffer.reserve(featuresCount);

    //create 2darray

    if(skipHeadings==true){
        //get next line
        while (getline(fileStream,line))
        {
            //split line by " " , "," "tab"
            char* chunk = strtok(const_cast<char*>(line.c_str()),separator);
            subbuffer.clear();

            if (skipLabels){
                labels->push_back(atoi(chunk));
                while ( (chunk = strtok(nullptr,separator))!=nullptr ){
                    subbuffer.push_back(atof(chunk));
                }
            }
            else{
                do{ subbuffer.push_back(atof(chunk));}
                while((chunk = strtok(nullptr,separator))!=nullptr );
            }
            buffer.push_back(subbuffer);
        }
    }
    else{
        do{
            //split line by " " , "," "tab"
            char* chunk = strtok(const_cast<char*>(line.c_str()),separator);
            subbuffer.clear();

            if (skipLabels){
                labels->push_back(atoi(chunk));
                while ( (chunk = strtok(nullptr,separator))!=nullptr ){
                    subbuffer.push_back(atof(chunk));
                }
            }
            else{
                do{subbuffer.push_back(atof(chunk));}
                while((chunk = strtok(nullptr,separator))!=nullptr );
            }
            buffer.push_back(subbuffer);
        }while(getline(fileStream,line));
    }
    //close filestream
    fileStream.close();

    size_t samples = buffer.size();
    Array2D<double> result(samples,featuresCount);

    for (size_t index=0;index<samples;index++){
        memcpy(result[index],buffer[index].data(),sizeof(double)*featuresCount);
    }

    return result;
}


struct PCA{

    PCA(){};

    PCA(const Array2D<double>& data){
        fit(std::ref(data));
    }

    void fit(const Array2D<double>& data){
        Array2D<double> copiedData = data.copy();
        size_t columns = copiedData.dim2();
        means = Array1D<double>(columns);
        centreMatrix(copiedData,means);

        Array2D<double> covarMatrix = computeCovarianceMatrix(copiedData);

        size_t dim = covarMatrix.dim1();

        Array2D<double> eigenVectorMatrix (dim, dim); //(not inversed yet)
        eigenValues = Array1D<double>(dim);

        eigen(covarMatrix, eigenVectorMatrix, eigenValues);


        //assuming eigenValue [i] correspondes to eigenVector[i]

        // transposed the matrix so that the eigen vectors are in the rows
        // so that it is easier to sort accoring to eigen values
         Array2D<double> tempEigenVectors = transpose(eigenVectorMatrix);
         Array2D<double> sortedEigenVectors (dim,dim);
         Array1D<double> sortedEigenValues (dim);

         typedef std::pair<double,size_t> couple;

         std::vector<couple> couples;
         for (size_t index=0;index<dim;index++){
            couples.push_back(couple(eigenValues[index],index));
         }


        const int EXP = 40;
        const int EPS = pow(10,-EXP);
        //fabs???
        std::function<bool(const couple&,const couple&)> CMP = [&](const couple& one, const couple& other){
            return (one.first - other.first > EPS);
        };

        sort(couples.begin(),couples.end(),CMP);


        for (size_t index = 0; index<dim;index++){
            sortedEigenValues[index] = couples[index].first;
            memcpy(sortedEigenVectors[index],tempEigenVectors[couples[index].second],sizeof(double)*dim);
        }

         //when all eigen vectors are used transposed and inversed eigen matrix are the same
         //when we have the inversed eigen vector matrix
         //we have the transformation matrix to put any vector in the basis of the eigen vectors
         // IinversedEigenVectorMatrix x someVectorT = (someVector in new basis)T
         // we already transponated the matrix before sorting the rows acording to eigenvalues
         // so we don't have to transpose again to get the inverse
         eigenVectorMatrixInversed = sortedEigenVectors.copy();
         eigenValues = sortedEigenValues.copy();
    }

    Array2D<double> eigenTransform(const Array2D<double>& matrix){

       Array2D<double> dataM = matrix.copy();

       size_t columns = dataM.dim2();
	   size_t rows = dataM.dim1();
	   for (size_t column=0; column<columns; column++) {
		   // subtract the mean
		   for (size_t row=0; row<rows; row++) {
			   dataM[row][column] -= means[column];
		   }
	   }

	   Array2D<double> transposedCentredData = transpose(dataM);
       dataM = transposedCentredData;

       Array2D<double> transposedResult = multiply(eigenVectorMatrixInversed,dataM);

       dataM = transpose(transposedResult);

       return dataM;

    }

    Array2D<double> eigenTransformBack(const Array2D<double>& matrix) = delete;//

    Array1D<double> means;//

    Array2D<double> eigenVectorMatrixInversed; //

    Array1D<double> eigenValues; //

};


void writeDownFile(const std::string& outputFilename,const Array2D<double>& data, bool labeled, bool headings,const std::vector<int>& labels,const std::string& heading){

        ofstream fileStream(outputFilename,ios::out|ios::trunc);
        if (!fileStream.is_open()){
            perror("PCA Could not open/create file!\n");
        }

        size_t rows = data.dim1();
        size_t columns = data.dim2();

        if (labeled && rows!=labels.size()){
                std::cout<<labeled<<outputFilename<<std::endl;
            std::cout<<"PCA Error in writing file! Labels less than rows in matrix!\n";
            return;
        }

        if (headings){
            fileStream<<heading<<"\n";
        }

        for (size_t index=0;index<rows;index++){
            std::string line="";
            if (labeled){
                line += std::to_string(labels[index]);
                line += ", ";
            }
            for (size_t subIndex=0;subIndex<columns;subIndex++){
                line += std::to_string(data[index][subIndex]);
                if (subIndex!=columns-1){
                    line += ", ";
                }
                else{
                    if (index!=rows-1){
                        line += "\n";
                    }
                }
            }
            fileStream<<line;
        }

        fileStream.close();
}



void performPCA(const std::string fitFilename, bool skipLabelFit , bool skipHeadingsFit, const std::string transformFilename, \
                    bool skipLabelTransform , bool skipHeadingsTransform, const std::string fitOutputFilename, const std::string transformOutputFilename ){

    std::vector<int> fitLabels;
    std::vector<int> transformLabels;
    char seperator = ',';
    std::cout<<"PCA Reading Data...\n";
    Array2D<double> fitData = readData(fitFilename,&fitLabels,skipLabelFit,skipHeadingsFit,seperator);
    Array2D<double> transformData = readData(transformFilename,&transformLabels,skipLabelTransform,skipHeadingsTransform,seperator);
    std::cout<<"Done...\n";


    std::cout<<"PCA Fitting PCA...\n";
    PCA P;
    P.fit(fitData);
    std::cout<<"Done.\n";

    std::cout<<"PCA Transforming Data...\n";
    Array2D<double> fitDataTransformed = P.eigenTransform(fitData);
    Array2D<double> transformDataTransformed = P.eigenTransform(transformData);
    std::cout<<"Done.\n";

    std::string fitHeading="";
    std::string transformHeading="";

    if (skipHeadingsFit){

        size_t columns = fitDataTransformed.dim2();

        fitHeading += "id";
        for (size_t j=0;j<columns;j++){
            fitHeading += ", coordinate" ;
            fitHeading += std::to_string(j+1);
        }
    }

    if (skipHeadingsTransform){

        size_t columns = transformDataTransformed.dim2();

        transformHeading += "id";
        for (size_t j=0;j<columns;j++){
            transformHeading += ", coordinate" ;
            transformHeading += std::to_string(j+1);
        }
    }

    std::cout<<"PCA Writing down files...\n";
    writeDownFile(fitOutputFilename,fitDataTransformed,skipLabelFit,skipHeadingsFit,fitLabels,fitHeading);
    writeDownFile(transformOutputFilename,transformDataTransformed,skipLabelTransform,skipHeadingsTransform,transformLabels,transformHeading);
    std::cout<<"Done.\n";

}
