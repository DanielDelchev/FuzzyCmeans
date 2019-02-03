#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <math.h>
#include <cstring>
#include <chrono>
#include <random>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/types.h>
#include <limits.h>

#include "Model.h"
#include "ICV.h"
#include "PCA.h"

std::string metricToStr(int metric)
{
    switch (metric){
        case ICV::PCmax: return "PC"; break;
        case ICV::PEmin: return "PE"; break;
        case ICV::MPCmax: return "MPC"; break;
        case ICV::FSImin: return "FSI"; break;
        case ICV::CHImax: return "CHI"; break;
        case ICV::XBImin: return "XBI"; break;
        case ICV::SCImax: return "SCI"; break;
        case ICV::KImin: return "KI"; break;
        default: break;
    }

    return "unknown_metric";
}


const double machineZeroEps = pow(10,-7);

//forward declaration
static double euclidianDistance(const std::vector<double>& instance1, const std::vector<double>& instance2);
static void standartize(std::vector<double>& sample,const std::vector<double>& m, const std::vector<double>& sd );
static void unstandartize(std::vector<double>& sample,const std::vector<double>& m, const std::vector<double>& sd );
static bool compareEqual(const double& one, const double& two);
static bool compareLess(const double& one, const double& two);

Model::Model(const std::string& filename, std::string dir){

    inputFilename = dir+"/"+filename;

    std::cout<<inputFilename<<std::endl;

    //try to open filestream
    std::fstream fileStream(inputFilename,std::ios::in);
    if (!fileStream.is_open()){
        perror("Model Could not open file!\n");
        exit(1);
    }

    //get file header (labels)
    getline(fileStream,header);

    std::string line;

    while (getline(fileStream,line))
    {
        std::vector<double> instance;
        //split line by " " , "," "tab"
        char* chunk = strtok(const_cast<char*>(line.c_str())," \t,");
        //get Features
        while ( (chunk = strtok(nullptr," \t,")) != nullptr){
            instance.push_back(atof(chunk));
        }
        //store them
        instances.push_back(instance);
    }
    //close filestream

    // set dimentions to the length of the first feature-vector
    P = instances[0].size();
    N = instances.size();
    if (N==0 || P==0){
        std::cerr<<"Model Features count or Instances count amounts to zero! Check input file! \n";
        exit(1);
    }
    //close file stream
    fileStream.close();

    Umatrix.reserve(N);
    UTmatrix.reserve(N);
    Smatrix.reserve(N);
    for (size_t i=0;i<N;i++){
        Umatrix.push_back(std::vector<double>(N,0));
        UTmatrix.push_back(std::vector<double>(N,0));
        Smatrix.push_back(std::vector<double>(N,0));
    }

    // get the means of the features
    means = std::vector<double>(P,0);
    for (size_t col=0;col<P;col++){
        for (size_t row=0;row<N;row++){
            means[col] += instances[row][col];
        }
    }
    for (double& x : means){
        x /= N;
    }

    // get the standart deviation of the features

    standartDeviations = std::vector<double>(P,0);
    for (size_t col=0;col<P;col++){
        for (size_t row=0;row<N;row++){
            standartDeviations [col] += pow((instances[row][col]-means[col]),2);
        }
    }
    for (double& x : standartDeviations){
        x = sqrt(x/(N-1));
    }

    // standartize the instances
    standartizedInstances = instances;
    for (std::vector<double>& i : standartizedInstances){
        standartize(i,means,standartDeviations);
    }
}

static void standartize(std::vector<double>& sample,const std::vector<double>& m, const std::vector<double>& sd ){
    size_t length = sample.size();
    for (size_t i = 0; i<length;i++){
        sample[i] = (sample[i]-m[i])/sd[i];
    }
}

static void unstandartize(std::vector<double>& sample,const std::vector<double>& m, const std::vector<double>& sd ){
    size_t length = sample.size();
    for (size_t i = 0; i<length;i++){
        sample[i] = (sample[i]*sd[i])+m[i];
    }
}

static double euclidianDistance (const std::vector<double>& instance1, const std::vector<double>& instance2){
     if (instance1.size() != instance2.size()){
        std::cerr<<"Model Vectors with different dimentions passed to euclidianDistance!\n";
        exit(1);
     }

     double sum = 0;

     for (size_t i = 0; i < instance1.size() ;i++){
        sum += pow( (instance2[i] - instance1[i]) ,2);
     }

     return sqrt(sum);
}


void Model::initRandom(size_t K){

    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator (seed);
    std::uniform_int_distribution<size_t> toss(0,N-1);
    std::uniform_real_distribution<double> chance(0,1);

    normalCentroids.clear();
    standartizedCentroids.clear();

    normalCentroids.reserve(K);
    standartizedCentroids.reserve(K);

    for (size_t i=0;i<K;i++){
        size_t index = toss(generator);
        normalCentroids.push_back(instances[index]);
        standartizedCentroids.push_back(standartizedInstances[index]);
    }
}

void Model::initCentroids(size_t K){
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator (seed);
    std::uniform_int_distribution<size_t> toss(0,N-1);
    std::uniform_real_distribution<double> chance(0,1);
    normalCentroids.clear();
    standartizedCentroids.clear();

    normalCentroids.reserve(K);
    standartizedCentroids.reserve(K);

    if (K > 1){
        size_t index = toss(generator);
        normalCentroids.push_back(instances[index]);
        standartizedCentroids.push_back(standartizedInstances[index]);
    }

    else{
        std::vector<double> medicentre;
        medicentre.reserve(P);
        double compSum = 0;
        for (size_t compIndex = 0;compIndex<P;compIndex++){
            for (size_t sampleIndex=0;sampleIndex<N;sampleIndex++){
                compSum += standartizedInstances[sampleIndex][compIndex];
            }
            compSum /= N;
            medicentre.push_back(compSum);
        }
        standartizedCentroids.push_back(medicentre);
        normalCentroids.push_back(medicentre);
        unstandartize(normalCentroids[0],means,standartDeviations);
    }

    std::vector<double> probabilities;

    // for K-1 more points
    for (size_t i=1;i<K;i++){
        probabilities.clear();
        probabilities.reserve(N);

        // get the distance from closest centroid to any of the N instances
        long double sum = 0;
        for (size_t inst = 0;inst<N;inst++){

            double minDistanceToCentroid = euclidianDistance(standartizedInstances[inst],standartizedCentroids[0]);
            for (size_t centIndex=0; centIndex<i; centIndex++){
                double alternative = euclidianDistance(standartizedInstances[inst],standartizedCentroids[centIndex]);
                if (compareLess(alternative,minDistanceToCentroid)){
                    minDistanceToCentroid = alternative;
                }
            }

            probabilities.push_back(pow(minDistanceToCentroid,2));
            sum += probabilities[inst];
        }
        for (double& x : probabilities){
            if (!compareEqual(sum,0)){
                x /= sum;
            }
        }

        std::vector<double> cumulativeProbabilities;

        cumulativeProbabilities.push_back(probabilities[0]);
        sum = cumulativeProbabilities[0];
        for (size_t j=1;j<N;j++){
            cumulativeProbabilities.push_back( sum+probabilities[j] );
            sum = cumulativeProbabilities[j];
        }

        double value = chance(generator);
        bool found = false;
        for (size_t j=0; (j<N) && (!found) ;j++){
            if ( compareLess(value,cumulativeProbabilities[j])){
                found = true;
                normalCentroids.push_back(instances[j]);
                standartizedCentroids.push_back(standartizedInstances[j]);
            }
        }

        // ( chance is in [0.1) )
        for (size_t j=0;j<N && (!found); j++){

            if (compareEqual(1,probabilities[j])){
                normalCentroids.push_back(instances[j]);
                standartizedCentroids.push_back(standartizedInstances[j]);
                found = true;
            }

            if (!found){
                size_t index = toss(generator);
                normalCentroids.push_back(instances[index]);
                standartizedCentroids.push_back(standartizedInstances[index]);
                found = true;
            }

        }
    }
}


static bool compareEqual(const double& one, const double& two){
    return (fabs(one-two) < machineZeroEps);
}

static bool compareLess(const double& one, const double& two){
    return ((two-one) > machineZeroEps);
}

double long Model::computeJ(double m){
    size_t K = standartizedCentroids.size();
    double long result = 0;
    for (size_t sampleIndex=0;sampleIndex<N;sampleIndex++){
        for (size_t clusterIndex=0;clusterIndex<K;clusterIndex++){
            result +=  pow(Umatrix[clusterIndex][sampleIndex],m)* \
                        pow((euclidianDistance(standartizedInstances[sampleIndex],standartizedCentroids[clusterIndex])),2);
        }
    }

    return result;
}

void Model::FKM (size_t K, double m, double epsCriteria, size_t maxIterations,bool verbose){


    if (K > N){
        std::cout<<"Only "<<N<<" samples, while "<<K<<" classes expected!\n";
        return;
    }

    std::cout<<"FKM...\n";

    initCentroids(K);
    //initRandom(K);

    size_t iterationsCounter = 0;
    bool stopCriteria = false;

    double difference = 0;

    while(!stopCriteria){
        if (verbose){
            std::cout<<"FKM iterations "<<iterationsCounter<<"\n";
        }
        difference = updateUmatrix(m);
        if (verbose){
            std::cout<<"max mu difference = "<<difference<<std::endl;
        }
        updateCentroids(m);
        if (verbose){
            std::cout<<"FKM Optimization value = "<<computeJ(m)<<"\n";
        }

        stopCriteria = ((iterationsCounter == maxIterations) || \
                        ( difference < epsCriteria));
        iterationsCounter++;
        }
    if ((iterationsCounter == maxIterations) && (difference > epsCriteria)){
        std::cout<<"FKM no convergance after "<<maxIterations<<" iterations!\n";
    }
    std::cout<<"FKM done...\n";
}

double Model::updateUmatrix(double m){
    //std::cout<<"updateUmatrix...\n";
    size_t K = standartizedCentroids.size();

    double biggestDelta = 0;

    std::vector<double> standartizedDistancesToCentroids;
    std::vector<bool> zeroDivisors;

    for (size_t sampleIndex=0;sampleIndex<N;sampleIndex++){
        standartizedDistancesToCentroids.clear();
        standartizedDistancesToCentroids.reserve(K);

        zeroDivisors.clear();
        zeroDivisors.reserve(K);
        size_t zeroDivisorsCounter = 0;

        for (size_t centroidIndex=0; centroidIndex<K; centroidIndex++){
            standartizedDistancesToCentroids.push_back(euclidianDistance(standartizedCentroids[centroidIndex],standartizedInstances[sampleIndex]));
            bool isZero = compareEqual(standartizedDistancesToCentroids[centroidIndex],0);
            zeroDivisors.push_back(isZero);
            if (zeroDivisors[centroidIndex]){
                zeroDivisorsCounter++;
            }
        }

        for (size_t centroidIndex=0; centroidIndex<K; centroidIndex++){
            if (zeroDivisorsCounter>0){
                if (!zeroDivisors[centroidIndex]){
                    Umatrix[centroidIndex][sampleIndex] = 0;
                }
                else{
                    double newValue = ((static_cast<double> (1))/zeroDivisorsCounter); // O my GODDDDDDDDDDDDDD
                    biggestDelta = std::max(biggestDelta, fabs(Umatrix[centroidIndex][sampleIndex] - newValue));
                    Umatrix[centroidIndex][sampleIndex] = newValue;
                }
            }
            else{
                double sum = 0;

                for (size_t j=0;j<K;j++){
                    sum += pow(  (standartizedDistancesToCentroids[centroidIndex] / standartizedDistancesToCentroids[j])  ,((static_cast<double> (2))/(m-1)) );
                }
                double newValue = ((static_cast<double> (1))/sum);
                biggestDelta = std::max(biggestDelta, fabs(Umatrix[centroidIndex][sampleIndex] - newValue));
                Umatrix[centroidIndex][sampleIndex] = newValue;
            }
        }
    }
    //std::cout<<"updateUmatrix done...\n";

    debugU(); // remove in prod

    return biggestDelta;
}

void Model::updateCentroids(double m){

    //std::cout<<"updateCentorids...\n";

    size_t K = standartizedCentroids.size();

    normalCentroids.clear();
    standartizedCentroids.clear();
    normalCentroids.reserve(K);
    standartizedCentroids.reserve(K);


    //for each centroid
    for (size_t centroidIndex=0;centroidIndex<K;centroidIndex++){
      std::vector<double> newStandartizedCentroid;
      newStandartizedCentroid.reserve(P);

      //for each component of the new centroid
      for (size_t component=0;component<P;component++){
            double numerator = 0;
            double denominator = 0;

            //for each sample
            for (size_t sampleIndex=0;sampleIndex<N;sampleIndex++){
                double mu = pow(Umatrix[centroidIndex][sampleIndex],m);
                denominator += mu;
                numerator += mu * standartizedInstances[sampleIndex][component];
            }
                        //
            newStandartizedCentroid.push_back((numerator/denominator));
      }

      standartizedCentroids.push_back(newStandartizedCentroid);
      normalCentroids.push_back(standartizedCentroids[centroidIndex]);
      unstandartize(normalCentroids[centroidIndex],means,standartDeviations);
    }

    //std::cout<<"updateCentorids done...\n\n";

}


bool  Model::debugU(){
    bool OK = true;

    size_t K = standartizedCentroids.size();

    double totalRowSum = 0;

    for (size_t row=0;row<K;row++){
        double rowSum = 0;
        for (size_t col=0;col<N;col++){
            rowSum += Umatrix[row][col];
        }
        totalRowSum += rowSum;
        if ((rowSum > N) || (rowSum <=0)){
            std::cout<<"rowSum "<<row<<" = "<<rowSum<<std::endl;
            std::cerr << "rowSum is wrong!\n";
            OK = false;
        }
    }

    if (!compareEqual(totalRowSum,N)){
        std::cerr << "totalRowSum is wrong!\n";
        std::cout<<"total row sum = "<<totalRowSum<<std::endl;
        OK = false;
    }

    for (size_t col=0;col<N;col++){
        double colSum = 0;
        for (size_t row=0;row<K;row++){
            colSum += Umatrix[row][col];
        }
        if (!compareEqual(colSum,1)){
            std::cerr << "colSum is wrong!\n";
            std::cout<<"colSum "<<col<<" = "<<colSum<<std::endl;
            OK = false;
        }
    }
    return OK;
}

void Model::printCentroids(){
    size_t K = standartizedCentroids.size();

    for (size_t i=0;i<K;i++){
        for (size_t j=0;j<P;j++){
            std::cout<<standartizedCentroids[i][j]<<" ,";
        }
        std::cout<<std::endl;

    }
}

void Model::writeDownFinalResults(std::string prefix, std::string dir, bool toXLSX){

    std::cout<<"---------------->writingDownFinalResults... \n";

    #if defined(__CYGWIN__) || defined(UNIX) || defined(__unix__) || defined(LINUX) || defined (__linux__)
    standartizedInstancesFilename = dir+"/"+prefix+"_standartizedInstancesFile.csv";
    standartizedCentroidsFilename = dir+"/"+prefix+"_standartizedCentroidsFile.csv";
    normalCentroidsFilename = dir+"/"+prefix+"_normalCentroidsFile.csv";
    characteristicValuesFilename = dir+"/"+prefix+"_characteristicValues.csv";
    hardClustersFilename = dir+"/"+prefix+"_hardClusters.csv";
    similarityRelationFilename = dir+"/"+prefix+"_similarityRelation.csv";
    similarityRelationFilenameXLSX = dir+"/"+prefix+"_similarityRelation.xlsx";
    fuzzynessMeasureFilename = dir+"/"+prefix+"_fuzzynessMeasure.txt"; // also K and m here
    PCA_basisFilename = dir+"/"+"PCA"+prefix+"_standartizedInputFile.csv";
    PCA_transformFilename = dir+"/"+"PCA"+prefix+"_standartizedCentroidsFile.csv";
    #else
    standartizedInstancesFilename = dir+"\\"+prefix+"_standartizedInstancesFile.csv";
    standartizedCentroidsFilename = dir+"\\"+prefix+"_standartizedCentroidsFile.csv";
    normalCentroidsFilename = dir+"\\"+prefix+"_normalCentroidsFile.csv";
    characteristicValuesFilename = dir+"\\"+prefix+"_characteristicValues.csv";
    hardClustersFilename = dir+"\\"+prefix+"_hardClusters.csv";
    similarityRelationFilename = dir+"\\"+prefix+"_similarityRelation.csv";
    similarityRelationFilenameXLSX = dir+"/"+prefix+"_similarityRelation.xlsx"
    fuzzynessMeasureFilename = dir+"\\"+prefix+"_fuzzynessMeasure.txt"; // also K and m here
    PCA_basisFilename = dir+"\\"+"PCA"+prefix+"_standartizedInputFile.csv";
    PCA_transformFilename = dir+"\\"+"PCA"+prefix+"_standartizedCentroidsFile.csv";
    #endif

    size_t K = standartizedCentroids.size();

    //characteristics values file
    std::ofstream fileStreamCVF(characteristicValuesFilename,std::ios::out|std::ios::trunc);

        if (!fileStreamCVF.is_open()){
            perror("Could not open/create file!\n");
            exit(1);
        }

        std::string heading = "id";
        for (size_t muIndex = 1; muIndex<=K; muIndex++){
            heading += ", mu";
            heading += std::to_string(muIndex);
        }
        fileStreamCVF<<heading<<"\n";

        for (size_t instanceIndex=0; instanceIndex<N;instanceIndex++){
            size_t writeIndex = instanceIndex+1;
            fileStreamCVF<<std::to_string(writeIndex);

            for (size_t clusterIndex=0;clusterIndex<K;clusterIndex++){
                fileStreamCVF<<", "<<std::to_string(Umatrix[clusterIndex][instanceIndex]);
            }
            fileStreamCVF<<"\n";
        }
        fileStreamCVF.close();



        //Hard Clusters
        std::ofstream fileStreamHCF(hardClustersFilename,std::ios::out|std::ios::trunc);

        if (!fileStreamHCF.is_open()){
            perror("Could not open/create file!\n");
            exit(1);
        }

        heading = "id, Cluster id, Closest centroid id";
        fileStreamHCF<<heading<<"\n";

        for (size_t instanceIndex=0; instanceIndex<N;instanceIndex++){
            size_t writeIndex = instanceIndex+1;
            fileStreamHCF<<std::to_string(writeIndex)<<", ";

            double maxVal = 0;
            size_t bestIndex = 0;
            for (size_t clusterIndex=0;clusterIndex<K;clusterIndex++){
                if( Umatrix[clusterIndex][instanceIndex] > maxVal){
                    maxVal = Umatrix[clusterIndex][instanceIndex];
                    bestIndex = clusterIndex;
                }
            }
            fileStreamHCF<<(bestIndex+1)<<", ";


            ///
                double minDist = euclidianDistance(standartizedInstances[instanceIndex],standartizedCentroids[0]);
                bestIndex = 0;
                for (size_t clusterIndex=0;clusterIndex<K;clusterIndex++){
                    double alternative = euclidianDistance(standartizedInstances[instanceIndex],standartizedCentroids[clusterIndex]);
                    if( alternative < minDist ){
                        minDist = alternative;
                        bestIndex = clusterIndex;
                    }
                }
                fileStreamHCF<<(bestIndex+1);

            ///


            fileStreamHCF<<"\n";
        }
        fileStreamHCF.close();



        //standartized centroids file
        std::ofstream fileStreamSCF(standartizedCentroidsFilename,std::ios::out|std::ios::trunc);

        fileStreamSCF <<header<<"\n";
        for (size_t centroidIndex=0; centroidIndex<K;centroidIndex++){
            size_t writeIndex = centroidIndex+1;
            fileStreamSCF<<std::to_string(writeIndex);

            for (size_t componentIndex=0;componentIndex<P;componentIndex++){

                fileStreamSCF<<", "<<std::to_string(standartizedCentroids[centroidIndex][componentIndex]);
            }
            fileStreamSCF<<"\n";
        }

        fileStreamSCF.close();


        //normal centroids file
        std::ofstream fileStreamNCF(normalCentroidsFilename,std::ios::out|std::ios::trunc);

        fileStreamNCF <<header<<"\n";
        for (size_t centroidIndex=0; centroidIndex<K;centroidIndex++){
            size_t writeIndex = centroidIndex+1;
            fileStreamNCF<<std::to_string(writeIndex);

            for (size_t componentIndex=0;componentIndex<P;componentIndex++){
                fileStreamNCF<<", "<<std::to_string(normalCentroids[centroidIndex][componentIndex]);
            }
            fileStreamNCF<<"\n";
        }

        fileStreamNCF.close();


        //standartized instances file
        std::ofstream fileStreamSIF(standartizedInstancesFilename,std::ios::out|std::ios::trunc);

        fileStreamSIF <<header<<"\n";
        for (size_t instanceIndex=0; instanceIndex<N;instanceIndex++){
            size_t writeIndex = instanceIndex+1;
            fileStreamSIF<<std::to_string(writeIndex);

            for (size_t componentIndex=0;componentIndex<P;componentIndex++){
                fileStreamSIF<<", "<<standartizedInstances[instanceIndex][componentIndex];
            }
            fileStreamSIF<<"\n";
        }

        fileStreamSIF.close();


        performPCA( standartizedInstancesFilename, true , true, standartizedCentroidsFilename, true, true , PCA_basisFilename, PCA_transformFilename );


        computeUTmatrix();
        double FPC = computeFuzzynessMetric();
        std::ofstream fileStreamFM(fuzzynessMeasureFilename,std::ios::out|std::ios::trunc);
        fileStreamFM<<"Fuzzy Partition Coefficient = "<<FPC<<"\n";
        fileStreamFM<<"1\\"<<K<<" <= "<<FPC<<" <= 1\n";
        fileStreamFM<<"1 means crisp clustering, "<<"1\\"<<K<<" means complete ambiguity\n";
        fileStreamFM.close();

        int res = callPlottingScripts(prefix,dir);
        std::cout<<"Plotting finished with exit code "<<res<<std::endl;
        if (res!=0){
            std::cout<<"Issue with creating plots, perhaps run the scripts manually?"<<std::endl;
        }

        std::cout<<"writingDownSimilarityRelationMatrix...\n";
        computeSimilarityMatrix();
        std::ofstream fileStreamSRF(similarityRelationFilename,std::ios::out|std::ios::trunc);
        fileStreamSRF<<"SAMPLES";
        for (size_t sampleIndex=0;sampleIndex<N;sampleIndex++){
            fileStreamSRF<<", sample"<<(sampleIndex+1);
        }
        fileStreamSRF<<"\n";
        for (size_t sampleIndex=0;sampleIndex<N;sampleIndex++){
            fileStreamSRF<<"sample"<<(sampleIndex+1);
            for (size_t relationSampleIndex=0;relationSampleIndex<N;relationSampleIndex++){
                fileStreamSRF<<", "<<Smatrix[sampleIndex][relationSampleIndex];
            }
            fileStreamSRF<<"\n";
        }

        fileStreamSRF.close();
        std::cout<<"writingDownSimilarityRelationMatrix done...\n";
        if (toXLSX){
            res = CSVtoExcel(similarityRelationFilename,similarityRelationFilenameXLSX);
            std::cout<<"Convession finished with exit code "<<res<<std::endl;
            if (res!=0){
            std::cout<<"Issue with converting csv, perhaps run the script manually?"<<std::endl;
            }
        }


        std::cout<<"---------------->writeDownFinalResults done...\n";
}

int Model::callPlottingScripts(std::string prefix, std::string dir){
    int result = 0;

    std::cout<<"Calling plotting scripts...\n";


    #if defined(__CYGWIN__) || defined(UNIX) || defined(__unix__) || defined(LINUX) || defined (__linux__)
    PCAgraphicFilename = dir+"/"+prefix+"_PCAgraphic.png";
    LDAgraphicFilename = dir+"/"+prefix+"_LDAgraphic.png";
    #else
    PCAgraphicFilename = dir+"\\"+prefix+"_PCAgraphic.png";
    LDAgraphicFilename = dir+"\\"+prefix+"_LDAgraphic.png";
    #endif

    std::cout<<"Using python located at: \n";
    result = system("which python");

    std::string commandPCAplot = "python "+PCApath+" "+standartizedInstancesFilename+" "+standartizedCentroidsFilename+" "+hardClustersFilename+" "+PCAgraphicFilename;
    result = system(commandPCAplot.c_str());


    size_t K = standartizedCentroids.size();
    if (K <= 2){
        std::cout<<"Skippping LDAgraph for <2 classes (as output would be 1 dimentional)\n";
    }
    else{
        std::string commandLDAplot = "python "+LDApath+" "+standartizedInstancesFilename+" "+standartizedCentroidsFilename+" "+hardClustersFilename+" "+LDAgraphicFilename;
        result = system(commandLDAplot.c_str());
    }

    std::cout<<"Calling plotting scripts done...\n";

    return result;

}

void Model::computeUTmatrix(){
    size_t K = standartizedCentroids.size();
    for (size_t rowIndex=0;rowIndex<K;rowIndex++){
        for (size_t colIndex=0;colIndex<N;colIndex++){
            UTmatrix[colIndex][rowIndex] = Umatrix[rowIndex][colIndex];
        }
    }
}

double Model::computeFuzzynessMetric(){
    size_t K = standartizedCentroids.size();

    std::vector<std::vector<double>> resultM;
    resultM.reserve(K);
    for (size_t i=0;i<K;i++){
        resultM.push_back(std::vector<double>(K,0));
    }

    if(Umatrix[0].size() != UTmatrix.size()){
        perror("FKM computeFuzzyMetric Invalid matrices passed for multiplication!\n");
        exit(1);
    }

    // left U = M KxN
    // right UT = M NxK

    for (size_t row=0; row<K; row++) {
        for (size_t column=0; column<K; column++) {
            double sum = 0;
            for (size_t index=0; index<N; index++) {
                sum += Umatrix[row][index] * UTmatrix[index][column];
            }
            resultM [row][column] = sum;
        }
    }

    double result=0;

    for (size_t index=0;index<K;index++){
        result += resultM[index][index];
    }

    result /= N;

    return result;
}


void Model::computeSimilarityMatrix(){
    size_t K = standartizedCentroids.size();

    if(Umatrix[0].size() != UTmatrix.size()){
        perror("FKM computeFuzzyMetric Invalid matrices passed for multiplication!\n");
        exit(1);
    }

    // left UT = M NxK
    // right U = M KxN

    for (size_t row=0; row<N; row++) {
        for (size_t column=0; column<N; column++) {
            double sum = 0;
            for (size_t index=0; index<K; index++) {
                sum += std::min(UTmatrix[row][index],Umatrix[index][column]);
            }
            Smatrix [row][column] = sum;
        }
    }
}

int Model::CSVtoExcel(const std::string& in, const std::string& out)const{

    std::cout<<"converting CSV ...\n";
    bool result = 0;

    std::cout<<"Using python located at: \n";
    result = system("which python");

    std::string convert = "python "+CSVtoExcelPath+" "+ in+" "+out;
    result = system(convert.c_str());

    std::cout<<"converting CSV done...\n";

    return result;
}


size_t Model::autoFKM(size_t metric,double m,double epsCriteria,size_t maxIterations,size_t span, size_t fold,std::string prefix, std::string dir){

    if (span > N){
        std::cout<<"Only "<<N<<" samples, while "<<span<<" classes expected!\n";
        return 1;
    }

    if ((span < 1) || (fold < 1)){
        std::cout<<"auto FKM invalid parameters!\n";
        return 0;
    }

    std::vector<double> values;
    values.reserve(span);
    double best = 0;
    size_t bestK = 2;

    for (size_t k = 2;k<=span;k++){
        std::cout<<"----------->Running autoFKM for k="<<k<<" \n";
        std::vector<double> avg;
        avg.reserve(fold);
        for (size_t runId=0;runId<fold;runId++){
            std::cout<<"fold "<<(runId+1)<<" \n";
            FKM(k,m,epsCriteria,maxIterations,false);
            switch (metric){
                case ICV::PCmax: avg.push_back(PC()); break;
                case ICV::PEmin: avg.push_back(PE()); break;
                case ICV::MPCmax: avg.push_back(MPC()); break;
                case ICV::CHImax: avg.push_back(CHI()); break;
                case ICV::FSImin: avg.push_back(FSI(m)); break;
                case ICV::XBImin: avg.push_back(XBI()); break;
                case ICV::KImin: avg.push_back(KI()); break;
                case ICV::SCImax: avg.push_back(SCI(m)); break;
                default: break;
            }
        }
        double average = 0;
        for (const double& val : avg){
            average += val;
        }
        average /= fold;

        values.push_back(average);
        if (k == 2){
            best = average;
        }
        else{
            switch (metric){
                case ICV::PCmax:
                case ICV::MPCmax:
                case ICV::CHImax:
                case ICV::SCImax:
                    if (average > best){
                        best = average;
                        bestK = k;
                    }
                    break;
                case ICV::PEmin:
                case ICV::FSImin:
                case ICV::XBImin:
                case ICV::KImin:
                    if (average < best){
                        best = average;
                        bestK = k;
                    }
                    break;
                default: break;
            }
        }
    }


    // write down in file
    // do graphics

    #if defined(__CYGWIN__) || defined(UNIX) || defined(__unix__) || defined(LINUX) || defined (__linux__)
    unsupervisedResultsFilename = dir+"/"+prefix+"_"+metricToStr(metric)+"_unsupervisedResultsFile.csv";
    unsupervisedGraphicsFilename = dir+"/"+prefix+"_"+metricToStr(metric)+"_unsupervisedResultsFile.png";
    #else
    unsupervisedResultsFilename = dir+"/"+prefix+"_"+metricToStr(metric)+"_unsupervisedResultsFile.csv";
    unsupervisedGraphicsFilename = dir+"/"+prefix+"_"+metricToStr(metric)+"_unsupervisedResultsFile.png";
    #endif


    std::cout<<"auto FKM writing down results from the run...\n";
    std::ofstream fileStreamResults(unsupervisedResultsFilename,std::ios::out|std::ios::trunc);

        if (!fileStreamResults.is_open()){
            perror("Could not open/create file!\n");
            exit(1);
        }

        std::string heading = "K, "+metricToStr(metric)+"\n";
        fileStreamResults<<heading;

        for (size_t k=2; k<=span;k++){
            fileStreamResults<<k<<", "<<values[k-2]<<"\n";
        }
        fileStreamResults.close();
    std::cout<<"auto FKM writing down results from the run done...\n";

    std::cout<<"auto FKM plotting graphic ...\n";

    bool result = 0;

    std::cout<<"Using python located at: \n";
    result = system("which python");

    std::string convert = "python "+unsupervisedPlotFilename+" "+ unsupervisedResultsFilename +" "+unsupervisedGraphicsFilename+" "+std::to_string(metric);

    result = system(convert.c_str());

    std::cout<<"Plotting finished with exit code "<<result<<std::endl;
    if (result!=0){
        std::cout<<"Issue with converting creating plot for auto FKM, perhaps run the script manually?"<<std::endl;
    }


    std::cout<<"auto FKM plotting graphic done...\n";


    std::cout<<"----------->With the given parametrs autoFKM found best k="<<bestK<<" for metric "<<metricToStr(metric)<<"\n";

    return bestK;

}


double Model::PC()const{
    size_t K = standartizedCentroids.size();
    double sum = 0;
    for (size_t clusterIndex=0;clusterIndex<K;clusterIndex++){
        for (size_t sampleIndex=0;sampleIndex<N;sampleIndex++){
            sum += pow(Umatrix[clusterIndex][sampleIndex],2);
        }
    }
    return (sum/N);
}

double Model::PE()const{
    size_t K = standartizedCentroids.size();
    double sum = 0;
    for (size_t clusterIndex=0;clusterIndex<K;clusterIndex++){
        for (size_t sampleIndex=0;sampleIndex<N;sampleIndex++){
            if (!compareEqual(Umatrix[clusterIndex][sampleIndex],0)){
                sum += Umatrix[clusterIndex][sampleIndex]*std::log2(Umatrix[clusterIndex][sampleIndex]);
            }
        }
    }
    return -(sum/N);
}
double Model::MPC()const{
    size_t K = standartizedCentroids.size();
    return (1 - (K*(1-PC())/(K-1)) );
}


std::vector<double>  Model::getCentralCentroid()const{
    std::vector<double> result;
    result.reserve(P);

    for (size_t compIndex=0;compIndex<P;compIndex++){
        double sum = 0;
        for (size_t instIndex=0;instIndex<N;instIndex++){
            sum += standartizedInstances[instIndex][compIndex];
        }
        sum /= N;
        result.push_back(sum);
    }
    return result;
}

std::vector<std::vector<double>> Model::getSamplesOfClass(size_t c)const{
    std::vector<std::vector<double>> res;
    size_t K = standartizedCentroids.size();
    if (c > K){
        std::cout<<"getSamplesOfClass error!\n";
    }
    for (size_t sampleIndex=0;sampleIndex<N;sampleIndex++){
        size_t indexFound = 0;
        double muValue = Umatrix[0][sampleIndex];
        for (size_t clusterIndex=0;clusterIndex<K;clusterIndex++){
            if (Umatrix[clusterIndex][sampleIndex] > muValue){
                indexFound = clusterIndex;
                muValue = Umatrix[clusterIndex][sampleIndex];
            }
        }
        if (indexFound == c){
            res.push_back(standartizedInstances[sampleIndex]);
        }
    }

    return res;
}

size_t Model::getCardinalityOfClass(size_t c)const{
    return (getSamplesOfClass(c).size());
}


double Model::CHI()const{

    std::vector<double> CC = getCentralCentroid();

    size_t K = standartizedCentroids.size();

    double BK = 0;
    double WK = 0;

    for (size_t centroidIndex=0;centroidIndex<K;centroidIndex++){
        size_t Nk = getCardinalityOfClass(centroidIndex);
        BK += ((double)Nk)*pow(euclidianDistance(standartizedCentroids[centroidIndex],CC),2);
    }

    for (size_t centroidIndex=0;centroidIndex<K;centroidIndex++){
        std::vector<std::vector<double>> clusterSamples = getSamplesOfClass(centroidIndex);
        for (const auto& cs : clusterSamples){
            WK += pow(euclidianDistance(cs,standartizedCentroids[centroidIndex]),2);
        }
    }

    if (N==K){
        std::cout<<"CHI metric not defined for N=K!\n";
        return 0;
    }

    return (BK/(K-1))/(WK/(N-K));

}

size_t Model::getN()const{
    return N;
}

double Model::FSI(double m)const{
    double sum1 = 0;
    double sum2 = 0;

    size_t K = standartizedCentroids.size();
    std::vector<double> CC = getCentralCentroid();

    for (size_t centroidIndex=0;centroidIndex<K;centroidIndex++){
        for (size_t sampleIndex=0;sampleIndex<K;sampleIndex++){
            sum1 += pow(Umatrix[centroidIndex][sampleIndex],m)* \
            pow(euclidianDistance(standartizedCentroids[centroidIndex],standartizedInstances[sampleIndex]),2);

            sum2 -= pow(Umatrix[centroidIndex][sampleIndex],m)* \
            pow(euclidianDistance( CC ,standartizedInstances[sampleIndex]),2);
        }
    }

    return (sum1-sum2);
}
double Model::XBI()const{
    double sum = 0;

    size_t K = standartizedCentroids.size();
    std::vector<double> CC = getCentralCentroid();

    for (size_t centroidIndex=0;centroidIndex<K;centroidIndex++){
        for (size_t sampleIndex=0;sampleIndex<K;sampleIndex++){
            sum += pow(Umatrix[centroidIndex][sampleIndex],2)* \
            pow(euclidianDistance(standartizedCentroids[centroidIndex],standartizedInstances[sampleIndex]),2);
        }
    }

    double minDist = euclidianDistance(standartizedCentroids[0],standartizedCentroids[1]);

    for (size_t i=0;i<K-1;i++){
        for (size_t j=i+1;j<K;j++){
            minDist = std::min(minDist, euclidianDistance(standartizedCentroids[i],standartizedCentroids[j]));
        }
    }

    if (compareEqual(minDist,0)){
        return INT_MAX;
    }
    return (sum / (N*minDist)) ;



}
double Model::KI()const{
    size_t K = standartizedCentroids.size();

    double sum1=0;

    for (size_t centroidIndex=0;centroidIndex<K;centroidIndex++){
        for (size_t sampleIndex=0;sampleIndex<K;sampleIndex++){
            sum1 += pow(Umatrix[centroidIndex][sampleIndex],2)* \
            pow(euclidianDistance(standartizedCentroids[centroidIndex],standartizedInstances[sampleIndex]),2);
        }
    }

    double minDist = euclidianDistance(standartizedCentroids[0],standartizedCentroids[1]);

    for (size_t i=0;i<K-1;i++){
        for (size_t j=i+1;j<K;j++){
            minDist = std::min(minDist, pow(euclidianDistance(standartizedCentroids[i],standartizedCentroids[j]),2));
        }
    }

    double sum2=0;

    std::vector<double> CC = getCentralCentroid();

    for (size_t i=0;i<K;i++){
        sum2 += pow(euclidianDistance(standartizedCentroids[i],CC),2);
    }
    sum2 /= K;

    if (compareEqual(minDist,0)){
        return INT_MAX;
    }

    return ((sum1+sum2)/minDist);

}
double Model::SCI(double m)const{
    size_t K = standartizedCentroids.size();
    double SC1 = 0;

    double sum1=0;

    std::vector<double> CC = getCentralCentroid();

    for (size_t i=0;i<K;i++){
        sum1 += euclidianDistance(standartizedCentroids[i],CC);
    }
    sum1 /= K;

    double sum2 = 0;
        for (size_t i=0;i<K;i++){
            double numerator=0;
            double denominator=0;
            for (size_t j=0;j<N;j++){
                numerator += pow(euclidianDistance(standartizedInstances[j],standartizedCentroids[i]),2)*pow(Umatrix[i][j],m);
                denominator += Umatrix[i][j];
            }
            sum2 += (numerator / denominator);
    }

    if (compareEqual(sum2,0)){
        SC1 = INT_MIN/2;
    }
    else{
        SC1 = (sum1/sum2);
    }

    double SC2 = 0;

    sum1 = 0;
    sum2 = 0;

    for (size_t i=0;i<K-1;i++){
        for (size_t k=i+1;k<K;k++){
            double numerator = 0;
            double denominator = 0;
            for (size_t j=0;j<N;j++){
                numerator += pow(std::min(Umatrix[i][j],Umatrix[k][j]),2);
                denominator += std::min(Umatrix[i][j],Umatrix[k][j]);
            }

            if (compareEqual(denominator,0)){
                return (INT_MIN/2);
            }

            sum1 += (numerator/denominator);
        }
    }


    double numerator = 0;
    double denominator = 0;
    for (size_t j=0;j<N;j++){
        double bestTop = pow(Umatrix[0][j],2);
        double bestBottom = Umatrix[0][j];
        for (size_t i=0;i<K;i++){
            bestTop = std::max(pow(Umatrix[i][j],2),bestTop);
            bestBottom = std::max(Umatrix[i][j],bestTop);
        }
        numerator += bestTop;
        denominator +=bestBottom;
    }


    sum2 = (numerator / denominator);


    if (compareEqual(denominator,0) || compareEqual(sum2,0)){
        return (INT_MIN/2);
    }

    SC2 = (sum1/sum2);


    return (SC1 - SC2);


}
