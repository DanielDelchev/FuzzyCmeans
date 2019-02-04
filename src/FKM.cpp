#include <iostream>

#include "Model.h"


void printUsage(){
    std::cout<<"Usage: FKM <directory> <inputFilename> <prefix for filenamse> <m value> <epsilon singificance difference> <max Itererations> <convert> <K> <metric>\n";
    std::cout<<"<directory> (string)- directory in which input file is located and output files will be generated\n";
    std::cout<<"<inputFilename.csv> (string)- name of the input .csv file (in the dir folder)\n";
    std::cout<<"<prefix> (string)- string to add to all created files in the dir folder\n";
    std::cout<<"<m-value>  (double)- m parameter for fuzzy K means \n";
    std::cout<<"<epsilon>  (double)- if there is no difference bigger than epsilon in a mu value of the current and last matrix, stop\n";
    std::cout<<"<max Iteratons> (integer)- max iterations, before stopping (regardless of epsilon)\n";
    std::cout<<"<convert> (t/T/TRUE/true/f/F/FALSE/false)- whether to convert the similarity matrix to XLSX\n";
    std::cout<<"<K> (integer)- number of clusters, if K=0, then <metric> is used for unsupervised choice of best K\n";
    std::cout<<"[metric] - !supplied when K is 0 for unsupervised choice of K! (integer)- the metric to use to use for choice of K, (used only when <K> = 0) \n";
    std::cout<<"metrix is one of {1,2,3,4,5,6,7,8} 1=PC+ 2=PE- 3=MPC+ 4=FSI- 5=CHI+ 6=XBI- 7=CSI+ 8=KI- \n";

}

int main(int argc, const char* argv[])
{

    std::string inputFile;
    std::string prefix;
    std::string dir;
    double m;
    double eps;
    int maxIter;
    int K; // if K is 0 then find it with the corresponding metric
    int metric = 0;
    bool convert;

    std::string convertStr;
    //validation


    if (argc!=9 && argc!=10){
        printUsage();
        return 1;
    }

    try{
        dir = std::string(argv[1]);
        inputFile = std::string(argv[2]);
        prefix = std::string(argv[3]);
        m = std::stod(std::string(argv[4]));
        eps = std::stod(std::string(argv[5]));
        maxIter = std::stoi(std::string(argv[6]));
        convertStr = std::string(argv[7]);
        K = std::stoi(std::string(argv[8]));
    }
    catch(const std::exception& e){
        printUsage();
        return 1;
    }

    if (argc == 9){

        if (K==0){
            std::cout<<"K=0, metric required!\n\n";
            printUsage();
            return 1;
        }

    }
    else if (argc == 10){
        try{
            metric = std::stoi(std::string(argv[9]));
        }
        catch(const std::exception& e){
            printUsage();
            return 1;
        }
        if (metric < minMetricNumber || metric > maxMetricNumber){
            std::cout<<"metric should be in ["<<minMetricNumber<<", "<<maxMetricNumber<<"]\n\n";
            printUsage();
            return 1;
        }
        if (K!=0) {
            std::cout<<"K should be = 0 when a metric is supplied!\n\n";
            printUsage();
            return 1;
        }
    }

    if (K<0) {
        std::cout<<"K should be >= 0 (0 if metric is supplied)\n\n";
        printUsage();
        return 1;
    }
    if (maxIter <=0 ){
        std::cout<<"maxIterations should be > 0\n\n";
        printUsage();
        return 1;
    }
    if (eps <= 0){
        std::cout<<"eps should be > 0\n\n";
        printUsage();
        return 1;
    }
    if (m<1 ){
        std::cout<<"m should be >= 1\n\n";
        printUsage();
        return 1;
    }

    if (convertStr == std::string("t") || convertStr == std::string("T") || convertStr == std::string("true") || convertStr == std::string("TRUE")){
        convert = true;
    }
    else if (convertStr == std::string("f") || convertStr == std::string("F") || convertStr == std::string("false") || convertStr == std::string("FALSE")){
        convert = false;
    }
    else{
        std::cout<<"convert parameter not valid!\n\n";
        printUsage();
        return 1;
    }


    Model cute = Model(inputFile,dir);

    if (K!=0){
        cute.FKM(K,m,eps,maxIter,true);
    }
    else{
        if ((metric == ICV::CHImax)){
            std::cout<<"Note! CHI metric is not defined for K==N! \n\n";
        }
        size_t bestK = cute.autoFKM(metric,m,eps,maxIter,16,5,prefix,dir);
        cute.FKM(bestK,m,eps,maxIter,true);
    }
    cute.writeDownFinalResults(prefix,dir,convert);

    return 0;
}
