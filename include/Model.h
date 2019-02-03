#ifndef MODEL_H_INCLUDED
#define MODEL_H_INCLUDED

#include <vector>
#include <string>
#include <math.h>



const int minMetricNumber = 1;
const int maxMetricNumber = 8;
enum ICV:int{
    PCmax = 1,
    PEmin = 2,
    MPCmax = 3,
    FSImin = 4,
    CHImax = 5,
    XBImin = 6,
    SCImax = 7,
    KImin = 8,
};


std::string metricToStr(int metric);

class Model{

public:

    // assuming file has initial headers row and first row of IDs from 1 to N
    Model(const std::string& filename, std::string dir = "data"); //
    void FKM (size_t K, double m, double epsCriteria, size_t maxIterations, bool verbose = false); //
    size_t autoFKM(size_t metric, double m, double epsCriteria, size_t maxIterations,size_t span = 16, size_t fold = 8,std::string prefix = "myPrefix", std::string dir ="data");
    void writeDownFinalResults(std::string prefix = "myPrefix" , std::string dir="data", bool toXLSX = false);//

    size_t getN()const;

//private:
    std::vector<std::vector<double>> Umatrix;
    std::vector<std::vector<double>> UTmatrix;
    std::vector<std::vector<double>> Smatrix;

    std::string standartizedInstancesFilename;
    std::string standartizedCentroidsFilename;
    std::string normalCentroidsFilename;
    std::string characteristicValuesFilename;
    std::string hardClustersFilename;
    std::string similarityRelationFilename;
    std::string similarityRelationFilenameXLSX;
    std::string fuzzynessMeasureFilename;
    std::string PCA_basisFilename;
    std::string PCA_transformFilename;

    std::string PCAgraphicFilename;
    std::string LDAgraphicFilename;


    std::string unsupervisedResultsFilename;
    std::string unsupervisedGraphicsFilename;

    #if defined(__CYGWIN__) || defined(UNIX) || defined(__unix__) || defined(LINUX) || defined (__linux__)
        std::string PCApath="scripts/visualize.py";
        std::string LDApath="scripts/LDA.py";
        std::string CSVtoExcelPath="scripts/CSVtoExcel.py";
        std::string unsupervisedPlotFilename="scripts/unsupervisedPlot.py";
    #else
        std::string PCApath="scripts\\visualize.py";
        std::string LDApath="scripts\\LDA.py";
        std::string CSVtoExcelPath="scripts\\CSVtoExcel.py";
        std::string unsupervisedPlotFilename="scripts\\unsupervisedPlot.py";
    #endif

    int callPlottingScripts(std::string prefix = "myPrefix" ,std::string dir="data");
    void initCentroids(size_t K); //
    void initRandom(size_t K); //
    double long computeJ(double m); //

    int CSVtoExcel(const std::string& in, const std::string& out)const;

    void computeUTmatrix();
    double computeFuzzynessMetric();
    void computeSimilarityMatrix();


    void updateCentroids(double m); //
    double updateUmatrix(double m); //

    bool debugU(); //
    void printCentroids();//

    double PC()const;
    double PE()const;
    double MPC()const;
    double CHI()const;
    double FSI(double m)const;
    double XBI()const;
    double KI()const;
    double SCI(double m)const;

    std::vector<double> getCentralCentroid()const;
    std::vector<std::vector<double>> getSamplesOfClass(size_t c)const;
    size_t getCardinalityOfClass(size_t c)const;

    //variables

    std::string inputFilename;

    //the instance at position X in the vector is expected to correspond to sample with id X+1 from the input file
    std::vector<std::vector<double>> instances; //
    std::vector<std::vector<double>> standartizedInstances; //

    std::string header; //
    std::vector<double> means; //
    std::vector<double> standartDeviations; //

    std::vector<std::vector<double>> normalCentroids; //
    std::vector<std::vector<double>> standartizedCentroids; //

    size_t P; //dimentions of a sample
    size_t N; //number of samples
};

#endif // MODEL_H_INCLUDED
