#ifndef PCA_H_INCLUDED
#define PCA_H_INCLUDED

#include <string>

void performPCA(const std::string inputBasis, bool skipFirstRowBasis , bool skipFirstColumnBasis, const std::string input2, \
                    bool skipFirstRowInput2 , bool skipFirstColumnInput2, const std::string outputFileBasis,const std::string outputFile2 );


#endif // PCA_H_INCLUDED
