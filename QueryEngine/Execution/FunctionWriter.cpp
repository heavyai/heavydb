#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <sstream>

using namespace std;

enum mapd_meta_type_t {TEXT_MT, NUM_MT, INT_MT, BOOL_MT, ERROR_MT};
enum mapd_type_t {TEXT_TYPE,DOUBLE_TYPE,FLOAT_TYPE,ULONG_TYPE,UINT_TYPE,INT_TYPE,BOOL_TYPE};  

std::istream& operator >> (std::istream& i, mapd_meta_type_t &meta_type) {
    meta_type = TEXT_MT;
    std::string value;
    if (i >> value) {
        if (value == "NUM_MT") {
            meta_type = NUM_MT;
        }
        else if (value == "INT_MT") {
            meta_type = INT_MT;
        }
        else if (value == "BOOL_MT") {
            meta_type = BOOL_MT;
        }
    }
    return i;
}

mapd_meta_type_t getMetaTypeForString(const string &typeString) {
    if (typeString == "TEXT_MT")
        return TEXT_MT;
    else if (typeString == "NUM_MT")
        return NUM_MT;
    else if (typeString == "INT_MT")
        return INT_MT;
    else if (typeString == "BOOL_MT")
        return BOOL_MT;
    else
        return ERROR_MT;
}

void writeHeader(ofstream &outFile) { 
    outFile << "int execNode (const mapd_op_type opType, const vector <void *> &data, const vector<mapd_type_t> &types) {\n";
    outFile << "\tswitch (opType) {\n";
}

void writeFooter(ofstream &outFile) {
    outFile << "\t}\n";
    outFile << "}\n";
}

void splitTypes (string &sourceString, vector <mapd_meta_type_t> &metaTypes) {
    std::istringstream ss(sourceString);
    //mapd_meta_type_t tempType;
    string stringMetaType;
    while (std::getline(ss, stringMetaType, ',')) { 
        metaTypes.push_back(getMetaTypeForString(stringMetaType));
        cout << stringMetaType << " " << metaTypes.back() << endl;
    }
}

int writeSwitch(const string &inFileName, const string &outFileName) {
    ifstream inFile(inFileName.c_str());
    if (!inFile.is_open())
        return 1;
    ofstream outFile(outFileName.c_str());
    if (!outFile.is_open())
        return 1;
    writeHeader(outFile);

    string line;
    while (getline (inFile,line)) {
        if (line.size() > 1) { // to not read in empty lines
            istringstream lineStream(line);
            string opType;
            string functionName;
            lineStream >> opType >> functionName;
            outFile << "\t\tcase(" << opType << ") {\n";
            outFile << "\t\t\tbreak;\n";
            outFile << "\t\t}\n";
            int numOps;
            lineStream >> numOps;
            vector <vector <mapd_meta_type_t> > metaTypesVec;
            for (int op = 0; op < numOperands; ++op) {
                string opMetaTypeString;
                lineStream >> opMetaTypeString;
                cout << opMetaTypeString << endl;
                vector <mapd_meta_type_t> metaTypes;
                splitTypes(opMetaTypeString,metaTypes);
                metaTypesVec.push_back(metaTypes);
            }
            for (int op = 0; op < numOperands; ++op) {

function <int *, float *> (void *, void *);
eqc <typename T, typename U> (T var1, U var2)
    if (var1 == var2)
eqc (string &var1, string &var2) {
    strcmp


    function (static_cast <int *> (data[0]), static_cast <float *> (data[1]));

            }



        }
    }
    writeFooter(outFile);
    inFile.close();
    outFile.close();
    return 0;
}


int main () {
    string inFileName("op_types.txt");
    string outFileName("ops.cpp");
    writeSwitch(inFileName,outFileName);
}
