#include <assert.h>

#include "OutputWriter.h"
#include "OutputBuffer.h"

using std::vector; 
using std::string; 

OutputWriter::OutputWriter(OutputBuffer &outputBuffer): outputBuffer_(outputBuffer) {}

void OutputWriter::writeError(const string &error) {
    outputBuffer_.addSubBuffer();
    int statusNum = -1; // should be made enum
    outputBuffer_.appendData(statusNum);
    outputBuffer_.appendData(error);
    outputBuffer_.finalize();
}

void OutputWriter::writeStatusMessage (const string &message) {
    outputBuffer_.addSubBuffer();
    int statusNum = -2;
    outputBuffer_.appendData(statusNum);
    outputBuffer_.appendData(message);
    outputBuffer_.finalize();
}

void OutputWriter::writeHeader(const vector <string> &columnNames, const vector <mapd_data_t> &columnTypes) {
    // first get number of columns for this output table
    numColumns_ = columnNames.size();
    // name and type vector need to be the same
    assert (numColumns_ == columnTypes.size());
    columnTypes_ = columnTypes;
    columnSizes_.resize(numColumns_);
    outputBuffer_.addSubBuffer();
    for (mapd_size_t c = 0; c != numColumns_; ++c) {
        // fill in columnSizes vector as we iterate 
        // through colums (we will need this later)
        columnSizes_[c] = getByteSizeForType(columnTypes_[c]);
        outputBuffer_.appendData(columnNames[c]);
        int colTypeInt = static_cast<int> (columnTypes[c]); 
        outputBuffer_.appendData(static_cast <void *> (&colTypeInt), 4);
    }
}

void OutputWriter::writeResults(const vector <mapd_addr_t> columnData, const mapd_size_t numRows) { 
    // remember that mapd_addr_t is an alias for unsigned char *
    assert (numColumns_ == columnData.size());
    outputBuffer_.addSubBuffer();
    for (mapd_size_t r = 0; r != numRows; ++r) {
        for (mapd_size_t c = 0; c != numColumns_; ++c) {
            outputBuffer_.appendData(columnData[c] + columnSizes_[c] * r, columnSizes_[c]); 
        }
    }
}

void OutputWriter::finalize() {
    outputBuffer_.finalize();
}

