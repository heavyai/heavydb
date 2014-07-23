#include "OutputWriter.h"
#include "OutputBuffer.h"

using std::string; 

OutputWriter::OutputWriter(OutputBuffer &outputBuffer): outputBuffer_(outputBuffer) {}

void OutputWriter::writeError(const string &error) {
    outputBuffer_.addSubBuffer();
    int statusNum = -1; // should be made enum
    outputBuffer_.writeData(statusNum);
    outputBuffer_.writeData(error);
    outputBuffer_.finalize();
}

void OutputWriter::writeStatusMessage (const string &message) {
    outputBuffer_.addSubBuffer();
    int statusNum = -2;
    outputBuffer_.writeData(statusNum);
    outputBuffer_.writeData(message);
    outputBuffer_.finalize();
}
