#include "OutputBuffer.h"

using std::string;
using std::vector;


void OutputBuffer::writeData (const void *data, const size_t size) {
    const char * dataCharPtr = reinterpret_cast <const char *> (data);
    dataQueue_.back().insert(dataQueue_.back().end(), dataCharPtr, dataCharPtr + size);
}

void OutputBuffer::writeData (const char *data, const size_t size) { // copies c-style string (not null-terminated)
    writeData(static_cast<int>(size));
    dataQueue_.back().insert(dataQueue_.back().end(), data, data + size);
}

void OutputBuffer::writeData (const string &data) {
    writeData(static_cast<int>(data.size()));
    const char * dataCharPtr = data.data(); // could be null terminated
    dataQueue_.back().insert(dataQueue_.back().end(), dataCharPtr, dataCharPtr + data.size()); // use data.size() to ensure we don't copy a null terminator
}

void OutputBuffer::writeLastSubBufferSize() {
    if (dataQueue_.size() > 0) { // won't work if things are being pulled off the other end - will need another measure for size
        vector <char> &subBuffer = dataQueue_.back();
        int subBufferSize = subBuffer.size() - 4;  // should not include 4 bytes reserved to store buffer size in actual calculated size
        char * dataCharPtr = reinterpret_cast <char *> (&subBufferSize);
        subBuffer.insert(subBuffer.begin(), dataCharPtr, dataCharPtr + 4);
    }
}




