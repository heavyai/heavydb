#include "OutputBuffer.h"
#include <iostream>

using std::string;
using std::vector;
using std::cout;
using std::endl;


void OutputBuffer::appendData (const void *data, const size_t size) {
    const char * dataCharPtr = reinterpret_cast <const char *> (data);
    dataQueue_.back().insert(dataQueue_.back().end(), dataCharPtr, dataCharPtr + size);
}

void OutputBuffer::appendData (const char *data, const size_t size) { // copies c-style string (not null-terminated)
    appendData(static_cast<int>(size));
    dataQueue_.back().insert(dataQueue_.back().end(), data, data + size);
}

void OutputBuffer::appendData (const string &data) {
    appendData(static_cast<int>(data.size()));
    const char * dataCharPtr = data.data(); // could be null terminated
    dataQueue_.back().insert(dataQueue_.back().end(), dataCharPtr, dataCharPtr + data.size()); // use data.size() to ensure we don't copy a null terminator
}

void OutputBuffer::writeLastSubBufferSize() {
    if (dataQueue_.size() > 0) { // won't work if things are being pulled off the other end - will need another measure for size
        int subBufferSize = dataQueue_.back().size() - 4;  // should not include 4 bytes reserved to store buffer size in actual calculated size
        writeDataAtPos(subBufferSize,0);
    }
}




