#ifndef OUTPUT_WRITER_H
#define OUTPUT_WRITER_H

#include <string>

class OutputBuffer; //forward declaration

class OutputWriter {
    private:
        OutputBuffer &outputBuffer_;

    public:
        OutputWriter(OutputBuffer &outputBuffer);

        void writeError(const std::string &error);

        void writeStatusMessage (const std::string &message);
};

#endif
