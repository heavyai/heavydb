/**
 * @file    OutputWriter.h
 * @author  Todd Mostak <todd@map-d.com>
 * @brief   This file contains the class specification and related data structures for OutputWriter.
 *
 * OutputWriter wraps and manages OutputBuffer
 *
 * @see OutputBuffer.h
 */

#ifndef OUTPUT_WRITER_H
#define OUTPUT_WRITER_H

#include <string>

class OutputBuffer; //forward declaration

/**
 * @type OutputWriter
 * @brief OutputWriter is responsible for writing to OutputBuffer.
 *
 * OutputWriter wraps an instance of OutputBuffer and has methods
 * for writing data to it - including error and status messages.
 *
 * @see OutputBuffer
 */

class OutputWriter {
    private:
        OutputBuffer &outputBuffer_;

    public:
        /**
         * @brief Constructor takes reference to an instance
         * of OutputBuffer as this already should have been
         * instanciated by connection
         * @param outputBuffer OutputBuffer to write to
         *
         * @see OutputBuffer
         */

        OutputWriter(OutputBuffer &outputBuffer);

        /**
         * @brief Write error string to OutputBuffer.
         * @param error Error string
         */

        void writeError(const std::string &error);

        /**
         * @brief Write status message to OutputBuffer.
         * @param message Status message string
         */

        void writeStatusMessage (const std::string &message);
};

#endif
