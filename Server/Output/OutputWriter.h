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

#include "../../Shared/types.h"

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
        mapd_size_t numColumns_;
        std::vector<mapd_data_t> columnTypes_;
        std::vector<mapd_size_t> columnSizes_;
        inline mapd_size_t getByteSizeForType(const mapd_data_t dataType) {
            switch (dataType) {
                case INT_TYPE:
                case FLOAT_TYPE:
                    return 4;
                    break;
                case BOOLEAN_TYPE:
                    return 1;
                    break;
            }
        }

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
        
        /**
         * @brief Write table header to output buffer
         *
         * Writes the name and type of the data type for
         * each header columns.
         * Also sets columnTypes_ and columnSizes_ vectors
         * that are used in the writeResults method 
         *
         * @param columnNames vector containing name of each 
         * output column
         *
         * @param columnTypes vector containing mapd_data_t 
         * type of each output column
         */

        void writeHeader(const std::vector <std::string> &columnNames, const std::vector <mapd_data_t> &columnTypes);


        /**
         * @brief Write a result set to output buffer
         *
         * Writes numRows of the data in columnData to the 
         * outputBuffer in row-major format. This function
         * can be called more than once per result set
         * (i.e. for the output of each partition/fragment)
         *
         * @param columnData vector containing mapd_addr_t
         * pointers to raw data
         * @param numRows number of rows in this partition
         */

        void writeResults(const std::vector <mapd_addr_t> columnData, const mapd_size_t numRows);

        /**
         * @brief Calls finalize on OutputBuffer
         *
         * Should be called when calling method has written
         * all of its output.
         */

        void finalize();


};

#endif
