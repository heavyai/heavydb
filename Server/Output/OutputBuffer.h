/**
 * @file    OutputBuffer.h
 * @author  Todd Mostak <todd@map-d.com>
 * @brief   This file contains the class specification 
 * for OutputBuffer.
 *
 * This file contains the OutputBuffer class specification,
 * which is essentially an (eventually thread-safe) queue by
 * which Database and the classes it calls can write output 
 * (including errors) to and that can be sent out by
 * TcpConnection to the client.
 *
 * Written to by OutputWriter
 *
 * @see OutputWriter.h
 */


#ifndef OUTPUT_BUFFER_H
#define OUTPUT_BUFFER_H

#include <vector>
#include <list>
#include <queue>
#include <string>

/**
 * @type OutputBuffer
 * @brief A queue of buffers that can be written to by Database
 * and friends.
 *
 * Wraps a queue of vectors of char called subBuffers.
 * All writes go to back subbuffer. Client of this class can
 * add a new subbuffer or finalize the queue at any time.
 * Currently instantciated by TcpConnection and passed by
 * reference to Database
 *
 * @see OutputWriter
 */

class OutputBuffer {

    private:
        std::queue<std::vector <char>, std::list <std::vector <char> > > dataQueue_;
        bool isFinal_;

    public:
        /**
         * @brief Constructor is default except initializes
         * isFinal_ private variable to false
         */

        OutputBuffer (): isFinal_(false) {}
        /**
         * @brief Pushes a new SubBuffer onto internal queue.
         *
         * First writes out the size of the last SubBuffer
         * and then pushes onto the queue a new SubBuffer of
         * size 4 - to reserve space for size written later.
         *
         * @see writeLastSubBufferSize()
         */
        inline void addSubBuffer () {
            writeLastSubBufferSize();
            dataQueue_.push(std::vector <char> (4)); // Start with 4 so we have space later to write in size
        }
       
        /**
         * @brief Finalizes OutputBuffer
         *
         * Like addSubBuffer(), first writes out size of last
         * SubBuffer and pushes a new one, but also sets 
         * isFinal_ flag.  Might in future need to notify
         * consumer thread that it just pushed last element.
         *
         * @see writeLastSubBufferSize()
         */

        inline void finalize() {
            writeLastSubBufferSize();
            isFinal_ = true;
            dataQueue_.push(std::vector <char> (4)); // Start with 4 so we have space later to write in size
            writeLastSubBufferSize();
            // if this was multithreaded would notify consumer this was last
            // element
        }
        
        /**
         * @brief Writes size of SubBuffer at back of queue to its first four bytes
         *
         * Only writes out size if internalQueue size is
         * greater than 0.
         */

        void writeLastSubBufferSize();

        /**
         * @brief Appends a void buffer to end of SubBuffer at the back of the queue.
         * @param data data to append
         * @size size in bytes of data to append
         */

        void appendData (const void *data, const size_t size);

        /**
         * @brief Appends a char buffer of size size to end of SubBuffer at the back of the queue.
         * @param data non-null terminated char data to append
         * @param size size of char data to append
         *
         * Expects string not to be null terminated.
         */

        void appendData (const char *data, const size_t size);

        /**
         * @brief Appends a std::string to end of SubBuffer at
         * the back of the internal queue
         * 
         * @param data std::string to append
         */

        void appendData (const std::string &data);

        /**
         * @brief Appends a templated POD type to end of SubBuffer
         * at the back of the internal queue.
         *
         * @param templated POD data to append
         */
        
        template <typename T> void appendData (T data) { // must leave in header file because templated
            size_t dataSize = sizeof(T);
            char * dataCharPtr = reinterpret_cast <char *> (&data);
            dataQueue_.back().insert(dataQueue_.back().end(), dataCharPtr, dataCharPtr + dataSize);
        }

        /**
         * @brief Writes a tempated POD type to position specified
         * by offset to the SubBuffer at the back of the internal
         * queue. 
         * @param data templated data to write
         * @param offset position in SubBuffer to write to
         *
         * No error checking is conducted to ensure that position
         * is less than the length of the buffer. Used by 
         * writeLastSubBufferSize() to write SubBuffer size
         * at the front of the SubBuffer.
         */

        template <typename T> void writeDataAtPos (T data, const size_t offset) { // must leave in header file because templated
            size_t dataSize = sizeof(T);
            char * dataCharPtr = reinterpret_cast <char *> (&data);
            std::copy(dataCharPtr,dataCharPtr+4,dataQueue_.back().begin() + offset);
        }


        inline bool empty () const {
            return dataQueue_.size() == 0;
        }

        inline size_t size() const {
            return dataQueue_.size();
        }

        /**
         * @brief Returns a reference to the front SubBuffer
         * in the internal queue.
         */

        inline const std::vector <char>& front() const {
            return dataQueue_.front();
        }

        inline void pop() {
            dataQueue_.pop();
        }
};

#endif // OUTPUT_BUFFER_H
