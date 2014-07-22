#ifndef OUTPUT_BUFFER_H
#define OUTPUT_BUFFER_H

#include <vector>
#include <list>
#include <queue>
#include <string>

class OutputBuffer {

    private:
        std::queue<std::vector <char>, std::list <std::vector <char> > > dataQueue_;
        bool isFinal_;

    public:

        OutputBuffer (): isFinal_(false) {}

        inline void addSubBuffer () {
            dataQueue_.push(std::vector <char>);
        }
        
        inline void finalize() {
            isFinal_ = true;
            // if this was multithreaded would notify consumer this was last
            // element
        }

        void writeData (const void *data, const size_t size);

        void writeData (const char *data, const size_t size);// copies c-style string (not null-terminated)

        void writeData (const std::string &data);

        template <typename T> void writeData (T data) { // must leave in header file because templated
            size_t dataSize = sizeof(T);
            char * dataCharPtr = static_cast <char *> (data);
            dataQueue_.back().insert(dataQueue_.end(), dataCharPtr, dataCharPtr + size);
        }

        inline bool empty () const {
            return dataQueue_.size() == 0;
        }

        inline size_t size() const {
            return dataQueue_.size();
        }

        inline const std::vector <char>& front() const {
            return dataQueue_.front();
        }

        inline void pop() {
            dataQueue_.pop();
        }
};

#endif // OUTPUT_BUFFER_H
