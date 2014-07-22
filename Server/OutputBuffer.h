#ifndef OUTPUT_BUFFER_H
#define OUTPUT_BUFFER_H

class OutputBuffer {

    private:
        std::queue<std::vector <char>, std::list <std::vector <char> > > dataQueue_;

    public:
        void addSubBuffer () {
            dataQueue_.push(std::vector <char>);
        }
        
        void finalize() {
            dataQueue_.push(std::vector <char>);
            // if this was multithreaded would notify consumer this was last
            // element
        }

        void addData (const void *data, const size_t size) {
            char * dataCharPtr = static_cast <char *> (data);
            dataQueue_.back().insert(dataQueue_.end(), dataCharPtr, dataCharPtr + size);
        }

        void addData (const char *data, const size_t size) {
            dataQueue_.back().insert(dataQueue_.end(), dataCharPtr, dataCharPtr + size);
        }

        void addData (const string &data) {
            char * dataCharPtr = data.data(); // could be null terminated
            dataQueue_.back().insert(dataQueue_.end(), dataCharPtr, dataCharPtr + data.size()); // use data.size() to ensure we don't copy a null terminator
        }

        template <typename T> void addData (T data) {
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

        const std::vector <char>& front() const {
            return dataQueue_.front();
        }

        void pop() {
            dataQueue_.pop();
        }
};

#endif // OUTPUT_BUFFER_H
