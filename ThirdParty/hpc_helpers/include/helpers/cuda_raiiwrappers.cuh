#ifndef HELPERS_CUDA_RAII_WRAPPERS_CUH
#define HELPERS_CUDA_RAII_WRAPPERS_CUH


#ifdef __NVCC__

#include "cuda_helpers.cuh"

class CudaEvent{
public:
    CudaEvent(){
        cudaGetDevice(&deviceId); CUERR;
        cudaEventCreate(&event); CUERR;
    }
    CudaEvent(unsigned int flags){
        cudaGetDevice(&deviceId); CUERR;
        cudaEventCreateWithFlags(&event, flags); CUERR;
    }

    CudaEvent(const CudaEvent&) = delete;
    CudaEvent(CudaEvent&& rhs){
        destroy();
        deviceId = rhs.deviceId;
        event = std::exchange(rhs.event, nullptr);
    }

    ~CudaEvent(){
        destroy();
    }

    void destroy(){
        if(event != nullptr){
            int d;
            cudaGetDevice(&d); CUERR;
            cudaSetDevice(deviceId); CUERR;

            cudaEventDestroy(event); CUERR;
            event = nullptr;

            cudaSetDevice(d); CUERR;
        }
    }

    CudaEvent& operator=(const CudaEvent&) = delete;

    CudaEvent& operator=(CudaEvent&& rhs){
        swap(*this, rhs);

        return *this;
    }

    friend void swap(CudaEvent& l, CudaEvent& r) noexcept
    {
        std::swap(l.deviceId, r.deviceId);
        std::swap(l.event, r.event);
    }

    cudaError_t query() const{
        return cudaEventQuery(event);
    }

    cudaError_t record(cudaStream_t stream = 0) const{
        return cudaEventRecord(event, stream);
    }

    cudaError_t synchronize() const{
        return cudaEventSynchronize(event);
    }

    cudaError_t elapsedTime(float* ms, cudaEvent_t end) const{
        return cudaEventElapsedTime(ms, event, end);
    }

    operator cudaEvent_t() const{
        return event;
    }

    int getDeviceId() const{
        return deviceId;
    }

    cudaEvent_t getEvent() const{
        return event;
    }
private:

    int deviceId{};
    cudaEvent_t event{};
};




class CudaStream{
public:
    CudaStream(){
        cudaGetDevice(&deviceId); CUERR;
        cudaStreamCreate(&stream); CUERR;
    }
    CudaStream(unsigned int flags){
        cudaGetDevice(&deviceId); CUERR;
        cudaStreamCreateWithFlags(&stream, flags); CUERR;
    }

    CudaStream(const CudaStream&) = delete;
    CudaStream(CudaStream&& rhs){
        destroy();
        deviceId = rhs.deviceId;
        stream = std::exchange(rhs.stream, nullptr);
    }

    ~CudaStream(){
        destroy();
    }

    void destroy(){
        if(stream != nullptr){
            int d;
            cudaGetDevice(&d); CUERR;
            cudaSetDevice(deviceId); CUERR;

            cudaStreamDestroy(stream); CUERR;
            stream = nullptr;

            cudaSetDevice(d); CUERR;
        }
    }

    CudaStream& operator=(const CudaStream&) = delete;

    CudaStream& operator=(CudaStream&& rhs){
        swap(*this, rhs);

        return *this;
    }

    friend void swap(CudaStream& l, CudaStream& r) noexcept
    {
        std::swap(l.deviceId, r.deviceId);
        std::swap(l.stream, r.stream);
    }

    cudaError_t query() const{
        return cudaStreamQuery(stream);
    }

    cudaError_t synchronize() const{
        return cudaStreamSynchronize(stream);
    }

    cudaError_t waitEvent(cudaEvent_t event, unsigned int flags) const{
        return cudaStreamWaitEvent(stream, event, flags);
    }

    operator cudaStream_t() const{
        return stream;
    }

    int getDeviceId() const{
        return deviceId;
    }

    cudaStream_t getStream() const{
        return stream;
    }
private:

    int deviceId{};
    cudaStream_t stream{};
};

#endif

#endif /* HELPERS_CUDA_RAII_WRAPPERS_CUH */
