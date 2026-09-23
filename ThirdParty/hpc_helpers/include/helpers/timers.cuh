#ifndef HELPERS_TIMERS_CUH
#define HELPERS_TIMERS_CUH

#include <chrono>
#include <ostream>
#include <iostream>


namespace helpers {

    class CpuTimer{
    public:

        CpuTimer() : CpuTimer("anonymous timer", std::cout)
        {
        }

        CpuTimer(const std::string& label) : CpuTimer(label, std::cout)
        {
        }

        CpuTimer(const std::string& label, std::ostream& outputstream)
            : ongoing(true), calculatedDelta(true), elapsedTime(0),
            begin(std::chrono::system_clock::now()),
            end(std::chrono::system_clock::now()),
            os(outputstream),
            name(label)
        {
        }

        ~CpuTimer(){
            if(ongoing){
                stop();
            }
        }

        void start(){
            if(!calculatedDelta){
                const std::chrono::duration<double> delta = end - begin;
                elapsedTime += delta.count();
                calculatedDelta = true;
            }

            begin = std::chrono::system_clock::now();
            end = std::chrono::system_clock::now();
            ongoing = true;
        }

        void stop(){
            end = std::chrono::system_clock::now();
            ongoing = false;
            calculatedDelta = false;
        }

        void reset(){
            ongoing = false;
            calculatedDelta = true;
            elapsedTime = 0;
        }

        double elapsed(){
            if(ongoing){
                stop();
            }

            if(!calculatedDelta){
                const std::chrono::duration<double> delta = end - begin;
                elapsedTime += delta.count();
                calculatedDelta = true;
            }

            return elapsedTime;
        }

        void print(){
            os << "# elapsed time ("<< name <<"): " << elapsed()  << "s\n";
        }

        void print_throughput(std::size_t bytes, int num){

            const double delta = elapsed(); //seconds

            const double gb = ((bytes)*(num))/1073741824.0;
            const double throughput = gb/delta;
            const double ops = (num)/delta;

            os << "THROUGHPUT: " << delta << " s @ " << gb << " GB "
                << "-> " << ops << " elements/s or " <<
                throughput << " GB/s (" << name << ")\n";
        }

    private:

        bool ongoing;
        bool calculatedDelta;
        double elapsedTime;
        std::chrono::time_point<std::chrono::system_clock> begin;
        std::chrono::time_point<std::chrono::system_clock> end;
        std::ostream& os;
        std::string name;
    };


    #ifdef __CUDACC__

    class GpuTimer{
    public:

        GpuTimer()
            : GpuTimer(0, "anonymous timer", std::cout)
        {
            //default stream, current device
        }

        GpuTimer(const std::string& label)
            : GpuTimer(0, label, std::cout)
        {
            //default stream, current device
        }


        GpuTimer(cudaStream_t stream, const std::string& label)
            : GpuTimer(stream, label, std::cout)
        {
            //user-defined stream, current device
        }

        GpuTimer(cudaStream_t stream, const std::string& label, std::ostream& outputstream)
            : calculatedDelta(true), elapsedTime(0), os(outputstream)
        {
            //user-defined stream, current device

            int curGpu = 0;
            cudaGetDevice(&curGpu);

            init(stream, label, curGpu);
            start();
        }

        GpuTimer(cudaStream_t stream, const std::string& label, int deviceId)
            : GpuTimer(stream, label, deviceId, std::cout)
        {
            //user-defined stream, user-defined device
        }

        GpuTimer(cudaStream_t stream, const std::string& label, int deviceId, std::ostream& outputstream)
            : calculatedDelta(true), elapsedTime(0), os(outputstream)
        {
            //user-defined stream, user-defined device

            init(stream, label, deviceId);
            start();
        }

        ~GpuTimer(){
            if(ongoing){
                stop();
            }

            int curGpu = 0;
            cudaGetDevice(&curGpu);
            cudaSetDevice(gpu);

            cudaEventDestroy(begin);
            cudaEventDestroy(end);

            cudaSetDevice(curGpu);
        }

        void start(){
            if(!calculatedDelta){
                float delta = 0.0f;
                cudaEventSynchronize(end);
                cudaEventElapsedTime(&delta, begin, end);
                elapsedTime += delta;
                calculatedDelta = true;
            }

            ongoing = true;

            int curGpu = 0;
            cudaGetDevice(&curGpu);
            cudaSetDevice(gpu);

            cudaEventRecord(begin, timedstream);

            cudaSetDevice(curGpu);
        }

        void stop(){
            int curGpu = 0;
            cudaGetDevice(&curGpu);
            cudaSetDevice(gpu);

            cudaEventRecord(end, timedstream);
            ongoing = false;
            calculatedDelta = false;

            cudaSetDevice(curGpu);
        }

        void reset(){
            ongoing = false;
            calculatedDelta = true;
            elapsedTime = 0;
        }

        float elapsed(){
            if(ongoing){
                stop();
            }

            if(!calculatedDelta){
                float delta = 0.0f;
                cudaEventSynchronize(end);
                cudaEventElapsedTime(&delta, begin, end);
                elapsedTime += delta;
                calculatedDelta = true;
            }

            return elapsedTime;
        }

        void print(){
            os << "TIMING: " << elapsed() << " ms (" << name << ")\n";
        }

        void print_throughput(std::size_t bytes, int num){
            const float delta = elapsed();

            const double gb = ((bytes)*(num))/1073741824.0;
            const double throughput = gb/((delta)/1000.0);
            const double ops = (num)/((delta)/1000.0);

            os << "THROUGHPUT: " << delta << " ms @ " << gb << " GB "
                << "-> " << ops << " elements/s or " <<
                throughput << " GB/s (" << name << ")\n";
        }

    private:

        void init(cudaStream_t stream, const std::string& label, int deviceId){
            gpu = deviceId;
            timedstream = stream;
            name = label;

            int curGpu = 0;
            cudaGetDevice(&curGpu);
            cudaSetDevice(gpu);

            cudaEventCreate(&begin);
            cudaEventCreate(&end);

            cudaSetDevice(curGpu);
        }


        bool ongoing;
        bool calculatedDelta;
        int gpu;
        float elapsedTime;
        cudaStream_t timedstream;
        cudaEvent_t begin;
        cudaEvent_t end;
        std::ostream& os;
        std::string name;
    };



    #endif

} //namespace helpers


#endif /* HELPERS_TIMERS_CUH */
