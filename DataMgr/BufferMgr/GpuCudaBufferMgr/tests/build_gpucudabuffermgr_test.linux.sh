nvcc -O3 -c ../../CudaUtils.cu
g++ --std=c++11 -O3 -o gpuCudaBufferMgrTest GpuCudaBufferMgrTest.cpp ../GpuCudaBufferMgr.cpp CudaUtils.o ../GpuCudaBuffer.cpp  ../../BufferMgr.cpp ../../Buffer.cpp ../../CpuBufferMgr/CpuBufferMgr.cpp ../../CpuBufferMgr/CpuBuffer.cpp -L/usr/local/lib -lboost_filesystem -lboost_timer -lboost_system -lgtest -L/usr/local/cuda/lib64 -lcudart -pthread
