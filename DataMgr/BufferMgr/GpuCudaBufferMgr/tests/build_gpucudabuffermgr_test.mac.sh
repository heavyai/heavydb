nvcc -O3 -c ../../CudaUtils.cu
g++ --std=c++11 -O3 -o gpuCudaBufferMgrTest GpuCudaBufferMgrTest.cpp ../GpuCudaBufferMgr.cpp CudaUtils.o ../GpuCudaBuffer.cpp  ../../BufferMgr.cpp ../../../Encoder.cpp ../../Buffer.cpp ../../CpuBufferMgr/CpuBufferMgr.cpp ../../CpuBufferMgr/CpuBuffer.cpp -L/usr/local/lib -lboost_filesystem-mt -lboost_timer-mt -lboost_system-mt -lgtest -L/usr/local/cuda/lib -lcudart
