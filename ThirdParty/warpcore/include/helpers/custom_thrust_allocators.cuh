#ifndef HELPERS_CUSTOM_THRUST_ALLOCATORS_HPP
#define HELPERS_CUSTOM_THRUST_ALLOCATORS_HPP

#include <stdexcept>
#include <exception>
#include <iostream>

#ifdef __NVCC__

#include <thrust/device_malloc_allocator.h>

namespace helpers {

template<typename T>
struct ThrustUninitializedDeviceAllocator : thrust::device_malloc_allocator<T>{

  __host__ __device__
  void construct(T *p)
  {
    // no-op
  }
};

template<class T, bool allowFallback>
struct ThrustFallbackDeviceAllocator;

template<class T>
struct ThrustFallbackDeviceAllocator<T, true> : thrust::device_malloc_allocator<T> {
    using value_type = T;
    using super_t = thrust::device_malloc_allocator<T>;

    using pointer = typename super_t::pointer;
    using size_type = typename super_t::size_type;
    using reference = typename super_t::reference;
    using const_reference = typename super_t::const_reference;

    pointer allocate(size_type n){
        //std::cerr << "alloc" << std::endl;

        T* ptr = nullptr;
		cudaError_t status = cudaMalloc(&ptr, n * sizeof(T));
		if(status == cudaSuccess){
			//std::cerr << "cudaMalloc\n";
		}else{
			cudaGetLastError(); //reset the error of failed allocation

	    	status = cudaMallocManaged(&ptr, n * sizeof(T));
    		if(status != cudaSuccess){
    			throw std::bad_alloc();
    		}
    		int deviceId = 0;
    		status = cudaGetDevice(&deviceId);
    		if(status != cudaSuccess){
    			throw std::bad_alloc();
    		}
    		status = cudaMemAdvise(ptr, n * sizeof(T), cudaMemAdviseSetAccessedBy, deviceId);
    		if(status != cudaSuccess){
    			throw std::bad_alloc();
    		}
			//std::cerr << "cudaMallocManaged\n";
		}
		return thrust::device_pointer_cast(ptr);
	}

    void deallocate(pointer ptr, size_type n){
    	//std::cerr << "dealloc" << std::endl;

    	cudaError_t status = cudaFree(ptr.get());
    	if(status != cudaSuccess){
    		throw std::bad_alloc();
    	}
    }
};

template<class T>
struct ThrustFallbackDeviceAllocator<T, false> : thrust::device_malloc_allocator<T>{
	using value_type = T;

	using super_t = thrust::device_malloc_allocator<T>;

	using pointer = typename super_t::pointer;
	using size_type = typename super_t::size_type;
	using reference = typename super_t::reference;
	using const_reference = typename super_t::const_reference;

	pointer allocate(size_type n){
		//std::cerr << "alloc" << std::endl;

		T* ptr = nullptr;
		cudaError_t status = cudaMalloc(&ptr, n * sizeof(T));
		if(status == cudaSuccess){
			//std::cerr << "cudaMalloc\n";
		}else{
			cudaGetLastError(); //reset the error of failed allocation

    		throw std::bad_alloc();
		}
		return thrust::device_pointer_cast(ptr);
	}

    void deallocate(pointer ptr, size_type n){
    	//std::cerr << "dealloc" << std::endl;

    	cudaError_t status = cudaFree(ptr.get());
    	if(status != cudaSuccess){
    		throw std::bad_alloc();
    	}
    }
};

} // namespace helpers

#endif

#endif /* HELPERS_CUSTOM_THRUST_ALLOCATORS_HPP */

