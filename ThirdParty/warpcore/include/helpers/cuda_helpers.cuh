#ifndef HELPERS_CUDA_HELPERS_CUH
#define HELPERS_CUDA_HELPERS_CUH

#include <iostream>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <assert.h>

#if CUDART_VERSION >= 9000
    #include <cooperative_groups.h>
#endif

#if !defined(CUDA_HELPERS_DONT_INCLUDE_V11_GROUP_HEADERS) && CUDART_VERSION >= 11000
    #include <cooperative_groups/reduce.h>
    #include <cooperative_groups/memcpy_async.h>
#endif

#include "hpc_helpers.h"

// error checking
#ifdef __CUDACC__
    #define CUERR {                                                            \
        cudaError_t err;                                                       \
        if ((err = cudaGetLastError()) != cudaSuccess) {                       \
            std::cout << "CUDA error: " << cudaGetErrorString(err) << " : "    \
                    << __FILE__ << ", line " << __LINE__ << std::endl;       \
            exit(1);                                                           \
        }                                                                      \
    }
#endif

// common CUDA constants
#define WARPSIZE (32)
#define MAXBLOCKSIZE (1024)
#define MAXSMEMBYTES (49152)
#define MAXCONSTMEMBYTES (65536)
#define H2D (cudaMemcpyHostToDevice)
#define D2H (cudaMemcpyDeviceToHost)
#define H2H (cudaMemcpyHostToHost)
#define D2D (cudaMemcpyDeviceToDevice)

// cross platform classifiers
#ifdef __CUDACC__
    #define HOSTDEVICEQUALIFIER  __host__ __device__
#else
    #define HOSTDEVICEQUALIFIER
#endif

#ifdef __CUDACC__
    #define INLINEQUALIFIER  __forceinline__
#else
    #define INLINEQUALIFIER inline
#endif

#ifdef __CUDACC__
    #define GLOBALQUALIFIER  __global__
#else
    #define GLOBALQUALIFIER
#endif

#ifdef __CUDACC__
    #define DEVICEQUALIFIER  __device__
#else
    #define DEVICEQUALIFIER
#endif

#ifdef __CUDACC__
    #define HOSTQUALIFIER  __host__
#else
    #define HOSTQUALIFIER
#endif

#ifdef __CUDACC__
    #define HD_WARNING_DISABLE #pragma hd_warning_disable
#else
    #define HD_WARNING_DISABLE
#endif

// redefinition of CUDA atomics for common cstdint types
#ifdef __CUDACC__
    // CAS
    DEVICEQUALIFIER INLINEQUALIFIER
    std::uint64_t atomicCAS(
        std::uint64_t* address,
        std::uint64_t compare,
        std::uint64_t val)
    {
        return atomicCAS(
            reinterpret_cast<unsigned long long int*>(address),
            static_cast<unsigned long long int>(compare),
            static_cast<unsigned long long int>(val));
    }

    // Add
    DEVICEQUALIFIER INLINEQUALIFIER
    std::uint64_t atomicAdd(std::uint64_t* address, std::uint64_t val)
    {
        return atomicAdd(
            reinterpret_cast<unsigned long long int*>(address),
            static_cast<unsigned long long int>(val));
    }

    // Exch
    DEVICEQUALIFIER INLINEQUALIFIER
    std::uint64_t atomicExch(std::uint64_t* address, std::uint64_t val)
    {
        return atomicExch(
            reinterpret_cast<unsigned long long int*>(address),
            static_cast<unsigned long long int>(val));
    }

    #if __CUDA_ARCH__ > 300

    // Min
    DEVICEQUALIFIER INLINEQUALIFIER
    std::uint64_t atomicMin(std::uint64_t* address, std::uint64_t val)
    {
        return atomicMin(
            reinterpret_cast<unsigned long long int*>(address),
            static_cast<unsigned long long int>(val));
    }

    // Max
    DEVICEQUALIFIER INLINEQUALIFIER
    std::uint64_t atomicMax(std::uint64_t* address, std::uint64_t val)
    {
        return atomicMax(
            reinterpret_cast<unsigned long long int*>(address),
            static_cast<unsigned long long int>(val));
    }

    // AND
    DEVICEQUALIFIER INLINEQUALIFIER
    std::uint64_t atomicAnd(std::uint64_t* address, std::uint64_t val)
    {
        return atomicAnd(
            reinterpret_cast<unsigned long long int*>(address),
            static_cast<unsigned long long int>(val));
    }

    // OR
    DEVICEQUALIFIER INLINEQUALIFIER
    std::uint64_t atomicOr(std::uint64_t* address, std::uint64_t val)
    {
        return atomicOr(
            reinterpret_cast<unsigned long long int*>(address),
            static_cast<unsigned long long int>(val));
    }

    // XOR
    DEVICEQUALIFIER INLINEQUALIFIER
    std::uint64_t atomicXor(std::uint64_t* address, uint64_t val)
    {
        return atomicXor(
            reinterpret_cast<unsigned long long int*>(address),
            static_cast<unsigned long long int>(val));
    }

    #endif

    DEVICEQUALIFIER INLINEQUALIFIER
    int ffs(std::uint32_t x)
    {
        return __ffs(x);
    }

    DEVICEQUALIFIER INLINEQUALIFIER
    int ffs(std::uint64_t x)
    {
        return __ffsll(x);
    }

    namespace helpers {

        #ifdef __CUDACC_EXTENDED_LAMBDA__
        template<class T>
        GLOBALQUALIFIER void lambda_kernel(T f)
        {
            f();
        }
        #endif

        // only valid for linear kernel i.e. y = z = 0
        DEVICEQUALIFIER INLINEQUALIFIER
        std::uint64_t global_thread_id() noexcept
        {
            return
                std::uint64_t(blockDim.x) * std::uint64_t(blockIdx.x) +
                std::uint64_t(threadIdx.x);
        }

        DEVICEQUALIFIER INLINEQUALIFIER
        unsigned int lane_id()
        {
            unsigned int lane;
            asm volatile("mov.u32 %0, %%laneid;" : "=r"(lane));
            return lane;
        }

        HOSTQUALIFIER INLINEQUALIFIER
        void init_cuda_context()
        {
            cudaFree(0);
        }

        HOSTQUALIFIER INLINEQUALIFIER
        std::uint64_t available_gpu_memory(float security_factor = 1.0)
        {
            assert(security_factor >= 1.0 && "invalid security factor");

            std::uint64_t free;
            std::uint64_t total;

            cudaMemGetInfo(&free, &total);

            return free / security_factor;
        }

        HOSTQUALIFIER INLINEQUALIFIER
        std::vector<std::uint64_t> available_gpu_memory(
            std::vector<std::uint64_t> device_ids,
            float security_factor = 1.0)
        {
            std::vector<std::uint64_t> available;

            for(auto id : device_ids)
            {
                cudaSetDevice(id);
                available.push_back(available_gpu_memory(security_factor));
            }

            return available;
        }

        HOSTQUALIFIER INLINEQUALIFIER
        std::uint64_t aggregated_available_gpu_memory(
            std::vector<std::uint64_t> device_ids,
            float security_factor = 1.0,
            bool uniform = false)
        {
            std::sort(device_ids.begin(), device_ids.end());
            device_ids.erase(
                std::unique(device_ids.begin(), device_ids.end()), device_ids.end());

            std::vector<std::uint64_t> available =
                available_gpu_memory(device_ids, security_factor);

            if(uniform)
            {
                std::uint64_t min_bytes =
                    *std::min_element(available.begin(), available.end());

                return min_bytes * device_ids.size();
            }
            else
            {
                std::uint64_t total = 0;

                for(auto bytes : available)
                {
                    total += bytes;
                }

                return total;
            }
        }

        #if CUDART_VERSION >= 9000
            template<typename index_t>
            DEVICEQUALIFIER INLINEQUALIFIER index_t atomicAggInc(index_t * ctr)
            {
                using namespace cooperative_groups;
                coalesced_group g = coalesced_threads();
                index_t prev;
                if (g.thread_rank() == 0) {
                    prev = atomicAdd(ctr, g.size());
                }
                prev = g.thread_rank() + g.shfl(prev, 0);
                return prev;
            }

            template<typename index_t>
            DEVICEQUALIFIER INLINEQUALIFIER index_t atomicAggAdd(index_t * ctr, index_t x)
            {
                namespace cg = cooperative_groups;

                // error case
                assert(x > 0);

                const auto g = cg::coalesced_threads();

                //inclusive prefix-sum
                index_t psum = x;
                for(std::uint32_t i = 1; i < g.size(); i <<= 1)
                {
                    const auto s = g.shfl_up(psum, i);

                    if(g.thread_rank() >= i) psum += s;
                }

                // last active lane increments ctr
                index_t offset;
                if(g.thread_rank() == g.size() - 1)
                {
                    offset = atomicAdd(ctr, psum);
                }

                // broadcast offset to group members
                offset = g.shfl(offset, g.size() - 1);

                return offset + psum - x;
            }
        #else
            template<typename index_t>
            DEVICEQUALIFIER INLINEQUALIFIER index_t atomicAggInc(index_t * ctr)
            {
                int lane = lane_id();
                //check if thread is active
                int mask = __ballot(1);
                //determine first active lane for atomic add
                int leader = __ffs(mask) - 1;
                index_t res;
                if (lane == leader) res = atomicAdd(ctr, __popc(mask));
                //broadcast to warp
                res = __shfl(res, leader);
                //compute index for each thread
                return res + __popc(mask & ((1 << lane) -1));
            }
        #endif

        DEVICEQUALIFIER INLINEQUALIFIER
        void die() { assert(0); } // mharris style

    } // namespace helpers

#endif

#endif /* HELPERS_CUDA_HELPERS_CUH */
