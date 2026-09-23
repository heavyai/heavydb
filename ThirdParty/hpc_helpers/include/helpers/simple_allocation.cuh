#ifndef HELPERS_SIMPLE_ALLOCATION_CUH
#define HELPERS_SIMPLE_ALLOCATION_CUH

#ifdef __NVCC__

    #include <cassert>
    #include <iostream>
    #include <stdexcept>

    #include "cuda_helpers.cuh"

    namespace helpers {

    enum class DataLocation {Host, PinnedHost, Device};

    template<DataLocation location, class T>
    struct SimpleAllocator;

    template<class T>
    struct SimpleAllocator<DataLocation::Host, T>{
        T* allocate(size_t elements){
            T* ptr{};
            ptr = new T[elements];

            assert(ptr != nullptr);
            return ptr;
        }

        void deallocate(T* ptr){
            delete [] ptr;
        }
    };

    template<class T>
    struct SimpleAllocator<DataLocation::PinnedHost, T>{
        T* allocate(size_t elements){
            T* ptr{};
            cudaError_t err = cudaMallocHost(&ptr, elements * sizeof(T));
            if(err != cudaSuccess){
                std::cerr << "SimpleAllocator: Failed to allocate " << (elements) << " * " << sizeof(T)
                            << " = " << (elements * sizeof(T))
                            << " bytes using cudaMallocHost!\n";

                throw std::bad_alloc();
            }

            assert(ptr != nullptr);

            return ptr;
        }

        void deallocate(T* ptr){
            cudaFreeHost(ptr); CUERR;
        }
    };

    template<class T>
    struct SimpleAllocator<DataLocation::Device, T>{
        T* allocate(size_t elements){
            T* ptr;
            cudaError_t err = cudaMalloc(&ptr, elements * sizeof(T));
            if(err != cudaSuccess){
                std::cerr << "SimpleAllocator: Failed to allocate " << (elements) << " * " << sizeof(T)
                            << " = " << (elements * sizeof(T))
                            << " bytes using cudaMalloc!\n";

                throw std::bad_alloc();
            }

            assert(ptr != nullptr);

            return ptr;
        }

        void deallocate(T* ptr){
            cudaFree(ptr); CUERR;
        }
    };


    template<DataLocation location, class T, int overprovisioningPercent = 10>
    struct SimpleAllocation{
        using Allocator = SimpleAllocator<location, T>;

        static_assert(overprovisioningPercent >= 0, "overprovisioningPercent < 0");

        static constexpr size_t getOverprovisionedSize(size_t requiredSize){
            if(overprovisioningPercent <= 0){
                return requiredSize;
            }else{
                const double onePercent = requiredSize / 100.0;
                const size_t extra = onePercent * overprovisioningPercent;
                return requiredSize + std::min(std::size_t(1), extra);
            }
        }

        T* data_{};
        size_t size_{};
        size_t capacity_{};

        SimpleAllocation() : SimpleAllocation(0){}
        SimpleAllocation(size_t size){
            resize(size);
        }

        SimpleAllocation(const SimpleAllocation&) = delete;
        SimpleAllocation& operator=(const SimpleAllocation&) = delete;

        SimpleAllocation(SimpleAllocation&& rhs) noexcept{
            *this = std::move(rhs);
        }

        SimpleAllocation& operator=(SimpleAllocation&& rhs) noexcept{
            if(data_ != nullptr){
                Allocator alloc;
                alloc.deallocate(data_);
            }

            data_ = rhs.data_;
            size_ = rhs.size_;
            capacity_ = rhs.capacity_;

            rhs.data_ = nullptr;
            rhs.size_ = 0;
            rhs.capacity_ = 0;

            return *this;
        }

        ~SimpleAllocation(){
            destroy();
        }

        friend void swap(SimpleAllocation& l, SimpleAllocation& r) noexcept{
            using std::swap;

            swap(l.data_, r.data_);
            swap(l.size_, r.size_);
            swap(l.capacity_, r.capacity_);
        }

        void destroy(){
            if(data_ != nullptr){
                Allocator alloc;
                alloc.deallocate(data_);
                data_ = nullptr;
            }
            size_ = 0;
            capacity_ = 0;
        }

        T& operator[](size_t i){
            return get()[i];
        }

        const T& operator[](size_t i) const{
            return get()[i];
        }

        T& at(size_t i){
            if(i < size()){
                return operator[](i);
            }else{
                throw std::out_of_range("SimpleAllocation::at out-of-bounds access.");
            }
        }

        const T& at(size_t i) const{
            if(i < size()){
                return operator[](i);
            }else{
                throw std::out_of_range("SimpleAllocation::at out-of-bounds access.");
            }
        }

        T* operator+(size_t i) const{
            return get() + i;
        }

        operator T*(){
            return get();
        }

        operator const T*() const{
            return get();
        }


        //size is number of elements of type T
        //return true if reallocation occured
        bool resize(size_t newsize){
            size_ = newsize;

            if(capacity_ < newsize){
                Allocator alloc;
                alloc.deallocate(data_);
                const size_t newCapacity = getOverprovisionedSize(newsize);
                data_ = alloc.allocate(newCapacity);
                capacity_ = newCapacity;

                return true;
            }else{
                return false;
            }
        }

        //reserve enough memory for at least max(newCapacity,newSize) elements, and set size to newSize
        //return true if reallocation occured
        bool reserveAndResize(size_t newCapacity, size_t newSize){
            size_ = newSize;

            newCapacity = std::max(newCapacity, newSize);

            if(capacity_ < newCapacity){
                Allocator alloc;
                alloc.deallocate(data_);
                data_ = alloc.allocate(newCapacity);
                capacity_ = newCapacity;

                return true;
            }else{
                return false;
            }
        }

        T* get() const{
            return data_;
        }

        size_t size() const{
            return size_;
        }

        size_t& sizeRef(){
            return size_;
        }

        size_t sizeInBytes() const{
            return size() * sizeof(T);
        }

        size_t capacity() const{
            return capacity_;
        }

        size_t capacityInBytes() const{
            return capacity() * sizeof(T);
        }

        T* data() const noexcept{
            return data_;
        }

        T* begin() const noexcept{
            return data();
        }

        T* end() const noexcept{
            return data() + size();
        }

        bool empty() const noexcept{
            return size() == 0;
        }
    };

    template<class T, int overprovisioningPercent = 10>
    using SimpleAllocationHost = SimpleAllocation<DataLocation::Host, T, overprovisioningPercent>;

    template<class T, int overprovisioningPercent = 10>
    using SimpleAllocationPinnedHost = SimpleAllocation<DataLocation::PinnedHost, T, overprovisioningPercent>;

    template<class T, int overprovisioningPercent = 10>
    using SimpleAllocationDevice = SimpleAllocation<DataLocation::Device, T, overprovisioningPercent>;

    } // namespace helpers

#endif

#endif /* HELPERS_SIMPLE_ALLOCATION_CUH */

