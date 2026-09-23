#ifndef HELPERS_COOP_GROUP_HELPERS_CUH
#define HELPERS_COOP_GROUP_HELPERS_CUH

#include "cuda_helpers.cuh"

/*
    Like cg::thread_block_tile<1>, but usable from the host.
*/
struct SingleThreadGroup{
public:

    HOSTDEVICEQUALIFIER
    constexpr void sync() const noexcept{
        ; //no-op
    }

    HOSTDEVICEQUALIFIER
    constexpr unsigned long long thread_rank() const noexcept{
        return 0;
    }

    HOSTDEVICEQUALIFIER
    constexpr unsigned long long size() const noexcept{
        return 1;
    }

    HOSTDEVICEQUALIFIER
    constexpr unsigned long long meta_group_size() const noexcept{
        return 1;
    }

    HOSTDEVICEQUALIFIER
    constexpr unsigned long long meta_group_rank() const noexcept{
        return 0;
    }

    template<class T>
    HOSTDEVICEQUALIFIER
    constexpr T shfl(T var, unsigned int src_rank) const noexcept{
        return var;
    }

    template<class T>
    HOSTDEVICEQUALIFIER
    constexpr T shfl_up(T var, int /*delta*/) const noexcept{
        return var;
    }

    template<class T>
    HOSTDEVICEQUALIFIER
    constexpr T shfl_down(T var, int /*delta*/) const noexcept{
        return var;
    }

    template<class T>
    HOSTDEVICEQUALIFIER
    constexpr T shfl_xor(T var, int /*delta*/) const noexcept{
        return var;
    }

    HOSTDEVICEQUALIFIER
    constexpr int any(int predicate) const noexcept{
        return predicate > 0;
    }

    HOSTDEVICEQUALIFIER
    constexpr int all(int predicate) const noexcept{
        return predicate > 0;
    }

    HOSTDEVICEQUALIFIER
    constexpr unsigned int ballot(int predicate) const noexcept{
        return predicate > 0;
    }

    template<class T>
    HOSTDEVICEQUALIFIER
    constexpr unsigned int match_any(T /*val*/) const noexcept{
        return 1u;
    }

    template<class T>
    HOSTDEVICEQUALIFIER
    constexpr unsigned int match_all(T /*val*/, int &pred) const noexcept{
        pred = 1;
        return 1u;
    }


};

#endif /* HELPERS_COOP_GROUP_HELPERS_CUH */
