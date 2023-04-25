#ifndef WARPCORE_RANDOM_DISTRIBUTIONS_CUH
#define WARPCORE_RANDOM_DISTRIBUTIONS_CUH

#include <limits>
#include <assert.h>
#include <iostream>
#include <kiss/kiss.cuh>
#include <warpcore/bloom_filter.cuh>

namespace warpcore
{

/*! \brief transforms interval (a, b) into interval (c, d)
* \tparam T data type
* \param[in] x value to be transformed into new interval
* \param[in] a lower bound of input interval
* \param[in] b upper bound of input interval
* \param[in] c lower bound of output interval
* \param[in] d upper bound of output interval
* \return value in new interval
*/
template<class T>
HOSTDEVICEQUALIFIER INLINEQUALIFIER
constexpr T transform_range(
    const T x,
    const T a,
    const T b,
    const T c,
    const T d) noexcept
{
    // check is intervals are valid
    assert(a < b);
    assert(c < d);
    // check if x is inside input interval
    assert(x >= a && x <= b);

    // see https://math.stackexchange.com/a/914843
    // TODO check for correctness
    //double tmp =  ((d - c) / (b - a)) * (x - a) * 1.0;
    return c + x % (d - c);
}

/*! \brief generates \c n uniform random elements on a CUDA device
* \tparam T data type to be generated
* \tparam Rng type of random number generator
* \param[out] out pointer to the output array
* \param[in] n number of elements to generate
* \param[in] seed initial random seed for the RNG
*/
template<class T, class Rng>
HOSTQUALIFIER  INLINEQUALIFIER
void uniform_distribution(
    T * out,
    std::uint64_t n,
    std::uint32_t seed) noexcept
{
    namespace wc = warpcore;

    // execute kernel
    helpers::lambda_kernel<<<4096, 32>>>
    ([=] DEVICEQUALIFIER
    {
        const std::uint32_t tid = blockDim.x * blockIdx.x + threadIdx.x;

        // generate initial local seed per thread
        const std::uint32_t local_seed =
            wc::hashers::MurmurHash<std::uint32_t>::hash(seed+tid);
            Rng rng{local_seed};
        T x;

        // grid-stride loop
        const auto grid_stride = blockDim.x * gridDim.x;
        for(std::uint64_t i = tid; i < n; i += grid_stride)
        {
            // generate random element
            x = rng.template next<T>();

            // write output
            out[i] = x;
        }
    })
    ; CUERR
}

template<class Rng = kiss::Kiss<std::uint32_t>>
HOSTQUALIFIER INLINEQUALIFIER
void uniform_distribution(
    std::uint32_t * out,
    std::uint64_t n,
    std::uint32_t seed) noexcept
{
    uniform_distribution<std::uint32_t, Rng>(
        out,
        n,
        seed);
}

template<class Rng = kiss::Kiss<std::uint64_t>>
HOSTQUALIFIER INLINEQUALIFIER
void uniform_distribution(
    std::uint64_t * out,
    std::uint64_t n,
    std::uint32_t seed) noexcept
{
    uniform_distribution<std::uint64_t, Rng>(
        out,
        n,
        seed);
}

template<class Rng = kiss::Kiss<std::uint32_t>>
HOSTQUALIFIER INLINEQUALIFIER
void uniform_distribution(
    float * out,
    std::uint64_t n,
    std::uint32_t seed) noexcept
{
    uniform_distribution<std::uint32_t, Rng>(
        reinterpret_cast<std::uint32_t*>(out),
        n,
        seed);
}

template<class Rng = kiss::Kiss<std::uint32_t>>
HOSTQUALIFIER INLINEQUALIFIER
void uniform_distribution(
    double * out,
    std::uint64_t n,
    std::uint32_t seed) noexcept
{
    uniform_distribution<std::uint32_t, Rng>(
        reinterpret_cast<std::uint32_t*>(out),
        n,
        seed);
}

/*! \brief generates \c n unique random elements on a CUDA device
* \tparam T data type to be generated
* \tparam Rng type of random number generator
* \param[out] out pointer to the output array
* \param[in] n number of elements to generate
* \param[in] seed initial random seed for the RNG
*/
template<class T, class Rng>
HOSTQUALIFIER INLINEQUALIFIER
void unique_distribution(
    T * out,
    std::uint64_t n,
    std::uint32_t seed) noexcept
{
    namespace wc = warpcore;
    namespace cg = cooperative_groups;
    using hasher_t = wc::hashers::MurmurHash<std::uint32_t>;
    using filter_t = wc::BloomFilter<T>;

    filter_t bf{n/4096, 8, T(seed)}; // TODO parameter search

    // execute kernel
    helpers::lambda_kernel
    <<<4096, 32>>>
    ([=] DEVICEQUALIFIER () mutable
    {
        const std::uint32_t tid = blockDim.x * blockIdx.x + threadIdx.x;
        // generate initial local seed per thread
        const std::uint32_t local_seed =
            hasher_t::hash(hasher_t::hash(tid) + seed);
        Rng rng{local_seed};
        // cooperative group to use for bloom filter
        const auto group =
            cg::tiled_partition<filter_t::cg_size()>(cg::this_thread_block());
        T x;
        bool is_unique;

        // grid-stride loop
        const auto grid_stride = blockDim.x * gridDim.x;
        for(std::uint64_t i = tid; i < n; i += grid_stride)
        {
            // do while randomly drawn element is not unique
            do
            {
                // generate random element
                x = rng.template next<T>();

                // insert into bloom filter and check if unique
                is_unique = bf.insert_and_query(x, group);
            }
            while(!is_unique);

            // write output
            out[i] = x;
        }
    }
    ); CUERR
}

template<class Rng = kiss::Kiss<std::uint32_t>>
HOSTQUALIFIER INLINEQUALIFIER
void unique_distribution(
    std::uint32_t * out,
    std::uint64_t n,
    std::uint32_t seed) noexcept
{
    unique_distribution<std::uint32_t, Rng>(
        out,
        n,
        seed);
}

template<class Rng = kiss::Kiss<std::uint64_t>>
HOSTQUALIFIER INLINEQUALIFIER
void unique_distribution(
    std::uint64_t * out,
    std::uint64_t n,
    std::uint32_t seed) noexcept
{
    unique_distribution<std::uint64_t, Rng>(
        out,
        n,
        seed);
}

template<class Rng = kiss::Kiss<std::uint32_t>>
HOSTQUALIFIER INLINEQUALIFIER
void unique_distribution(
    float * out,
    std::uint64_t n,
    std::uint32_t seed) noexcept
{
    unique_distribution<std::uint32_t, Rng>(
        reinterpret_cast<std::uint32_t*>(out),
        n,
        seed);
}

template<class Rng = kiss::Kiss<std::uint32_t>>
HOSTQUALIFIER INLINEQUALIFIER
void unique_distribution(
    double * out,
    std::uint64_t n,
    std::uint32_t seed) noexcept
{
    unique_distribution<std::uint32_t, Rng>(
        reinterpret_cast<std::uint32_t*>(out),
        n,
        seed);
}

/*! \brief generates \c n zipf distributed random elements on a CUDA device
* \tparam T data type to be generated
* \tparam Rng type of random number generator
* \tparam P data type used for probabilities
* \param[in] in set of unique elements to be considered
* \param[in] n_in cardinality of input set
* \param[out] out pointer to the output array
* \param[in] n_out number of elements to generate
* \param[in] s skew of the distribution
* \param[in] seed initial random seed for the RNG
*/
template<class T, class Rng = kiss::Kiss<std::uint32_t>, class P = double>
HOSTQUALIFIER INLINEQUALIFIER
void zipf_distribution(
    T * in,
    std::uint64_t n_in,
    T * out,
    std::uint64_t n_out,
    P s,
    std::uint32_t seed) noexcept
{
    namespace wc = warpcore;

    // zipf is undefined for s = 1.0
    assert(s != 1.0);

    helpers::lambda_kernel
    <<<4096, 32>>>
    ([=] DEVICEQUALIFIER
    {
        const std::uint32_t tid = blockDim.x * blockIdx.x + threadIdx.x;
        // generate initial local seed per thread
        const std::uint32_t local_seed =
            wc::hashers::MurmurHash<std::uint32_t>::hash(seed+tid);
        Rng rng{local_seed};

        // see https://medium.com/@jasoncrease/rejection-sampling-the-zipf-distribution-6b359792cffa
        const P t = (pow(double(n_in), 1.0 - s) - s) / (1.0 - s);
        std::uint64_t k;
        P y;
        P rat;

        // grid-stride loop
        const auto grid_stride = blockDim.x * gridDim.x;
        for(std::uint64_t i = tid; i < n_out; i += grid_stride)
        {
            do
            {
                // generate random element
                P p = rng.template next<P>();
                // calculate bounding function of inverse CDF
                P inv_b =
                    ((p * t) <= 1.0) ? p * t : pow((p * t) * (1.0 - s) + s, 1.0 / (1.0 - s));
                // draw a target rank
                k = (T)(inv_b + 1.0);
                // generate random element
                y = rng.template next<P>();
                // added corner case if k <= 1.0
                P b = (k <= 1.0) ? 1.0 / t : pow(inv_b, -s) / t;
                // calculate acceptance boundary
                rat = pow(double(k), -s) / (b * t);
            }
            while(y >= rat);

            // iff T is of index type and pointer in is a nullptr return rank k
            if((std::is_same<T, std::uint32_t>::value ||
                std::is_same<T, std::uint64_t>::value) &&
               in == nullptr)
            {
                out[i] = k + 1;
            }
            else
            {
                out[i] = in[k];
            }
        }
    }
    ); CUERR
}

} // namespace warpcore

#endif /* WARPCORE_RANDOM_DISTRIBUTIONS_CUH */
