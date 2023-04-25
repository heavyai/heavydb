#ifndef WARPCORE_BLOOM_FILTER_CUH
#define WARPCORE_BLOOM_FILTER_CUH

#include "base.cuh"

namespace warpcore
{

/*! \brief bloom filter
 * \tparam Key key type (\c std::uint32_t or \c std::uint64_t)
 * \tparam Hasher hasher from \c warpcore::hashers
 * \tparam Slot slot type (\c std::uint32_t or \c std::uint64_t)
 * \tparam CGSize size of cooperative group
 */
template<
    class Key,
    class Hasher = defaults::hasher_t<Key>,
    class Slot = std::uint64_t,
    index_t CGSize = 1>
class BloomFilter
{
    static_assert(
        checks::is_valid_key_type<Key>(),
        "invalid key type");

    static_assert(
        checks::is_hasher<Hasher>(),
        "not a valid hasher type");

    static_assert(
        checks::is_valid_slot_type<Slot>(),
        "invalid slot type");

    static_assert(
        checks::is_valid_cg_size(CGSize),
        "invalid cooperative group size");

public:
    using key_type = Key;
    using value_type = bool;
    using index_type = index_t;
    using slot_type = Slot;
    using status_type = Status;

    /*! \brief get cooperative group size
     * \return cooperative group size
     */
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    static constexpr index_type cg_size() noexcept
    {
        return CGSize;
    }

    /*! \brief get bits per slot
     * \return number of bits
     */
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    static constexpr index_type slot_bits() noexcept
    {
        return sizeof(slot_type) * CHAR_BIT;
    }

    /*! \brief get bits per cooperative group block of slots
     * \return number of bits
     */
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    static constexpr index_type block_bits() noexcept
    {
        return slot_bits() * cg_size();
    }

    /*! \brief constructor
     * \param[in] num_bits total number of bits (m) of the bloom filter
     * \param[in] k number of hash functions to apply
     * \param[in] seed random seed
     */
    HOSTQUALIFIER
    explicit BloomFilter(
        const index_type num_bits,
        const index_type k,
        const key_type seed = defaults::seed<key_type>()) noexcept :
        bloom_filter_(nullptr),
        num_bits_(num_bits),
        num_slots_(SDIV(num_bits, slot_bits())),
        num_blocks_(SDIV(num_slots_, cg_size())),
        k_(k),
        seed_(seed),
        is_copy_(false)
    {
        cudaMalloc(&bloom_filter_, sizeof(slot_type) * num_slots_);

        init();
    }

    /*! \brief copy-constructor (shallow)
     *  \param[in] object to be copied
     */
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    BloomFilter(const BloomFilter& o) noexcept :
        bloom_filter_(o.bloom_filter_),
        num_bits_(o.num_bits_),
        num_slots_(o.num_slots_),
        num_blocks_(o.num_blocks_),
        k_(o.k_),
        seed_(o.seed_),
        is_copy_(true)
    {}

    /*! \brief move-constructor
     *  \param[in] object to be moved
     */
    HOSTQUALIFIER INLINEQUALIFIER
    BloomFilter(BloomFilter&& o) noexcept :
        bloom_filter_(std::move(o.bloom_filter_)),
        num_bits_(std::move(o.num_bits_)),
        num_slots_(std::move(o.num_slots_)),
        num_blocks_(std::move(o.num_blocks_)),
        k_(std::move(o.k_)),
        seed_(std::move(o.seed_)),
        is_copy_(std::move(o.is_copy_))
    {
        o.is_copy_ = true;
    }

    #ifndef __CUDA_ARCH__
    /*! \brief destructor
     */
    HOSTQUALIFIER
    ~BloomFilter() noexcept
    {
        if(!is_copy_)
        {
            if(bloom_filter_ != nullptr) cudaFree(bloom_filter_);
        }
    }
    #endif

    /*! \brief (re)initialize the hash table
     * \param[in] seed random seed
     * \param[in] stream CUDA stream in which this operation is executed in
     */
    HOSTQUALIFIER INLINEQUALIFIER
    void init(
        const key_type seed,
        const cudaStream_t stream = 0) noexcept
    {
        seed_ = seed;

        kernels::memset
        <<<SDIV(num_slots_, WARPCORE_BLOCKSIZE), WARPCORE_BLOCKSIZE, 0, stream>>>
        (bloom_filter_, num_slots_);
    }

    /*! \brief (re)initialize the hash table
    * \param[in] stream CUDA stream in which this operation is executed in
    */
    HOSTQUALIFIER INLINEQUALIFIER
    void init(const cudaStream_t stream = 0) noexcept
    {
        init(seed_, stream);
    }

    /*! \brief inserts a key into the bloom filter
     * \param[in] key key to insert into the bloom filter
     * \param[in] group cooperative group
     */
    DEVICEQUALIFIER INLINEQUALIFIER
    void insert(
        const key_type key,
        const cg::thread_block_tile<cg_size()>& group) noexcept
    {
        const index_type slot_index =
            ((Hasher::hash(key + seed_) % num_blocks_) *
            cg_size() + group.thread_rank()) % num_slots_;

        slot_type slot = 0;
        for(index_type k = 0; k < k_; k++)
        {
            const key_type seeded_key = key + seed_ + k;
            const slot_type hash = Hasher::hash(seeded_key) % block_bits();

            if((hash / slot_bits()) == group.thread_rank())
            {
                slot |= slot_type(1) << (hash % slot_bits());
            }
        }

        if((slot & bloom_filter_[slot_index]) != slot)
        {
            atomicOr(bloom_filter_ + slot_index, slot);
        }
    }

    /*! \brief inserts a set of keys into the bloom filter
     * \param[in] keys_in pointer to keys to insert into the bloom filter
     * \param[in] num_in number of keys to insert
     * \param[in] stream CUDA stream in which this operation is executed in
     */
    HOSTQUALIFIER INLINEQUALIFIER
    void insert(
        const Key * const keys_in,
        const index_t num_in,
        const cudaStream_t stream = 0) noexcept
    {
        kernels::bloom_filter::insert
        <<<SDIV(num_in * cg_size(), WARPCORE_BLOCKSIZE), WARPCORE_BLOCKSIZE, 0, stream>>>
        (keys_in, num_in, *this);
    }

    /*! \brief retrieve a key
     * \param[in] key key to query
     * \param[in] group cooperative group
     * \return whether the key is already inside the filter or not
     */
    DEVICEQUALIFIER INLINEQUALIFIER
    bool retrieve(
        const key_type key,
        const cg::thread_block_tile<cg_size()>& group) const noexcept
    {
        const index_type slot_index =
            ((Hasher::hash(key+seed_) % num_blocks_) *
            cg_size() + group.thread_rank()) % num_slots_;

        slot_type slot = 0;
        for(index_type k = 0; k < k_; k++)
        {
            const key_type seeded_key = key + seed_ + k;
            const slot_type hash = Hasher::hash(seeded_key) % block_bits();

            if((hash / slot_bits()) == group.thread_rank())
            {
                slot |= slot_type(1) << (hash % slot_bits());
            }
        }

        return (group.all((slot & bloom_filter_[slot_index]) == slot)) ? true : false;
    }

    /*! \brief retrieve a set of keys
     * \param[in] keys_in pointer to keys
     * \param[in] num_in number of keys
     * \param[out] flags_out result per key
     ' \param[in] stream CUDA stream in which this operation is executed in
     */
    HOSTQUALIFIER INLINEQUALIFIER
    void retrieve(
        const key_type * const keys_in,
        const index_type num_in,
        bool * const flags_out,
        const cudaStream_t stream = 0) const noexcept
    {
        kernels::bloom_filter::retrieve
        <<<SDIV(num_in * cg_size(), WARPCORE_BLOCKSIZE), WARPCORE_BLOCKSIZE, 0, stream>>>
        (keys_in, num_in, flags_out, *this);
    }

    /*! \brief queries and subsequently inserts a key into the bloom filter
     * \note can only be used when \c CGSize==1 to prevent from race conditions
     * \param[in] key key to query
     * \param[in] group cooperative group this operation is executed in
     * \param[out] flag whether the key was already inside the filter before insertion
     */
    template<
        index_type CGSize_ = cg_size(),
        class = std::enable_if_t<CGSize_ == 1>>
    DEVICEQUALIFIER INLINEQUALIFIER
    bool insert_and_query(
         const key_type key,
         const cg::thread_block_tile<cg_size()>& group) noexcept
    {
        const index_type slot_index =
            ((Hasher::hash(key+seed_) % num_blocks_) *
            cg_size() + group.thread_rank()) % num_slots_;

        slot_type slot = slot_type{0};
        for(index_type k = 0; k < k_; k++)
        {
            const key_type seeded_key = key + seed_ + k;
            const slot_type hash = Hasher::hash(seeded_key) % block_bits();

            if((hash / slot_bits()) == group.thread_rank())
            {
                slot |= slot_type{1} << (hash % slot_bits());
            }
        }

        if((slot & bloom_filter_[slot_index]) != slot)
        {
            const auto old = atomicOr(bloom_filter_ + slot_index, slot);

            return ((slot & old) != slot) ? false : true;
        }
        else
        {
            return true;
        }
    }

    /*! \brief get number of bits (m)
     * \return number of bits (m)
     */
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    index_type num_bits() const noexcept
    {
        return num_bits_;
    }

    /*! \brief get number of slots
     * \return number of slots
     */
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    index_type num_slots() const noexcept
    {
        return num_slots_;
    }

    /*! \brief get number of blocks
     * \return number of blocks
     */
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    index_type num_blocks() const noexcept
    {
        return num_blocks_;
    }

    /*! \brief get number of hash functions (k)
     * \return number of hash functions (k)
     */
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    index_type k() const noexcept
    {
        return k_;
    }

    // TODO incorporate CG size
    /*! \brief estimated false positive rate of pattern-blocked bloom filter
     * \param[in] n number of inserted elements
     * \return false positive rate
     * \warning computationally expensive for large filters
     */
    HOSTQUALIFIER INLINEQUALIFIER
    double fpr(const index_type n) const noexcept
    {
        double res = 0.0;
        const double b = num_bits_ / block_bits();

        #pragma omp parallel for reduction(+:res)
        for(index_type i = 0; i < 5*n/(num_bits_ / block_bits()); ++i)
        {
            res += binom(n, i, 1.0/b) * fpr_base(num_bits_/b, i, k_);
        }

        return res;
    }

    /*! \brief indicates if this object is a shallow copy
     * \return \c bool
     */
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    bool is_copy() const noexcept
    {
        return is_copy_;
    }

private:
    /*! \brief binomial coefficient
     * \param[in] n
     * \param[in] k
     * \param[in] p
     * \return binomial coefficient
     */
    HOSTQUALIFIER INLINEQUALIFIER
    double binom(
        const index_type n,
        const index_type k,
        const double p) const noexcept
    {
        double res = 1.0;

        for(index_type i = n - k + 1; i <= n; ++i)
        {
            res = res * i;
        }

        for(index_type i = 1; i <= k; ++i)
        {
            res = res / i;
        }

        res = res * pow(p, k) * pow(1.0 - p, n - k);

        return res;
    }

    /*! \brief FPR of traditional bloom filters
     * \param[in] m
     * \param[in] n
     * \param[in] k
     * \return FPR
     */
    HOSTQUALIFIER INLINEQUALIFIER
    double fpr_base(
        const index_type m,
        const index_type n,
        const index_type k) const noexcept
    {
        return std::pow(1.0 - std::pow(1.0 - 1.0 / m, n * k), k);
    }

    slot_type * bloom_filter_; //< pointer to the bit vector
    const index_type num_bits_; //< number of bits (m)
    const index_type num_slots_; //< number of slots
    const index_type num_blocks_; //< number of CG blocks
    const index_type k_; //< number of hash functions
    key_type seed_; //< random seed
    bool is_copy_; //< indicates if this object is a shallow copy

}; // class BloomFilter

} // namespace warpcore

#endif /* WARPCORE_BLOOM_FILTER_CUH */