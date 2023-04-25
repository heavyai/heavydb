#ifndef WARPCORE_COUNTING_HASH_TABLE_CUH
#define WARPCORE_COUNTING_HASH_TABLE_CUH

#include "single_value_hash_table.cuh"

namespace warpcore
{

/*! \brief counting hash table
 * \tparam Key key type (\c std::uint32_t or \c std::uint64_t)
 * \tparam Value value type
 * \tparam EmptyKey key which represents an empty slot
 * \tparam TombstoneKey key which represents an erased slot
 * \tparam ProbingScheme probing scheme from \c warpcore::probing_schemes
 * \tparam TableStorage internal storage class \c warpcore::storage::key_value
 * \tparam TempMemoryBytes size of temporary storage (typically a few kB)
 */
template<
    class Key,
    class Value = index_t,
    Key EmptyKey = defaults::empty_key<Key>(),
    Key TombstoneKey = defaults::tombstone_key<Key>(),
    class ProbingScheme = defaults::probing_scheme_t<Key, 4>,
    class TableStorage = defaults::table_storage_t<Key, Value>,
    index_t TempMemoryBytes = defaults::temp_memory_bytes()>
class CountingHashTable
{
    using base_type = SingleValueHashTable<
        Key, Value, EmptyKey, TombstoneKey, ProbingScheme, TableStorage>;

    static_assert(
        checks::is_valid_counter_type<typename base_type::value_type>(),
        "counter type must be either int, std::uint32_t or std::uint64_t");

public:
    using key_type = typename base_type::key_type;
    using value_type = typename base_type::value_type;
    using index_type = typename base_type::index_type;
    using status_type = typename base_type::status_type;

    /*! \brief get empty key
     * \return empty key
     */
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    static constexpr key_type empty_key() noexcept
    {
        return base_type::empty_key();
    }

    /*! \brief get tombstone key
     * \return tombstone key
     */
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    static constexpr key_type tombstone_key() noexcept
    {
        return base_type::tombstone_key();
    }

    /*! \brief get cooperative group size
     * \return cooperative group size
     */
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    static constexpr index_type cg_size() noexcept
    {
        return base_type::cg_size();
    }

    /*! \brief constructor
     * \param[in] min_capacity minimum number of slots in the hash table
     * \param[in] seed random seed
     * \param[in] max_count count after which to stop counting new occurrences
     * \param[in] no_init whether to initialize the table at construction or not
     */
    HOSTQUALIFIER INLINEQUALIFIER
    explicit CountingHashTable(
        index_type min_capacity,
        key_type seed = defaults::seed<key_type>(),
        value_type max_count = std::numeric_limits<value_type>::max(),
        bool no_init = false) noexcept :
        base_table_(min_capacity, seed, true),
        max_count_(max_count),
        is_copy_(false)
    {
        if(!no_init) init(seed);
    }

    /*! \brief copy-constructor (shallow)
     *  \param[in] object to be copied
     */
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    CountingHashTable(const CountingHashTable& o) noexcept :
        base_table_(o.base_table_),
        max_count_(o.max_count_),
        is_copy_(true)
    {}

    /*! \brief move-constructor
     *  \param[in] object to be moved
     */
    HOSTQUALIFIER INLINEQUALIFIER
    CountingHashTable(CountingHashTable&& o) noexcept :
        base_table_(std::move(o.base_table_)),
        max_count_(std::move(o.max_count_)),
        is_copy_(std::move(o.is_copy_))
    {
        o.is_copy_ = true;
    }

    /*! \brief (re)initialize the hash table
     * \param[in] seed random seed
     * \param[in] stream CUDA stream in which this operation is executed in
     */
    HOSTQUALIFIER INLINEQUALIFIER
    void init(
        const key_type seed,
        const cudaStream_t stream = 0) noexcept
    {
        base_table_.init(seed, stream);

        if(base_table_.is_initialized())
        {
            base_table_.table_.init_values(0, stream);
        }
    }

    /*! \brief (re)initialize the hash table
     * \param[in] stream CUDA stream in which this operation is executed in
     */
    HOSTQUALIFIER INLINEQUALIFIER
    void init(const cudaStream_t stream = 0) noexcept
    {
        init(base_table_.seed(), stream);
    }

    /*! \brief inserts a key into the hash table
     * \param[in] key_in key to insert into the hash table
     * \param[in] group cooperative group
     * \param[in] probing_length maximum number of probing attempts
     * \return status (per thread)
     */
    DEVICEQUALIFIER INLINEQUALIFIER
    status_type insert(
        key_type key_in,
        const cg::thread_block_tile<cg_size()>& group,
        index_type probing_length = defaults::probing_length()) noexcept
    {
        status_type status = status_type::unknown_error();

        value_type * value_ptr =
            base_table_.insert_impl(key_in, status, group, probing_length);

        if(group.thread_rank() == 0 && value_ptr != nullptr)
        {
            // this may not exactly stop counting when max_count is reached
            // however, this is resolved during retireval
            if(*value_ptr < max_count_)
            {
                const value_type old = atomicAdd(value_ptr, 1);

                // guard wrap-around
                if(old > *value_ptr)
                {
                    status += status_type::index_overflow();
                    base_table_.status_->atomic_join(
                        status_type::index_overflow());
                    atomicExch(value_ptr, max_count_);
                }
            }
        }

        return status;
    }

    /*! \brief insert a set of keys into the hash table
     * \tparam StatusHandler handles status per key (see \c status_handlers)
     * \param[in] keys_in pointer to keys to insert into the hash table
     * \param[in] num_in number of keys to insert
     * \param[in] stream CUDA stream in which this operation is executed in
     * \param[in] probing_length maximum number of probing attempts
     * \param[out] status_out status information per key
     */
    template<class StatusHandler = defaults::status_handler_t>
    HOSTQUALIFIER INLINEQUALIFIER
    void insert(
        const key_type * keys_in,
        index_type num_in,
        cudaStream_t stream = 0,
        index_type probing_length = defaults::probing_length(),
        typename StatusHandler::base_type * status_out = nullptr) noexcept
    {
        kernels::insert<CountingHashTable, StatusHandler>
        <<<SDIV(num_in * cg_size(), WARPCORE_BLOCKSIZE), WARPCORE_BLOCKSIZE, 0, stream>>>
        (keys_in, num_in, *this, probing_length, status_out);
    }

    /*! \brief retrieves a key from the hash table
     * \param[in] key_in key to retrieve from the hash table
     * \param[out] value_out count of \c key_in
     * \param[in] group cooperative group
     * \param[in] probing_length maximum number of probing attempts
     * \return status (per thread)
     */
    DEVICEQUALIFIER INLINEQUALIFIER
    status_type retrieve(
        key_type key_in,
        value_type& value_out,
        const cg::thread_block_tile<cg_size()>& group,
        index_type probing_length = defaults::probing_length()) const noexcept
    {
        value_type out;

        const status_type status =
            base_table_.retrieve(key_in, out, group, probing_length);

        if(status.has_any())
        {
            value_out = 0;
        }
        else
        {
            value_out = (out > max_count_) ? max_count_ : out;
        }

        return status;
    }

    /*! \brief retrieve a set of keys from the hash table
     * \tparam StatusHandler handles status per key (see \c status_handlers)
     * \param[in] keys_in pointer to keys to retrieve from the hash table
     * \param[in] num_in number of keys to retrieve
     * \param[out] values_out corresponding counts of keys in \c key_in
     * \param[in] stream CUDA stream in which this operation is executed in
     * \param[in] probing_length maximum number of probing attempts
     * \param[out] status_out status information (per key)
     */
    template<class StatusHandler = defaults::status_handler_t>
    HOSTQUALIFIER INLINEQUALIFIER
    void retrieve(
        const key_type * keys_in,
        index_type num_in,
        value_type * values_out,
        cudaStream_t stream = 0,
        index_type probing_length = defaults::probing_length(),
        typename StatusHandler::base_type * status_out = nullptr) const noexcept
    {
        kernels::retrieve<CountingHashTable, StatusHandler>
        <<<SDIV(num_in * cg_size(), WARPCORE_BLOCKSIZE), WARPCORE_BLOCKSIZE, 0, stream>>>
        (keys_in, num_in, values_out, *this, probing_length, status_out);
    }

    /*! \brief retrieves all elements from the hash table
     * \param[out] keys_out location to store all retrieved keys
     * \param[out] values_out location to store all retrieved counts
     * \param[out] num_out number of of key/value pairs retrieved
     * \param[in] stream CUDA stream in which this operation is executed in
     */
    HOSTQUALIFIER INLINEQUALIFIER
    void retrieve_all(
        key_type * keys_out,
        value_type * values_out,
        index_t& num_out,
        cudaStream_t stream = 0) const noexcept
    {
        base_table_.retrieve_all(keys_out, values_out, num_out, stream);
    }

    /*! \brief erases a key from the hash table
     * \param[in] key_in key to erase from the hash table
     * \param[in] group cooperative group
     * \param[in] probing_length maximum number of probing attempts
     * \return status (per thread)
     */
    DEVICEQUALIFIER INLINEQUALIFIER
    status_type erase(
        key_type key_in,
        const cg::thread_block_tile<cg_size()>& group,
        index_type probing_length = defaults::probing_length()) noexcept
    {
        return base_table_.erase(key_in, group, probing_length);
    }

    /*! \brief erases a set of keys from the hash table
     * \tparam StatusHandler handles status per key (see \c status_handlers)
     * \param[in] keys_in pointer to keys to erase from the hash table
     * \param[in] num_in number of keys to erase
     * \param[in] stream CUDA stream in which this operation is executed in
     * \param[in] probing_length maximum number of probing attempts
     * \param[out] status_out status information (per key)
     */
    template<class StatusHandler = defaults::status_handler_t>
    HOSTQUALIFIER INLINEQUALIFIER
    void erase(
        key_type * keys_in,
        index_type num_in,
        cudaStream_t stream = 0,
        index_type probing_length = defaults::probing_length(),
        typename StatusHandler::base_type * status_out = nullptr) noexcept
    {
        base_table_.template erase<StatusHandler>(
            keys_in, num_in, probing_length, status_out);
    }

    /*! \brief applies a funtion on all key value pairs inside the table
     * \tparam Func type of map i.e. CUDA device lambda
     * \param[in] f map to apply
     * \param[in] stream CUDA stream in which this operation is executed in
     * \param[in] size of shared memory to reserve for this execution
     */
    template<class Func>
    HOSTQUALIFIER INLINEQUALIFIER
    void for_each(
        Func f,
        cudaStream_t stream = 0,
        index_type smem_bytes = 0) const noexcept
    {
        base_table_.template for_each<Func>(f, stream, smem_bytes);
    }

    /*! \brief number of key/value pairs stored inside the hash table
     * \param stream CUDA stream in which this operation is executed in
     * \return the number of key/value pairs inside the hash table
     */
    HOSTQUALIFIER INLINEQUALIFIER
    index_type size(cudaStream_t stream = 0) const noexcept
    {
        return base_table_.size(stream);
    }

    /*! \brief current load factor of the hash table
     * \param stream CUDA stream in which this operation is executed in
     * \return load factor
     */
    HOSTQUALIFIER INLINEQUALIFIER
    float load_factor(cudaStream_t stream = 0) const noexcept
    {
        return base_table_.load_factor(stream);
    }

    /*! \brief current storage density of the hash table
     * \param stream CUDA stream in which this operation is executed in
     * \return storage density
     */
    HOSTQUALIFIER INLINEQUALIFIER
    float storage_density(cudaStream_t stream = 0) const noexcept
    {
        return base_table_.storage_density(stream);
    }

    /*! \brief get the capacity of the hash table
     * \return number of slots in the hash table
     */
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    index_type capacity() const noexcept
    {
        return base_table_.capacity();
    }

    /*! \brief get the status of the hash table
     * \param stream CUDA stream in which this operation is executed in
     * \return the status
     */
    HOSTQUALIFIER INLINEQUALIFIER
    status_type peek_status(cudaStream_t stream = 0) const noexcept
    {
        return base_table_.peek_status(stream);
    }

    /*! \brief get and reset the status of the hash table
     * \param stream CUDA stream in which this operation is executed in
     * \return the status
     */
    HOSTQUALIFIER INLINEQUALIFIER
    status_type pop_status(cudaStream_t stream = 0) noexcept
    {
        return base_table_.pop_status(stream);
    }

    /*! \brief checks if \c key is equal to \c EmptyKey
     * \return \c bool
     */
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    static constexpr bool is_empty_key(key_type key) noexcept
    {
        return base_type::is_empty_key(key);
    }

    /*! \brief checks if \c key is equal to \c TombstoneKey
     * \return \c bool
     */
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    static constexpr bool is_tombstone_key(key_type key) noexcept
    {
        return base_type::is_tombstone_key(key);
    }

    /*! \brief checks if \c key is equal to \c (EmptyKey||TombstoneKey)
     * \return \c bool
     */
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    static constexpr bool is_valid_key(key_type key) noexcept
    {
        return base_type::is_valid_key(key);
    }

    /*! \brief get random seed
     * \return seed
     */
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    key_type seed() const noexcept
    {
        return base_table_.seed();
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
    base_type base_table_; //< base type aka SingleValueHashTable
    const value_type max_count_; //< count after which new occurrences are ignored
    bool is_copy_; //< indicates if this object is a shallow copy

}; // class CountingHashTable

} // namespace warpcore

#endif /* WARPCORE_COUNTING_HASH_TABLE_CUH */