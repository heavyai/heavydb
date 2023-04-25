#ifndef WARPCORE_HASH_SET_CUH
#define WARPCORE_HASH_SET_CUH

#include "base.cuh"

namespace warpcore
{

/*! \brief hash set
 * \tparam Key key type (\c std::uint32_t or \c std::uint64_t)
 * \tparam EmptyKey key which represents an empty slot
 * \tparam TombstoneKey key which represents an erased slot
 * \tparam ProbingScheme probing scheme from \c warpcore::probing_schemes
 * \tparam TempMemoryBytes size of temporary storage (typically a few kB)
 */
template<
    class Key,
    Key EmptyKey = defaults::empty_key<Key>(),
    Key TombstoneKey = defaults::tombstone_key<Key>(),
    class ProbingScheme = defaults::probing_scheme_t<Key, 16>,
    index_t TempMemoryBytes = defaults::temp_memory_bytes()>
class HashSet
{
    static_assert(
        checks::is_valid_key_type<Key>(),
        "invalid key type");

    static_assert(
        EmptyKey != TombstoneKey,
        "empty key and tombstone key must not be identical");

    static_assert(
        checks::is_probing_scheme<ProbingScheme>(),
        "not a valid probing scheme type");

    static_assert(
        std::is_same<typename ProbingScheme::key_type, Key>::value,
        "probing key type differs from table's key type");

    static_assert(
        TempMemoryBytes >= sizeof(index_t),
        "temporary storage must at least be of size index_type");

public:
    using key_type = Key;
    using index_type = index_t;
    using status_type = Status;

    /*! \brief get empty key
     * \return empty key
     */
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    static constexpr key_type empty_key() noexcept
    {
        return EmptyKey;
    }

    /*! \brief get tombstone key
     * \return tombstone key
     */
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    static constexpr key_type tombstone_key() noexcept
    {
        return TombstoneKey;
    }

    /*! \brief get cooperative group size
     * \return cooperative group size
     */
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    static constexpr index_type cg_size() noexcept
    {
        return ProbingScheme::cg_size();
    }

    /*! \brief constructor
     * \param[in] capacity maximum cardinality of the set
     * \param[in] seed random seed
     */
    HOSTQUALIFIER INLINEQUALIFIER
    explicit HashSet(
        index_type min_capacity,
        key_type seed = defaults::seed<key_type>()) noexcept :
        status_(nullptr),
        keys_(nullptr),
        capacity_(detail::get_valid_capacity(min_capacity, cg_size())),
        temp_(TempMemoryBytes / sizeof(index_type)),
        seed_(seed),
        is_initialized_(false),
        is_copy_(false)
    {
        assign_status(Status::not_initialized());

        const auto total_bytes = (sizeof(key_type) * capacity()) + sizeof(Status);

        if(helpers::available_gpu_memory() >= total_bytes)
        {
            cudaMalloc(&keys_, sizeof(key_type) * capacity_);
            cudaMalloc(&status_, sizeof(Status));

            assign_status(Status::none());
            is_initialized_ = true;

            init();
        }
        else
        {
            join_status(Status::out_of_memory());
        }
    }

    /*! \brief copy-constructor (shallow)
     *  \param[in] object to be copied
     */
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    HashSet(const HashSet& o) noexcept :
        status_(o.status_),
        keys_(o.keys_),
        capacity_(o.capacity_),
        temp_(o.temp_),
        seed_(o.seed_),
        is_initialized_(o.is_initialized_),
        is_copy_(true)
    {}

    /*! \brief move-constructor
     *  \param[in] object to be moved
     */
    HOSTQUALIFIER INLINEQUALIFIER
    HashSet(HashSet&& o) noexcept :
        status_(std::move(o.status_)),
        keys_(std::move(o.keys_)),
        capacity_(std::move(o.capacity_)),
        temp_(std::move(o.temp_)),
        seed_(std::move(o.seed_)),
        is_initialized_(std::move(o.is_initialized_)),
        is_copy_(std::move(o.is_copy_))
    {
        o.is_copy_ = true;
    }

    #ifndef __CUDA_ARCH__
    /*! \brief destructor
     */
    HOSTQUALIFIER INLINEQUALIFIER
    ~HashSet() noexcept
    {
        if(!is_copy_)
        {
            if(keys_   != nullptr) cudaFree(keys_);
            if(status_ != nullptr) cudaFree(status_);
        }
    }
    #endif

    /*! \brief (re)initialize the hash set
     * \param[in] seed random seed
     * \param[in] stream CUDA stream in which this operation is executed in
     */
    HOSTQUALIFIER INLINEQUALIFIER
    void init(
        const key_type seed,
        const cudaStream_t stream = 0) noexcept
    {
        seed_ = seed;

        if(is_initialized_)
        {
            kernels::memset<key_type, empty_key()>
            <<<SDIV(capacity_, WARPCORE_BLOCKSIZE), WARPCORE_BLOCKSIZE, 0, stream>>>
            (keys_, capacity_);

            assign_status(Status::none(), stream);
        }
    }

    /*! \brief (re)initialize the hash set
     * \param[in] stream CUDA stream in which this operation is executed in
     */
    HOSTQUALIFIER INLINEQUALIFIER
    void init(const cudaStream_t stream = 0) noexcept
    {
        init(seed_, stream);
    }

    /*! \brief inserts a key into the hash set
     * \param[in] key_in key to insert into the hash set
     * \param[in] group cooperative group
     * \param[in] probing_length maximum number of probing attempts
     * \return status (per thread)
     */
    DEVICEQUALIFIER INLINEQUALIFIER
    Status insert(
        key_type key_in,
        const cg::thread_block_tile<cg_size()>& group,
        index_type probing_length = defaults::probing_length()) noexcept
    {
        if(!is_initialized_) return Status::not_initialized();

        if(!is_valid_key(key_in))
        {
            status_->atomic_join(Status::invalid_key());
            return Status::invalid_key();
        }

        ProbingScheme iter(capacity_, probing_length, group);

        for(index_type i = iter.begin(key_in, seed_); i != iter.end(); i = iter.next())
        {
            key_type table_key = keys_[i];
            const bool hit = (table_key == key_in);
            const auto hit_mask = group.ballot(hit);

            if(hit_mask)
            {
                status_->atomic_join(Status::duplicate_key());
                return Status::duplicate_key();
            }

            // !not_is_valid_key?
            auto empty_mask = group.ballot(is_empty_key(table_key));

            bool success = false;
            bool duplicate = false;

            while(empty_mask)
            {
                const auto leader = ffs(empty_mask) - 1;

                if(group.thread_rank() == leader)
                {
                    const auto old = atomicCAS(keys_ + i, table_key, key_in);
                    success = (old == table_key);
                    duplicate = (old == key_in);
                }

                if(group.any(duplicate))
                {
                    status_->atomic_join(Status::duplicate_key());
                    return Status::duplicate_key();
                }

                if(group.any(success))
                {
                    return Status::none();
                }

                empty_mask ^= 1UL << leader;
            }
        }

        status_->atomic_join(Status::probing_length_exceeded());
        return Status::probing_length_exceeded();
    }

    /*! \brief insert a set of keys into the hash set
     * \tparam StatusHandler handles returned status per key (see \c status_handlers)
     * \param[in] keys_in pointer to keys to insert into the hash set
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
        static_assert(
            checks::is_status_handler<StatusHandler>(),
            "not a valid status handler type");

        if(!is_initialized_) return;

        kernels::insert<HashSet, StatusHandler>
        <<<SDIV(num_in * cg_size(), WARPCORE_BLOCKSIZE), WARPCORE_BLOCKSIZE, 0, stream>>>
        (keys_in, num_in, *this, probing_length, status_out);
    }

    /*! \brief retrieves a key from the hash set
     * \param[in] key_in key to retrieve from the hash set
     * \param[out] flag_out \c true iff \c key_in is member of the set
     * \param[in] group cooperative group
     * \param[in] probing_length maximum number of probing attempts
     * \return status (per thread)
     */
    DEVICEQUALIFIER INLINEQUALIFIER
    Status retrieve(
        Key key_in,
        bool& flag_out,
        const cg::thread_block_tile<cg_size()>& group,
        index_type probing_length = defaults::probing_length()) const noexcept
    {
        if(!is_initialized_) return Status::not_initialized();

        if(!is_valid_key(key_in))
        {
            status_->atomic_join(Status::invalid_key());
            return Status::invalid_key();
        }

        ProbingScheme iter(capacity_, probing_length, group);

        for(index_type i = iter.begin(key_in, seed_); i != iter.end(); i = iter.next())
        {
            key_type table_key = keys_[i];
            const bool hit = (table_key == key_in);
            const auto hit_mask = group.ballot(hit);

            if(hit_mask)
            {
                flag_out = true;
                return Status::none();
            }

            if(group.any(is_empty_key(table_key)))
            {
                flag_out = false;
                return Status::none();
            }
        }

        status_->atomic_join(Status::probing_length_exceeded());
        return Status::probing_length_exceeded();
    }

    /*! \brief retrieve a set of keys from the hash table
     * \tparam StatusHandler handles returned status per key (see \c status_handlers)
     * \param[in] keys_in pointer to keys to retrieve from the hash table
     * \param[in] num_in number of keys to retrieve
     * \param[out] flags_out flags membership of \c keys_in in the set
     * \param[in] stream CUDA stream in which this operation is executed in
     * \param[in] probing_length maximum number of probing attempts
     * \param[out] status_out status information (per key)
     */
    template<class StatusHandler = defaults::status_handler_t>
    HOSTQUALIFIER INLINEQUALIFIER
    void retrieve(
        const key_type * keys_in,
        index_type num_in,
        bool * flags_out,
        cudaStream_t stream = 0,
        index_type probing_length = defaults::probing_length(),
        typename StatusHandler::base_type * status_out = nullptr) const noexcept
    {
        static_assert(
            checks::is_status_handler<StatusHandler>(),
            "not a valid status handler type");

        if(!is_initialized_) return;

        kernels::retrieve<HashSet, StatusHandler>
        <<<SDIV(num_in * cg_size(), WARPCORE_BLOCKSIZE), WARPCORE_BLOCKSIZE, 0, stream>>>
        (keys_in, num_in, flags_out, *this, probing_length, status_out);
    }

    /*! \brief retrieves all elements from the hash set
     * \param[out] keys_out location to store all retrieved keys
     * \param[out] num_out number of of keys retrieved
     * \param[in] stream CUDA stream in which this operation is executed in
     */
    HOSTQUALIFIER INLINEQUALIFIER
    void retrieve_all(
        key_type * keys_out,
        index_type& num_out,
        cudaStream_t stream = 0) const noexcept
    {
        if(!is_initialized_) return;

        index_type * tmp = temp_.get();

        cudaMemsetAsync(tmp, 0, sizeof(index_type), stream);

        for_each([=, *this] DEVICEQUALIFIER (key_type key)
        { keys_out[helpers::atomicAggInc(tmp)] = key; }, stream);

        cudaMemcpyAsync(&num_out, tmp, sizeof(index_type), D2H);

        if(stream == 0)
        {
            cudaStreamSynchronize(stream);
        }
    }

    /*! \brief erases a key from the hash table
     * \param[in] key_in key to erase from the hash table
     * \param[in] group cooperative group
     * \param[in] probing_length maximum number of probing attempts
     * \return status (per thread)
     */
    DEVICEQUALIFIER INLINEQUALIFIER
    Status erase(
        key_type key_in,
        const cg::thread_block_tile<cg_size()>& group,
        index_type probing_length = defaults::probing_length()) noexcept
    {
        if(!is_initialized_) return Status::not_initialized();

        if(!is_valid_key(key_in))
        {
            status_->atomic_join(Status::invalid_key());
            return Status::invalid_key();
        }

        ProbingScheme iter(capacity_, probing_length, group);

        for(index_type i = iter.begin(key_in, seed_); i != iter.end(); i = iter.next())
        {
            key_type table_key = keys_[i];
            bool hit = (table_key == key_in);
            auto hit_mask = group.ballot(hit);

            if(hit_mask)
            {
                auto leader = ffs(hit_mask)-1;

                if(group.thread_rank() == leader)
                {
                    keys_[i] = tombstone_key();
                }

                return Status::none();
            }

            if(group.any(is_empty_key(table_key)))
            {
                return Status::none();
                return Status::key_not_found();
            }
        }

        return Status::none();
        return Status::probing_length_exceeded();
    }

    /*! \brief erases a set of keys from the hash table
     * \tparam StatusHandler handles returned status per key (see \c status_handlers)
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
        if(!is_initialized_) return;

        kernels::erase<HashSet, StatusHandler>
        <<<SDIV(num_in * cg_size(), WARPCORE_BLOCKSIZE), WARPCORE_BLOCKSIZE, 0, stream>>>
        (keys_in, num_in, *this, probing_length, status_out);
    }

    /*! \brief applies a funtion on all keys inside the table
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
        if(!is_initialized_) return;

        helpers::lambda_kernel
        <<<SDIV(capacity(), MAXBLOCKSIZE), MAXBLOCKSIZE, smem_bytes, stream>>>
        ([=, *this] DEVICEQUALIFIER // TODO mutable?
        {
            const index_type tid = helpers::global_thread_id();

            if(tid < capacity())
            {
                const key_type key = keys_[tid];
                if(is_valid_key(key))
                {
                    f(key);
                }
            }
        });
    }

    /*! \brief number of key/value pairs stored inside the hash set
     * \return the number of key/value pairs inside the hash table
     */
    HOSTQUALIFIER INLINEQUALIFIER
    index_type size(cudaStream_t stream = 0) const noexcept
    {
        if(!is_initialized_) return 0;

        index_type out;
        index_type * tmp = temp_.get();

        cudaMemsetAsync(tmp, 0, sizeof(index_type), stream);

        helpers::lambda_kernel
        <<<SDIV(capacity_, WARPCORE_BLOCKSIZE), WARPCORE_BLOCKSIZE, sizeof(index_type), stream>>>
        ([=, *this] DEVICEQUALIFIER
        {
            __shared__ index_type smem;

            const index_type tid = helpers::global_thread_id();
            const auto block = cooperative_groups::this_thread_block();

            if(tid >= capacity()) return;

            const bool empty = !is_valid_key(keys_[tid]);

            if(block.thread_rank() == 0)
            {
                smem = 0;
            }

            block.sync();

            if(!empty)
            {
                auto active_threads = cooperative_groups::coalesced_threads();

                if(active_threads.thread_rank() == 0)
                {
                    atomicAdd(&smem, active_threads.size());
                }
            }

            block.sync();

            if(block.thread_rank() == 0)
            {
                atomicAdd(tmp, smem);
            }
        });

        cudaMemcpyAsync(
            &out,
            tmp,
            sizeof(index_type),
            D2H,
            stream);

        cudaStreamSynchronize(stream);

        return out;
    }

    /*! \brief current load factor of the hash set
     * \param stream CUDA stream in which this operation is executed in
     * \return load factor
     */
    HOSTQUALIFIER INLINEQUALIFIER
    float load_factor(cudaStream_t stream = 0) const noexcept
    {
        return float(size(stream)) / float(capacity());
    }

    /*! \brief current storage density of the hash set
     * \param stream CUDA stream in which this operation is executed in
     * \return storage density
     */
    HOSTQUALIFIER INLINEQUALIFIER
    float storage_density(cudaStream_t stream = 0) const noexcept
    {
        return load_factor(stream);
    }

    /*! \brief get the capacity of the hash table
     * \return number of slots in the hash table
     */
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    index_type capacity() const noexcept
    {
        return capacity_;
    }

    /*! \brief get the total number of bytes occupied by this data structure
     *  \return bytes
     */
    HOSTQUALIFIER INLINEQUALIFIER
    index_type bytes_total() const noexcept
    {
        return capacity_ * sizeof(key_type) + temp_.bytes_total() + sizeof(status_type);
    }

    /*! \brief get the status of the hash table
     * \param stream CUDA stream in which this operation is executed in
     * \return the status
     */
    HOSTQUALIFIER INLINEQUALIFIER
    Status peek_status(cudaStream_t stream = 0) const noexcept
    {
        Status status = Status::not_initialized();

        if(status_ != nullptr)
        {
            cudaMemcpyAsync(
                &status,
                status_,
                sizeof(Status),
                D2H,
                stream);

            cudaStreamSynchronize(stream);
        }

        return status;
    }

    /*! \brief get and reset the status of the hash table
     * \param stream CUDA stream in which this operation is executed in
     * \return the status
     */
    HOSTQUALIFIER INLINEQUALIFIER
    Status pop_status(cudaStream_t stream = 0) noexcept
    {
        Status status = Status::not_initialized();

        if(status_ != nullptr)
        {
            cudaMemcpyAsync(
                &status,
                status_,
                sizeof(Status),
                D2H,
                stream);

            assign_status(Status::none(), stream);

            cudaStreamSynchronize(stream);
        }

        return status;
    }

    /*! \brief checks if \c key is equal to \c EmptyKey
    * \return \c bool
    */
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    static constexpr bool is_empty_key(key_type key) noexcept
    {
        return (key == empty_key());
    }

    /*! \brief checks if \c key is equal to \c TombstoneKey
    * \return \c bool
    */
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    static constexpr bool is_tombstone_key(key_type key) noexcept
    {
        return (key == tombstone_key());
    }

    /*! \brief checks if \c key is equal to \c (EmptyKey||TombstoneKey)
    * \return \c bool
    */
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    static constexpr bool is_valid_key(key_type key) noexcept
    {
        return (key != empty_key() && key != tombstone_key());
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
    /*! \brief assigns the hash set's status
     * \param[in] status new status
     * \param[in] stream CUDA stream in which this operation is executed in
     */
    HOSTQUALIFIER INLINEQUALIFIER
    void assign_status(
        Status status,
        cudaStream_t stream = 0) const noexcept
    {
        if(!is_initialized_) return;

        cudaMemcpyAsync(
            status_,
            &status,
            sizeof(Status),
            H2D,
            stream);

        cudaStreamSynchronize(stream);
    }

    /*! \brief joins additional flags to the hash set's status
     * \param[in] status new status
     * \param[in] stream CUDA stream in which this operation is executed in
     */
    HOSTQUALIFIER INLINEQUALIFIER
    void join_status(
        Status status,
        cudaStream_t stream = 0) const noexcept
    {
        if(!is_initialized_) return;

        const Status joined = status + peek_status(stream);

        cudaMemcpyAsync(
            status_,
            &joined,
            sizeof(Status),
            H2D,
            stream);

        cudaStreamSynchronize(stream);
    }

    Status * status_; //< pointer to status
    key_type * keys_ ; //< pointer to key store
    const index_type capacity_; //< number of slots in the hash table
    storage::CyclicStore<index_type> temp_; //< temporary memory
    key_type seed_; //< random seed
    bool is_initialized_; //< indicates if the set is properly initialized
    bool is_copy_; //< indicates if this object is a shallow copy

}; // class HashSet

} // namespace warpcore

#endif /* WARPCORE_HASH_SET_CUH */