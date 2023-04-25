#ifndef WARPCORE_BUCKET_LIST_HASH_TABLE_CUH
#define WARPCORE_BUCKET_LIST_HASH_TABLE_CUH

#include "single_value_hash_table.cuh"

namespace warpcore
{

/*! \brief bucket list hash table
 * \tparam Key key type (\c std::uint32_t or \c std::uint64_t)
 * \tparam Value value type
 * \tparam EmptyKey key which represents an empty slot
 * \tparam TombstoneKey key which represents an erased slot
 * \tparam ValueStore storage class from \c warpcore::storage::multi_value
 * \tparam ProbingScheme probing scheme from \c warpcore::probing_schemes
 */
template<
    class Key,
    class Value,
    Key   EmptyKey = defaults::empty_key<Key>(),
    Key   TombstoneKey = defaults::tombstone_key<Key>(),
    class ValueStore = defaults::value_storage_t<Value>,
    class ProbingScheme = defaults::probing_scheme_t<Key, 8>>
class BucketListHashTable
{
    static_assert(
        checks::is_value_storage<ValueStore>(),
        "not a valid storage type");

public:
    // TODO why public?
    using handle_type = typename ValueStore::handle_type;

private:
    using hash_table_type = SingleValueHashTable<
        Key,
        handle_type,
        EmptyKey,
        TombstoneKey,
        ProbingScheme>;

    using value_store_type = ValueStore;

public:
    using key_type = Key;
    using value_type = Value;
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

    /*! \brief checks if \c key is equal to \c (EmptyKey||TombstoneKey)
     * \return \c bool
     */
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    static constexpr bool is_valid_key(const key_type key) noexcept
    {
        return (key != empty_key() && key != tombstone_key());
    }

    /*! \brief get cooperative group size
     * \return cooperative group size
     */
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    static constexpr index_type cg_size() noexcept
    {
        return hash_table_type::cg_size();
    }

    /*! \brief maximum bucket size
     * \return size
     */
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    static constexpr index_type max_bucket_size() noexcept
    {
        return handle_type::max_bucket_size();
    }

    /*! \brief constructor
    * \param[in] key_capacity guaranteed number of key slots in the hash table
    * \param[in] value_capacity total number of value slots
    * \param[in] seed random seed
    * \param[in] grow_factor bucket grow factor for \c warpcore::storage::multi_value::BucketListStore
    * \param[in] min_bucket_size initial size of value buckets for \c warpcore::storage::multi_value::BucketListStore
    * \param[in] max_bucket_size bucket size of \c warpcore::storage::multi_value::BucketListStore after which no more growth occurs
    * \param[in] max_values_per_key maximum number of values to store per key
    */
    HOSTQUALIFIER
    explicit BucketListHashTable(
        const index_type key_capacity,
        const index_type value_capacity,
        const key_type seed = defaults::seed<key_type>(),
        const float grow_factor = 1.1,
        const index_type min_bucket_size = 1,
        const index_type max_bucket_size = max_bucket_size(),
        const index_type max_values_per_key = handle_type::max_value_count(),
        const bool no_init = false) noexcept :
        hash_table_(key_capacity, seed, true),
        value_store_(value_capacity, grow_factor, min_bucket_size, max_bucket_size),
        max_values_per_key_(std::min(max_values_per_key, handle_type::max_value_count())),
        is_copy_(false)
    {
        join_status(value_store_.status());

        if(!no_init) init(seed);
    }

    /*! \brief copy-constructor (shallow)
     *  \param[in] object to be copied
     */
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    BucketListHashTable(const BucketListHashTable& o) noexcept :
        hash_table_(o.hash_table_),
        value_store_(o.value_store_),
        max_values_per_key_(o.max_values_per_key_),
        is_copy_(true)
    {}

    /*! \brief move-constructor
     *  \param[in] object to be moved
     */
    HOSTQUALIFIER INLINEQUALIFIER
    BucketListHashTable(BucketListHashTable&& o) noexcept :
        hash_table_(std::move(o.hash_table_)),
        value_store_(std::move(o.value_store_)),
        max_values_per_key_(std::move(o.max_values_per_key_)),
        is_copy_(std::move(o.is_copy_))
    {
        o.is_copy_ = true;
    }

    /*! \brief (re)initialize the hash table
     * \param seed random seed
     * \param stream CUDA stream in which this operation is executed
     */
    HOSTQUALIFIER INLINEQUALIFIER
    void init(
        const key_type seed,
        const cudaStream_t stream = 0) noexcept
    {
        const auto status = hash_table_.peek_status(stream);

        if(!status.has_not_initialized())
        {
            hash_table_.init(seed, stream);
            value_store_.init(stream);
            hash_table_.table_.init_values(
                ValueStore::uninitialized_handle(), stream);
        }
    }

    /*! \brief (re)initialize the hash table
     * \param stream CUDA stream in which this operation is executed
     */
    HOSTQUALIFIER INLINEQUALIFIER
    void init(const cudaStream_t stream = 0) noexcept
    {
        init(hash_table_.seed(), stream);
    }

    /*! \brief inserts a key/value pair into the hash table
     * \param[in] key_in key to insert into the hash table
     * \param[in] value_in value that corresponds to \c key_in
     * \param[in] group cooperative group
     * \param[in] probing_length maximum number of probing attempts
     * \return status (per thread)
     */
    DEVICEQUALIFIER INLINEQUALIFIER
    status_type insert(
        const key_type key_in,
        const value_type& value_in,
        const cg::thread_block_tile<cg_size()>& group,
        const index_type probing_length = defaults::probing_length()) noexcept
    {
        status_type status = status_type::unknown_error();

        handle_type * handle_ptr =
            hash_table_.insert_impl(key_in, status, group, probing_length);

        if(handle_ptr != nullptr)
        {
            if(handle_ptr->value_count() >= max_values_per_key_)
            {
                device_join_status(status_type::max_values_for_key_reached());

                return status + status_type::max_values_for_key_reached();
            }
            else
            {
                status_type append_status = Status::unknown_error();

                if(group.thread_rank() == 0)
                {
                    append_status = value_store_.append(*handle_ptr, value_in, max_values_per_key_);

                    if(append_status.has_any())
                    {
                        device_join_status(append_status);
                    }
                }

                status += append_status.group_shuffle(group, 0);
            }
        }

        return status;
    }

    /*! \brief insert a set of keys into the hash table
     * \tparam StatusHandler handles returned status per key (see \c status_handlers)
     * \param[in] keys_in pointer to keys to insert into the hash table
     * \param[in] values_in corresponds values to \c keys_in
     * \param[in] num_in number of keys to insert
     * \param[in] stream CUDA stream in which this operation is executed in
     * \param[in] probing_length maximum number of probing attempts
     * \param[out] status_out status information per key
     */
    template<class StatusHandler = defaults::status_handler_t>
    HOSTQUALIFIER INLINEQUALIFIER
    void insert(
        const key_type * const keys_in,
        const value_type * const values_in,
        const index_type num_in,
        const cudaStream_t stream = 0,
        const index_type probing_length = defaults::probing_length(),
        typename StatusHandler::base_type * const status_out = nullptr) noexcept
    {
        static_assert(
            checks::is_status_handler<StatusHandler>(),
            "not a valid status handler type");

        if(!hash_table_.is_initialized_) return;

        static constexpr index_type block_size = 1024;
        static constexpr index_type groups_per_block = block_size / cg_size();
        static constexpr index_type smem_status_size =
            std::is_same<StatusHandler, status_handlers::ReturnNothing>::value ?
            1 : groups_per_block;

        helpers::lambda_kernel
        <<<SDIV(num_in * cg_size(), block_size), block_size, 0, stream>>>
        ([=, *this] DEVICEQUALIFIER () mutable
        {
            const index_type  tid = helpers::global_thread_id();
            const index_type btid = threadIdx.x;
            const index_type  gid = tid / cg_size();
            const index_type bgid = gid % groups_per_block;
            const auto block = cg::this_thread_block();
            const auto group = cg::tiled_partition<cg_size()>(block);

            __shared__ handle_type * handles[groups_per_block];
            __shared__ status_type status[smem_status_size];

            if(gid < num_in)
            {
                status_type probing_status = status_type::unknown_error();

                handles[bgid] = hash_table_.insert_impl(
                    keys_in[gid],
                    probing_status,
                    group,
                    probing_length);

                if(!std::is_same<
                    StatusHandler,
                    status_handlers::ReturnNothing>::value &&
                    group.thread_rank() == 0)
                {
                    status[bgid] = probing_status;
                }

                block.sync();

                if(btid < groups_per_block && handles[btid] != nullptr)
                {
                    status_type append_status;

                    const index_type block_offset =
                            blockIdx.x * groups_per_block;

                    if(value_store_.size(*(handles[btid])) >= max_values_per_key_)
                    {
                        append_status = status_type::max_values_for_key_reached();
                    }
                    else
                    {
                        if(block_offset + btid < num_in){
                            append_status = value_store_.append(
                                *(handles[btid]),
                                values_in[block_offset + btid],
                                max_values_per_key_);
                        }
                    }

                    if(append_status.has_any())
                    {
                        device_join_status(append_status);
                    }

                    if(block_offset + btid < num_in){

                        // TODO not zero-cost
                        if(!std::is_same<
                            StatusHandler,
                            status_handlers::ReturnNothing>::value)
                        {
                            StatusHandler::handle(
                                status[btid]+append_status,
                                status_out,
                                block_offset + btid);
                        }

                    }
                }
            }

        });

        if(stream == 0)
        {
            cudaStreamSynchronize(stream);
        }
    }

    /*! \brief retrieves a key from the hash table
     * \param[in] key_in key to retrieve from the hash table
     * \param[out] values_out pointer to storage fo the retrieved values
     * \param[out] num_out number of values retrieved
     * \param[in] group cooperative group
     * \param[in] probing_length maximum number of probing attempts
     * \return status (per thread)
     */
    DEVICEQUALIFIER INLINEQUALIFIER
    status_type retrieve(
        const key_type key_in,
        value_type * const values_out,
        index_type& num_out,
        const cg::thread_block_tile<cg_size()>& group,
        const index_type probing_length = defaults::probing_length()) const noexcept
    {
        handle_type handle;

        status_type status =
            hash_table_.retrieve(key_in, handle, group, probing_length);

        if(!status.has_any())
        {
            value_store_.for_each(
                [=] DEVICEQUALIFIER (
                    const value_type& value,
                    index_type offset)
                {
                    values_out[offset] = value;
                },
                handle,
                group);

            num_out = value_store_.size(handle);
        }
        else
        {
            num_out = 0;
        }

        return status;
    }

     /*! \brief retrieve a set of keys from the hash table
     * \note this method has a dry-run mode where it only calculates the needed array sizes in case no memory (aka \c nullptr ) is provided
     * \note \c end_offsets_out can be \c begin_offsets_out+1
     * \tparam StatusHandler handles returned status per key (see \c status_handlers)
     * \param[in] keys_in pointer to keys to retrieve from the hash table
     * \param[in] num_in number of keys to retrieve
     * \param[out] begin_offsets_out
     * \param[out] end_offsets_out
     * \param[out] values_out retrieved values of keys in \c key_in
     * \param[out] num_out total number of values retrieved by this operation
     * \param[in] stream CUDA stream in which this operation is executed in
     * \param[in] probing_length maximum number of probing attempts
     * \param[out] status_out status information (per key)
     */
    template<class StatusHandler = defaults::status_handler_t>
    HOSTQUALIFIER INLINEQUALIFIER
    void retrieve(
        const key_type * const keys_in,
        const index_type num_in,
        index_type * const begin_offsets_out,
        index_type * const end_offsets_out,
        value_type * const values_out,
        index_type& num_out,
        const cudaStream_t stream = 0,
        const index_type probing_length = defaults::probing_length(),
        typename StatusHandler::base_type * const status_out = nullptr) const noexcept
    {
        static_assert(
            checks::is_status_handler<StatusHandler>(),
            "not a valid status handler type");

        if(!hash_table_.is_initialized_) return;

        // cub::DeviceScan::InclusiveSum takes input sizes of type int
        if(num_in > std::numeric_limits<int>::max())
        {
            join_status(status_type::index_overflow(), stream);

            return;
        }

        num_values(
            keys_in,
            num_in,
            num_out,
            end_offsets_out,
            stream,
            probing_length);

        if(values_out != nullptr)
        {
            index_type temp_bytes = num_out * sizeof(value_type);

            cub::DeviceScan::InclusiveSum(
                values_out,
                temp_bytes,
                end_offsets_out,
                end_offsets_out,
                num_in,
                stream);

            cudaMemsetAsync(begin_offsets_out, 0, sizeof(index_type), stream);

            if(end_offsets_out != begin_offsets_out + 1)
            {
                cudaMemcpyAsync(
                    begin_offsets_out + 1,
                    end_offsets_out,
                    sizeof(index_type) * (num_in - 1),
                    D2D,
                    stream);
            }

            kernels::retrieve<BucketListHashTable, StatusHandler>
            <<<SDIV(num_in * cg_size(), WARPCORE_BLOCKSIZE), WARPCORE_BLOCKSIZE, 0, stream>>>
            (
                keys_in,
                num_in,
                begin_offsets_out,
                end_offsets_out,
                values_out,
                *this,
                probing_length,
                status_out);
        }
        else
        {
            if(status_out != nullptr)
            {
                helpers::lambda_kernel
                <<<SDIV(num_in, WARPCORE_BLOCKSIZE), WARPCORE_BLOCKSIZE, 0, stream>>>
                ([=, *this] DEVICEQUALIFIER
                {
                    const index_type tid = helpers::global_thread_id();

                    if(tid < num_in)
                    {
                        StatusHandler::handle(Status::dry_run(), status_out, tid);
                    }
                });
            }

            join_status(status_type::dry_run(), stream);
        }

        if(stream == 0)
        {
            cudaStreamSynchronize(stream);
        }
    }

    // TODO host retrieve which also returns the set of unique keys

    /*! \brief applies a funtion over all values of a corresponding key
     * \tparam Func type of map i.e. CUDA device lambda
     * \param[in] f map to apply
     * \param[in] key_in key to retrieve
     * \param[in] stream CUDA stream in which this operation is executed in
     */
    template<class Func>
    DEVICEQUALIFIER INLINEQUALIFIER
    status_type for_each(
        Func f, // TODO const?
        const key_type key_in,
        const cg::thread_block_tile<cg_size()>& group,
        const index_type probing_length = defaults::probing_length()) const noexcept
    {
        handle_type handle;

        status_type status =
            hash_table_.retrieve(key_in, handle, group, probing_length);

        if(!status.has_any())
        {
            value_store_.for_each(f, handle, group);
        }

        return status;
    }

    // TODO host functions for_each
    // TODO get_key_set

        /*! \brief retrieves all elements from the hash table
     * \info this method has a dry-run mode where it only calculates the needed array sizes in case no memory (aka \c nullptr ) is provided
     * \info this method implements a multi-stage dry-run mode
     * \param[out] keys_out pointer to the set of unique keys
     * \param[out] num_keys_out number of unique keys
     * \param[out] begin_offsets_out begin of value range for a corresponding key in \c values_out
     * \param[out] end_offsets_out end of value range for a corresponding key in \c values_out
     * \param[out] values_out array which holds all retrieved values
     * \param[out] num_values_out total number of values retrieved by this operation
     * \param[in] stream CUDA stream in which this operation is executed in
     */
    HOSTQUALIFIER INLINEQUALIFIER
    void retrieve_all(
        key_type * const keys_out,
        index_type& num_keys_out,
        index_type * const begin_offsets_out,
        index_type * const end_offsets_out,
        value_type * const values_out,
        value_type& num_values_out,
        const cudaStream_t stream = 0) const noexcept
    {
        retrieve_all_keys(keys_out, num_keys_out, stream);

        if(keys_out != nullptr)
        {
            retrieve(
                keys_out,
                num_keys_out,
                begin_offsets_out,
                end_offsets_out,
                values_out,
                num_values_out,
                stream);
        }

        if(stream == 0)
        {
            cudaStreamSynchronize(stream);
        }
    }

    /*! \brief retrieves the set of all keys stored inside the hash table
     * \param[out] keys_out pointer to the retrieved keys
     * \param[out] num_out number of retrieved keys
     * \param[in] stream CUDA stream in which this operation is executed in
     * \note if \c keys_out==nullptr then only \c num_out will be computed
     */
    HOSTQUALIFIER INLINEQUALIFIER
    void retrieve_all_keys(
        key_type * const keys_out,
        index_type& num_out,
        const cudaStream_t stream = 0) const noexcept
    {
        if(!hash_table_.is_initialized_) return;

        if(keys_out == nullptr)
        {
            num_out = hash_table_.size(stream);
        }
        else
        {
            index_type * key_count = hash_table_.temp_.get();
            cudaMemsetAsync(key_count, 0, sizeof(index_type), stream);

            hash_table_.for_each(
            [=] DEVICEQUALIFIER (key_type key, const auto&)
            {
                keys_out[helpers::atomicAggInc(key_count)] = key;
            }, stream);

            cudaMemcpyAsync(
                &num_out, key_count, sizeof(index_type), D2H, stream);
        }

        if(stream == 0 || keys_out == nullptr)
        {
            cudaStreamSynchronize(stream);
        }
    }

    /*! \brief get load factor of the key store
     * \param stream CUDA stream in which this operation is executed in
     * \return load factor
     */
    HOSTQUALIFIER INLINEQUALIFIER
    float key_load_factor(const cudaStream_t stream = 0) const noexcept
    {
        return hash_table_.load_factor(stream);
    }

    /*! \brief get load factor of the value store
     * \param stream CUDA stream in which this operation is executed in
     * \return load factor
     */
    HOSTQUALIFIER INLINEQUALIFIER
    float value_load_factor(const cudaStream_t stream = 0) const noexcept
    {
        return value_store_.load_factor(stream);
    }

    /*! \brief get the the total number of bytes occupied by this data structure
     * \return bytes
     */
    HOSTQUALIFIER INLINEQUALIFIER
    index_type bytes_total() const noexcept
    {
        return hash_table_.bytes_total() + value_store_.bytes_total();
    }

    /*! \brief get the the number of bytes in this data structure occupied by keys
     * \param stream CUDA stream in which this operation is executed in
     * \return bytes
     */
    HOSTQUALIFIER INLINEQUALIFIER
    index_type bytes_keys(const cudaStream_t stream = 0) const noexcept
    {
        return num_keys(stream) * sizeof(key_type);
    }

    /*! \brief get the the number of bytes in this data structure occupied by values
     * \param stream CUDA stream in which this operation is executed in
     * \return bytes
     */
    HOSTQUALIFIER INLINEQUALIFIER
    index_type bytes_values(const cudaStream_t stream = 0) const noexcept
    {
        return num_values(stream) * sizeof(value_type);
    }

    /*! \brief get the the number of bytes in this data structure occupied by actual information
     * \param stream CUDA stream in which this operation is executed in
     * \return bytes
     */
    HOSTQUALIFIER INLINEQUALIFIER
    index_type bytes_payload(const cudaStream_t stream = 0) const noexcept
    {
        return bytes_keys(stream) + bytes_values(stream);
    }

    /*! \brief current storage density of the hash table
     * \param stream CUDA stream in which this operation is executed in
     * \return storage density
     */
    HOSTQUALIFIER INLINEQUALIFIER
    float storage_density(const cudaStream_t stream = 0) const noexcept
    {
        return float(bytes_payload(stream)) / float(bytes_total());
    }

    /*! \brief current relative storage density of the hash table
     * \param stream CUDA stream in which this operation is executed in
     * \return storage density
     */
    HOSTQUALIFIER INLINEQUALIFIER
    float relative_storage_density(const cudaStream_t stream = 0) const noexcept
    {
        const float bytes_hash_table =
            hash_table_.capacity() * (sizeof(key_type) + sizeof(handle_type));
        const float bytes_value_store =
            value_store_.bytes_occupied(stream);

        return float(bytes_payload(stream)) / (bytes_value_store + bytes_hash_table);
    }

    /*! \brief indicates if the hash table is properly initialized
     * \return \c true iff the hash table is properly initialized
     */
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    bool is_initialized() const noexcept
    {
        return hash_table_.is_initialized();
    }

    /*! \brief get the status of the hash table
     * \param stream CUDA stream in which this operation is executed in
     * \return the status
     */
    HOSTQUALIFIER INLINEQUALIFIER
    status_type peek_status(const cudaStream_t stream = 0) const noexcept
    {
        return hash_table_.peek_status(stream);
    }

    /*! \brief get and reset the status of the hash table
     * \param[in] stream CUDA stream in which this operation is executed in
     * \return the status
     */
    HOSTQUALIFIER INLINEQUALIFIER
    status_type pop_status(const cudaStream_t stream = 0) noexcept
    {
        return hash_table_.pop_status(stream);
    }

    /*! \brief get the key capacity of the hash table
     * \return number of key slots in the hash table
     */
    HOSTQUALIFIER INLINEQUALIFIER
    index_type key_capacity() const noexcept
    {
        return hash_table_.capacity();
    }

    /*! \brief get the maximum value capacity of the hash table
     * \return maximum value capacity
     */
    HOSTQUALIFIER INLINEQUALIFIER
    index_type value_capacity() const noexcept
    {
        return value_store_.capacity();
    }

    /*! \brief number of keys stored inside the hash table
     * \param[in] stream CUDA stream in which this operation is executed in
     * \return number of keys inside the hash table
     */
    HOSTQUALIFIER INLINEQUALIFIER
    index_type num_keys(const cudaStream_t stream = 0) const noexcept
    {
        return hash_table_.size(stream);
    }

    /*! \brief get number of values to a corresponding key inside the hash table
     * \param[in] key_in key to probe
     * \param[out] num_out number of values
     * \param[in] group cooperative group this operation is executed in
     * \param[in] probing_length maximum number of probing attempts
     * \return status (per thread)
     */
    DEVICEQUALIFIER INLINEQUALIFIER
    status_type num_values(
        const key_type key_in,
        index_type& num_out,
        const cg::thread_block_tile<cg_size()>& group,
        const index_type probing_length = defaults::probing_length()) const noexcept
    {
        handle_type handle;

        status_type status =
            hash_table_.retrieve(key_in, handle, group, probing_length);

        num_out = (!status.has_any()) ? value_store_.size(handle) : 0;

        return status;
    }

    /*! \brief get number of values to a corresponding set of keys inside the hash table
     * \param[in] keys_in keys to probe
     * \param[in] num_in input size
     * \param[out] num_out total number of values in this query
     * \param[out] num_per_key_out number of values per key
     * \param[in] probing_length maximum number of probing attempts
     * \param[in] stream CUDA stream in which this operation is executed in
     * \param[out] status_out status information (per key)
     */
    template<class StatusHandler = defaults::status_handler_t>
    HOSTQUALIFIER INLINEQUALIFIER
    void num_values(
        const key_type * const keys_in,
        const index_type num_in,
        index_type& num_out,
        index_type * const num_per_key_out = nullptr,
        const cudaStream_t stream = 0,
        const index_type probing_length = defaults::probing_length(),
        typename StatusHandler::base_type * const status_out = nullptr) const noexcept
    {
        static_assert(
            checks::is_status_handler<StatusHandler>(),
            "not a valid status handler type");

        if(!hash_table_.is_initialized_) return;

        index_type * const tmp = hash_table_.temp_.get();
        cudaMemsetAsync(tmp, 0, sizeof(index_type), stream);

        kernels::num_values<BucketListHashTable, StatusHandler>
        <<<SDIV(num_in * cg_size(), WARPCORE_BLOCKSIZE), WARPCORE_BLOCKSIZE, 0, stream>>>
        (keys_in, num_in, tmp, num_per_key_out, *this, probing_length, status_out);

        cudaMemcpyAsync(&num_out, tmp, sizeof(index_type), D2H, stream);

        if(stream == 0)
        {
            cudaStreamSynchronize(stream);
        }
    }

    /*! \brief get number of values inside the hash table
     * \param[in] stream CUDA stream in which this operation is executed in
     * \return total number of values
     */
    HOSTQUALIFIER INLINEQUALIFIER
    index_type num_values(const cudaStream_t stream = 0) const noexcept
    {
        index_type * tmp = hash_table_.temp_.get();

        cudaMemsetAsync(tmp, 0, sizeof(index_type), stream);

        hash_table_.for_each(
            [=, *this] DEVICEQUALIFIER (key_type, const handle_type& handle)
            {
                atomicAdd(tmp, value_store_.size(handle));
            },
            stream);

        index_type out = 0;

        cudaMemcpyAsync(&out, tmp, sizeof(index_type), D2H, stream);

        cudaStreamSynchronize(stream);

        return out;
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
    /*! \brief joins additional flags to the hash table's status
     * \info \c const on purpose
     * \param[in] status new status
     * \param[in] stream CUDA stream in which this operation is executed in
     */
    HOSTQUALIFIER INLINEQUALIFIER
    void join_status(
        const status_type status,
        const cudaStream_t stream = 0) const noexcept
    {
        hash_table_.join_status(status, stream);
    }

    /*! \brief joins additional flags to the hash table's status
     * \info \c const on purpose
     * \param[in] status new status
     */
    DEVICEQUALIFIER INLINEQUALIFIER
    void device_join_status(const status_type status) const noexcept
    {
        hash_table_.device_join_status(status);
    }

    hash_table_type hash_table_; //< storage class for keys
    value_store_type value_store_; //< multi-value storage class
    const index_type max_values_per_key_; //< maximum number of values to store per key
    bool is_copy_; //< indicates if this object is a shallow copy

    template<class Core, class StatusHandler>
    GLOBALQUALIFIER
    friend void kernels::retrieve(
        const typename Core::key_type * const,
        const index_type,
        const index_type * const,
        const index_type * const,
        typename Core::value_type * const,
        const Core,
        const index_type,
        typename StatusHandler::base_type * const);

}; // class BucketListHashTable

} // namespace warpcore

#endif /* WARPCORE_BUCKET_LIST_HASH_TABLE_CUH */