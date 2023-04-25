#ifndef WARPCORE_MULTI_BUCKET_HASH_TABLE_CUH
#define WARPCORE_MULTI_BUCKET_HASH_TABLE_CUH

#include "hash_set.cuh"

namespace warpcore
{

template<
    class Value,
    std::uint32_t BucketSize = 1>
struct ArrayBucket {
    using value_type = Value;
    using index_type = std::uint32_t;

    static_assert(
        BucketSize > 0,
        "invalid bucket size of 0");

    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    explicit ArrayBucket(value_type value) noexcept
    {
        #ifdef __CUDA_ARCH__
        #pragma unroll
        #endif
        for(index_type i = 0; i < bucket_size(); ++i)
            values_[i] = value;
    }

    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    ArrayBucket(const ArrayBucket& other) noexcept
    {
        #ifdef __CUDA_ARCH__
        #pragma unroll
        #endif
        for(index_type i = 0; i < bucket_size(); ++i)
            values_[i] = other.values_[i];
    }

    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    ArrayBucket& operator =(const ArrayBucket& other) noexcept
    {
        #ifdef __CUDA_ARCH__
        #pragma unroll
        #endif
        for(index_type i = 0; i < bucket_size(); ++i)
            values_[i] = other.values_[i];
        return *this;
    }

    /*! \brief get bucket size
     * \return bucket size
     */
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    static constexpr index_type bucket_size() noexcept
    {
        return BucketSize;
    }

    /*! \brief accessor
     * \param[in] i index to access
     * \return value at position \c i
     */
    DEVICEQUALIFIER INLINEQUALIFIER
    constexpr value_type& operator[](const index_type i) noexcept
    {
        return values_[i];
    }

    /*! \brief const accessor
     * \param[in] i index to access
     * \return value at position \c i
     */
    DEVICEQUALIFIER INLINEQUALIFIER
    constexpr const value_type& operator[](const index_type i) const noexcept
    {
        return values_[i];
    }

    value_type values_[BucketSize];
};


/*! \brief multi-value hash table
 * \tparam Key key type ( \c std::uint32_t or \c std::uint64_t )
 * \tparam Value value type
 * \tparam EmptyKey key which represents an empty slot
 * \tparam TombstoneKey key which represents an erased slot
 * \tparam ProbingScheme probing scheme from \c warpcore::probing_schemes
 * \tparam TableStorage memory layout from \c warpcore::storage::key_value
 * \tparam TempMemoryBytes size of temporary storage (typically a few kB)
 */
template<
    class Key,
    class Value,
    Key EmptyKey = defaults::empty_key<Key>(),
    Key TombstoneKey = defaults::tombstone_key<Key>(),
    Value EmptyValue = defaults::empty_key<Value>(),
    class ProbingScheme = defaults::probing_scheme_t<Key, 8>,
    class TableStorage = defaults::table_storage_t<Key, ArrayBucket<Value,2>>,
    index_t TempMemoryBytes = defaults::temp_memory_bytes()>
class MultiBucketHashTable
{
    static_assert(
        checks::is_valid_key_type<Key>(),
        "invalid key type");

    static_assert(
        checks::is_valid_slot_type<Value>(),
        "invalid value type");

    static_assert(
        EmptyKey != TombstoneKey,
        "empty key and tombstone key must not be identical");

    static_assert(
        checks::is_cycle_free_probing_scheme<ProbingScheme>(),
        "not a valid probing scheme type");

    static_assert(
        std::is_same<typename ProbingScheme::key_type, Key>::value,
        "probing key type differs from table's key type");

    static_assert(
        checks::is_key_value_storage<TableStorage>(),
        "not a valid storage type");

    static_assert(
        std::is_same<typename TableStorage::key_type, Key>::value,
        "storage's key type differs from table's key type");

    static_assert(
        std::is_same<typename TableStorage::value_type::value_type, Value>::value,
        "storage's value type differs from table's value type");

    static_assert(
        TempMemoryBytes >= sizeof(index_t),
        "temporary storage must at least be of size index_type");

    using temp_type = storage::CyclicStore<index_t>;

public:
    using key_type = Key;
    using value_type = Value;
    using bucket_type = typename TableStorage::value_type;
    using index_type = index_t;
    using status_type = Status;
    using probing_scheme_type = ProbingScheme;

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

    /*! \brief get empty value
     * \return empty value
     */
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    static constexpr value_type empty_value() noexcept
    {
        return EmptyValue;
    }


    /*! \brief get cooperative group size
     * \return cooperative group size
     */
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    static constexpr index_type cg_size() noexcept
    {
        return ProbingScheme::cg_size();
    }

    /*! \brief get bucket size
     * \return bucket size
     */
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    static constexpr index_type bucket_size() noexcept
    {
        return TableStorage::value_type::bucket_size();
    }

    /*! \brief constructor
     * \param[in] min_capacity minimum number of slots in the hash table
     * \param[in] seed random seed
     * \param[in] max_values_per_key maximum number of values to store per key
     * \param[in] no_init whether to initialize the table at construction or not
     */
    HOSTQUALIFIER INLINEQUALIFIER
    explicit MultiBucketHashTable(
        const index_type min_capacity,
        const key_type seed = defaults::seed<key_type>(),
        const index_type max_values_per_key =
            std::numeric_limits<index_type>::max(),
        const bool no_init = false) noexcept :
        status_(nullptr),
        table_(detail::get_valid_capacity(min_capacity, cg_size())),
        temp_(TempMemoryBytes / sizeof(index_type)),
        seed_(seed),
        max_values_per_key_(max_values_per_key),
        num_keys_(nullptr),
        num_occupied_(nullptr),
        is_copy_(false),
        is_initialized_(false)
    {
        cudaMalloc(&status_, sizeof(status_type));
        cudaMalloc(&num_keys_, sizeof(index_type));
        cudaMalloc(&num_occupied_, sizeof(index_type));

        assign_status(table_.status() + temp_.status());

        if(!no_init) init();
    }

    /*! \brief copy-constructor (shallow)
     *  \param[in] object to be copied
     */
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    MultiBucketHashTable(const MultiBucketHashTable& o) noexcept :
        status_(o.status_),
        table_(o.table_),
        temp_(o.temp_),
        seed_(o.seed_),
        max_values_per_key_(o.max_values_per_key_),
        num_keys_(o.num_keys_),
        num_occupied_(o.num_occupied_),
        is_copy_(true),
        is_initialized_(o.is_initialized_)
    {}

    /*! \brief move-constructor
     *  \param[in] object to be moved
     */
    HOSTQUALIFIER INLINEQUALIFIER
    MultiBucketHashTable(MultiBucketHashTable&& o) noexcept :
        status_(std::move(o.status_)),
        table_(std::move(o.table_)),
        temp_(std::move(o.temp_)),
        seed_(std::move(o.seed_)),
        max_values_per_key_(std::move(o.max_values_per_key_)),
        num_keys_(std::move(o.num_keys_)),
        num_occupied_(std::move(o.num_occupied_)),
        is_copy_(std::move(o.is_copy_)),
        is_initialized_(std::move(o.is_initialized_))
    {
        o.is_copy_ = true;
    }

    #ifndef __CUDA_ARCH__
    /*! \brief destructor
     */
    HOSTQUALIFIER INLINEQUALIFIER
    ~MultiBucketHashTable() noexcept
    {
        if(!is_copy_)
        {
            if(status_ != nullptr) cudaFree(status_);
            if(num_keys_ != nullptr) cudaFree(num_keys_);
            if(num_occupied_ != nullptr) cudaFree(num_occupied_);
        }
    }
    #endif

    /*! \brief (re)initialize the hash table
     * \param[in] stream CUDA stream in which this operation is executed in
     */
    HOSTQUALIFIER INLINEQUALIFIER
    void init(const cudaStream_t stream = 0) noexcept
    {
        is_initialized_ = false;

        if(!table_.status().has_not_initialized() &&
            !temp_.status().has_not_initialized())
        {
            table_.init_keys(empty_key(), stream);
            table_.init_values(bucket_type(empty_value()), stream);

            assign_status(table_.status() + temp_.status(), stream);

            cudaMemsetAsync(num_keys_, 0, sizeof(index_type), stream);
            cudaMemsetAsync(num_occupied_, 0, sizeof(index_type), stream);

            is_initialized_ = true;
        }
    }

private:
    DEVICEQUALIFIER INLINEQUALIFIER
    bool insert_into_bucket(
        const index_type last_key_pos,
        const value_type value_in,
        const cg::thread_block_tile<cg_size()>& group,
        index_type num_values,
        status_type& status) noexcept
    {
        #pragma unroll
        for(index_type i = 0;
                       i < SDIV(bucket_size(),cg_size())*cg_size();
                       i += cg_size())
        {
            // first bucket value always written after key insert
            const value_type table_value =
                ((0 < group.thread_rank()) && (i + group.thread_rank() < bucket_size())) ?
                table_[last_key_pos].value[group.thread_rank()] :
                ~empty_value();

            auto empty_value_mask = group.ballot(is_empty_value(table_value));

            num_values += min(bucket_size(),cg_size()) - __popc(empty_value_mask);

            if(num_values >= max_values_per_key_)
            {
                status = status_type::duplicate_key() +
                         status_type::max_values_for_key_reached();
                device_join_status(status);
                return true;
            }

            bool success = false;

            while(empty_value_mask)
            {
                const auto leader = ffs(empty_value_mask) - 1;

                if(group.thread_rank() == leader)
                {
                    const auto old =
                        atomicCAS(&(table_[last_key_pos].value[i+group.thread_rank()]), table_value, value_in);

                    success = (old == table_value);
                }

                if(group.any(success))
                {
                    status = (num_values > 0) ?
                        status_type::duplicate_key() : status_type::none();
                    return true;
                }

                ++num_values;
                if(num_values >= max_values_per_key_)
                {
                    status = status_type::duplicate_key() +
                             status_type::max_values_for_key_reached();
                    device_join_status(status);
                    return true;
                }

                empty_value_mask ^= 1UL << leader;
            }
        }

        return false;
    }

public:
    /*! \brief inserts a key into the hash table
     * \param[in] key_in key to insert into the hash table
     * \param[in] value_in value that corresponds to \c key_in
     * \param[in] group cooperative group
     * \param[in] probing_length maximum number of probing attempts
     * \return status (per thread)
     */
    DEVICEQUALIFIER INLINEQUALIFIER
    status_type insert(
        const key_type key_in,
        const value_type value_in,
        const cg::thread_block_tile<cg_size()>& group,
        const index_type probing_length = defaults::probing_length()) noexcept
    {
        if(!is_initialized_)
        {
            return status_type::not_initialized();
        }

        if(!is_valid_key(key_in))
        {
            device_join_status(status_type::invalid_key());
            return status_type::invalid_key();
        }

        if(!is_valid_value(value_in))
        {
            device_join_status(status_type::invalid_value());
            return status_type::invalid_value();
        }

        ProbingScheme iter(capacity(), probing_length, group);
        index_type num_values_plus_bucket_size = 0; // count one bucket less

        index_type last_key_pos = std::numeric_limits<index_type>::max();
        for(index_type i = iter.begin(key_in, seed_); i != iter.end(); i = iter.next())
        {
            const key_type table_key = cub::ThreadLoad<cub::LOAD_VOLATILE>(&table_[i].key);

            auto empty_key_mask = group.ballot(is_empty_key(table_key));

            const auto key_found_mask = group.ballot(table_key == key_in);

            const auto new_last_key_pos = group.shfl(i, 31 - __clz(key_found_mask));

            last_key_pos = key_found_mask ? new_last_key_pos : last_key_pos;

            num_values_plus_bucket_size += bucket_size() * __popc(key_found_mask);

            // early exit
            if(num_values_plus_bucket_size >= max_values_per_key_)
            {
                if(bucket_size() == 1)
                {
                    // num values = num buckets, so no space left
                    status_type status = status_type::duplicate_key() +
                                         status_type::max_values_for_key_reached();
                    device_join_status(status);
                    return status;
                }
                else
                {
                    status_type status = status_type::unknown_error();
                    // check if space left in last bucket
                    insert_into_bucket(last_key_pos, value_in, group,
                        num_values_plus_bucket_size - bucket_size(), status);
                    return status;
                }
            }

            while(empty_key_mask)
            {
                status_type status;
                if((bucket_size() > 1) &&
                   (last_key_pos < std::numeric_limits<index_type>::max()) &&
                   insert_into_bucket(last_key_pos, value_in, group,
                        num_values_plus_bucket_size - bucket_size(), status))
                    return status;

                // insert key
                bool success = false;
                bool key_collision = false;

                const auto leader = ffs(empty_key_mask) - 1;

                if(group.thread_rank() == leader)
                {
                    const auto old =
                       atomicCAS(&(table_[i].key), table_key, key_in);

                    success = (old == table_key);
                    key_collision = (old == key_in);

                    if(success)
                    {
                        // relaxed write to first slot in value array
                        table_[i].value[0] = value_in;

                        helpers::atomicAggInc(num_occupied_);

                        if(num_values_plus_bucket_size == 0)
                        {
                            helpers::atomicAggInc(num_keys_);
                        }
                    }
                }

                if(group.any(success))
                {
                    return (num_values_plus_bucket_size > 0) ?
                        status_type::duplicate_key() : status_type::none();
                }

                key_collision = group.any(key_collision);
                num_values_plus_bucket_size += key_collision*bucket_size();

                if(bucket_size() == 1)
                {
                    if(num_values_plus_bucket_size >= max_values_per_key_)
                    {
                        status_type status = status_type::duplicate_key() +
                                             status_type::max_values_for_key_reached();
                        device_join_status(status);
                        return status;
                    }
                }
                else
                {
                    // check position in next iteration
                    const auto new_last_key_pos = group.shfl(i, leader);
                    last_key_pos =  key_collision ? new_last_key_pos : last_key_pos;
                }

                empty_key_mask ^= 1UL << leader;
            }
        }

        status_type status;
        if((bucket_size() > 1) &&
           (last_key_pos < std::numeric_limits<index_type>::max()) &&
           insert_into_bucket(last_key_pos, value_in, group,
                num_values_plus_bucket_size - bucket_size(), status))
            return status;

        status = (num_values_plus_bucket_size > 0) ?
            status_type::probing_length_exceeded() + status_type::duplicate_key() :
            status_type::probing_length_exceeded();
        device_join_status(status);
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

        if(!is_initialized_) return;

        kernels::insert<MultiBucketHashTable, StatusHandler>
        <<<SDIV(num_in * cg_size(), MAXBLOCKSIZE), MAXBLOCKSIZE, 0, stream>>>
        (keys_in, values_in, num_in, *this, probing_length, status_out);
    }

     /*! \brief retrieves all values to a corresponding key
     * \param[in] key_in key to retrieve from the hash table
     * \param[out] values_out values for \c key_in
     * \param[out] num_out number of retrieved values
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
        if(values_out == nullptr)
        {
            const auto status = num_values(key_in, num_out, group, probing_length);
            device_join_status(status_type::dry_run());
            return status_type::dry_run() + status;
        }
        else
        {
            return for_each([=, *this] DEVICEQUALIFIER
                (const key_type /* key */, const value_type& value, const index_type index)
                {
                    values_out[index] = value;
                },
                key_in,
                num_out,
                group,
                probing_length);
        }
    }

    /*! \brief retrieve a set of keys from the hash table
     * \note this method has a dry-run mode where it only calculates the needed array sizes in case no memory (aka \c nullptr ) is provided
     * \note \c end_offsets_out can be \c begin_offsets_out+1
     * \tparam StatusHandler handles returned status per key (see \c status_handlers)
     * \param[in] keys_in pointer to keys to retrieve from the hash table
     * \param[in] num_in number of keys to retrieve
     * \param[out] begin_offsets_out begin of value range for a corresponding key in \c values_out
     * \param[out] end_offsets_out end of value range for a corresponding key in \c values_out
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

        if(!is_initialized_) return;

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

            kernels::retrieve<MultiBucketHashTable, StatusHandler>
            <<<SDIV(num_in * cg_size(), MAXBLOCKSIZE), MAXBLOCKSIZE, 0, stream>>>
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
                <<<SDIV(num_in, MAXBLOCKSIZE), MAXBLOCKSIZE, 0, stream>>>
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

    /*! \brief retrieves all elements from the hash table
     * \note this method has a dry-run mode where it only calculates the needed array sizes in case no memory (aka \c nullptr ) is provided
     * \note this method implements a multi-stage dry-run mode
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
        if(!is_initialized_) return;

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

   /*! \brief retrieve all unqiue keys
    * \info this method has a dry-run mode where it only calculates the needed array sizes in case no memory (aka \c nullptr ) is provided
    * \param[out] keys_out retrieved unqiue keys
    * \param[out] num_out numof unique keys
    * \param[in] stream CUDA stream in which this operation is executed in
    */
    HOSTQUALIFIER INLINEQUALIFIER
    void retrieve_all_keys(
        key_type * const keys_out,
        index_type& num_out,
        const cudaStream_t stream = 0) const noexcept
    {
        if(!is_initialized_) return;

        if(keys_out != nullptr)
        {
            index_type * const tmp = temp_.get();
            cudaMemsetAsync(tmp, 0, sizeof(index_type), stream);

            kernels::for_each_unique_key
            <<<SDIV(capacity()*cg_size(), MAXBLOCKSIZE), MAXBLOCKSIZE, 0, stream>>>
            ([=] DEVICEQUALIFIER (const key_type& key)
            {
                index_type out = helpers::atomicAggInc(tmp);
                keys_out[out] = key;
            }, *this);

            cudaMemcpyAsync(&num_out, tmp, sizeof(index_type), D2H, stream);

            if(stream == 0)
            {
                cudaStreamSynchronize(stream);
            }
        }
        else
        {
            num_out = num_keys(stream);
            join_status(status_type::dry_run(), stream);
        }
    }

    /*! \brief applies a funtion over all values of a specified key
     * \tparam Func type of map i.e. CUDA device lambda
     * \param[in] f map to apply
     * \param[in] key_in key to consider
     * \param[out] num_values_out number of values associated to \c key_in
     * \param[in] group cooperative group
     * \param[in] probing_length maximum number of probing attempts
     * \return status (per thread)
     */
    template<class Func>
    DEVICEQUALIFIER INLINEQUALIFIER
    status_type for_each(
        Func f,
        const key_type key_in,
        index_type& num_values_out,
        const cg::thread_block_tile<cg_size()>& group,
        const index_type probing_length = defaults::probing_length()) const noexcept
    {
        if(!is_initialized_) return status_type::not_initialized();

        if(!is_valid_key(key_in))
        {
            num_values_out = 0;
            device_join_status(status_type::invalid_key());
            return status_type::invalid_key();
        }

        ProbingScheme iter(capacity(), min(probing_length, capacity()), group);

        index_type num = 0;
        for(index_type i = iter.begin(key_in, seed_); i != iter.end(); i = iter.next())
        {
            const auto table_key = table_[i].key;
            const auto hit = (table_key == key_in);
            const auto hit_mask = group.ballot(hit);

            index_type num_empty = 0;
            if(hit)
            {
                const auto j =
                    num + bucket_size() * __popc(hit_mask & ((1U << group.thread_rank()) - 1));

                const auto bucket = table_[i].value;
                #pragma unroll
                for(index_type b = 0; b < bucket_size(); ++b) {
                    const auto& value = bucket[b];
                    // if(value != empty_value() && j+b < max_values_per_key_)
                    if(value != empty_value())
                        f(key_in, value, j+b);
                    else
                        ++num_empty;
                }
            }

            // get num_empty from last bucket in group
            // if not hit this return 0 from last thread
            num_empty = group.shfl(num_empty, 31 - __clz(hit_mask));

            num += bucket_size() * __popc(hit_mask) - num_empty;

            if(group.any(is_empty_key(table_key) || num >= max_values_per_key_))
            {
                num_values_out = num;

                if(num == 0)
                {
                    device_join_status(status_type::key_not_found());
                    return status_type::key_not_found();
                }
                else
                {
                    return status_type::none();
                }
            }
        }

        num_values_out = num;
        device_join_status(status_type::probing_length_exceeded());
        return status_type::probing_length_exceeded();
    }

    /*! \brief applies a funtion over all key bucket pairs inside the table
     * \tparam Func type of map i.e. CUDA device lambda
     * \param[in] f map to apply
     * \param[in] stream CUDA stream in which this operation is executed in
     * \param[in] size of dynamic shared memory to reserve for this execution
     */
    template<class Func>
    HOSTQUALIFIER INLINEQUALIFIER
    void for_each_bucket(
        Func f, // TODO const?
        const cudaStream_t stream = 0,
        const index_type smem_bytes = 0) const noexcept
    {
        if(!is_initialized_) return;

        kernels::for_each
        <<<SDIV(capacity(), MAXBLOCKSIZE), MAXBLOCKSIZE, smem_bytes, stream>>>
        (f, *this);
    }

    /*! \brief applies a funtion over all key value pairs inside the table
     * \tparam Func type of map i.e. CUDA device lambda
     * \param[in] f map to apply
     * \param[in] stream CUDA stream in which this operation is executed in
     * \param[in] size of dynamic shared memory to reserve for this execution
     */
    template<class Func>
    HOSTQUALIFIER INLINEQUALIFIER
    void for_each_value(
        Func f, // TODO const?
        const cudaStream_t stream = 0,
        const index_type smem_bytes = 0) const noexcept
    {
        if(!is_initialized_) return;

        auto bucket_f = [=, f=std::move(f)] DEVICEQUALIFIER
        (const key_type key, const bucket_type bucket) mutable
        {
            #pragma unroll
            for(index_type b = 0; b < bucket_size(); ++b) {
                const auto& value = bucket[b];
                if(value != empty_value())
                    f(key, value);
            }
        };

        kernels::for_each
        <<<SDIV(capacity(), MAXBLOCKSIZE), MAXBLOCKSIZE, smem_bytes, stream>>>
        (bucket_f, *this);
    }

    /*! \brief applies a funtion over all key value pairs
     * \tparam Func type of map i.e. CUDA device lambda
     * \tparam StatusHandler handles returned status per key (see \c status_handlers)
     * \param[in] f map to apply
     * \param[in] keys_in keys to consider
     * \param[in] num_in number of keys
     * \param[in] stream CUDA stream in which this operation is executed in
     * \param[in] probing_length maximum number of probing attempts
     * \param[out] status_out status information (per key)
     * \param[in] size of dynamic shared memory to reserve for this execution
     */
    template<class Func, class StatusHandler = defaults::status_handler_t>
    HOSTQUALIFIER INLINEQUALIFIER
    void for_each(
        Func f, // TODO const?
        const key_type * const keys_in,
        const index_type num_in,
        const cudaStream_t stream = 0,
        const index_type probing_length = defaults::probing_length(),
        typename StatusHandler::base_type * const status_out = nullptr,
        const index_type smem_bytes = 0) const noexcept
    {
        static_assert(
            checks::is_status_handler<StatusHandler>(),
            "not a valid status handler type");

        if(!is_initialized_) return;

        kernels::for_each<Func, MultiBucketHashTable>
        <<<SDIV(capacity(), MAXBLOCKSIZE), MAXBLOCKSIZE, smem_bytes, stream>>>
        (f, keys_in, num_in, *this, status_out);
    }

    /*! \brief number of unique keys inside the table
     * \param[in] stream CUDA stream in which this operation is executed in
     * \return number of unique keys
     */
    HOSTQUALIFIER INLINEQUALIFIER
    index_type num_keys(const cudaStream_t stream = 0) const noexcept
    {
        index_type num = 0;

        cudaMemcpyAsync(&num, num_keys_, sizeof(index_type), D2H, stream);

        cudaStreamSynchronize(stream);

        return num;
    }

    /*! \brief number of occupied slots in the hash table
     * \param[in] stream CUDA stream in which this operation is executed in
     * \return the number of occupied slots
     */
    HOSTQUALIFIER INLINEQUALIFIER
    index_type num_occupied(const cudaStream_t stream = 0) const noexcept
    {
        index_type num = 0;

        cudaMemcpyAsync(&num, num_occupied_, sizeof(index_type), D2H, stream);

        cudaStreamSynchronize(stream);

        return num;
    }

    /*! \brief total number of values inside the table
     * \param[in] key_in key to be probed
     * \param[out] num_out number of values associated to \c key_in*
     * \param[in] group cooperative group
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
        return for_each([=] DEVICEQUALIFIER (
                const key_type /* key */,
                const value_type& /* value */,
                const index_type /* index */) {},
            key_in,
            num_out,
            group,
            probing_length);
    }

    /*! \brief number of values associated to a set of keys
     * \info this function returns only \c num_out if \c num_per_key_out==nullptr
     * \tparam StatusHandler handles returned status per key (see \c status_handlers)
     * \param[in] keys_in keys to consider
     * \param[in] num_in number of keys
     * \param[out] num_out total number of values
     * \param[out] num_per_key_out number of values per key
     * \param[in] stream CUDA stream in which this operation is executed in
     * \param[in] probing_length maximum number of probing attempts
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
        if(!is_initialized_) return;

        // TODO check if shared memory is benefitial

        index_type * const tmp = temp_.get();
        cudaMemsetAsync(tmp, 0, sizeof(index_type), stream);

        kernels::num_values<MultiBucketHashTable, StatusHandler>
        <<<SDIV(num_in * cg_size(), MAXBLOCKSIZE), MAXBLOCKSIZE, 0, stream>>>
        (keys_in, num_in, tmp, num_per_key_out, *this, probing_length, status_out);

        cudaMemcpyAsync(&num_out, tmp, sizeof(index_type), D2H, stream);

        if(stream == 0)
        {
            cudaStreamSynchronize(stream);
        }
    }

    /*! \brief number of values stored inside the hash table
     * \info alias for \c size()
     * \param[in] stream CUDA stream in which this operation is executed in
     * \return the number of values
     */
    HOSTQUALIFIER INLINEQUALIFIER
    index_type num_values(const cudaStream_t stream = 0) const noexcept
    {
        return size(stream);
    }

    /*! \brief number of values stored inside the hash table
     * \param[in] stream CUDA stream in which this operation is executed in
     * \return the number of values
     */
    HOSTQUALIFIER INLINEQUALIFIER
    index_type size(const cudaStream_t stream = 0) const noexcept
    {
        if(!is_initialized_) return 0;

        index_type out;
        index_type * tmp = temp_.get();

        cudaMemsetAsync(tmp, 0, sizeof(index_t), stream);

        kernels::num_values
        <<<SDIV(capacity(), MAXBLOCKSIZE), MAXBLOCKSIZE, 0, stream>>>
        (tmp, *this);

        cudaMemcpyAsync(
            &out,
            tmp,
            sizeof(index_type),
            D2H,
            stream);

        cudaStreamSynchronize(stream);

        return out;
    }

    /*! \brief current load factor of the hash table
     * \param[in] stream CUDA stream in which this operation is executed in
     * \return load factor
     */
    HOSTQUALIFIER INLINEQUALIFIER
    float key_load_factor(const cudaStream_t stream = 0) const noexcept
    {
        return float(num_occupied(stream)) / float(capacity());
    }

    /*! \brief current load factor of the hash table
     * \param[in] stream CUDA stream in which this operation is executed in
     * \return load factor
     */
    HOSTQUALIFIER INLINEQUALIFIER
    float value_load_factor(const cudaStream_t stream = 0) const noexcept
    {
        return float(num_values(stream)) / float(capacity()*bucket_size());
    }

    /*! \brief current storage density of the hash table
     * \param[in] stream CUDA stream in which this operation is executed in
     * \return storage density
     */
    HOSTQUALIFIER INLINEQUALIFIER
    float storage_density(const cudaStream_t stream = 0) const noexcept
    {
        const index_type key_bytes = num_keys(stream) * sizeof(key_type);
        const index_type value_bytes = num_values(stream) * sizeof(value_type);
        const index_type table_bytes = bytes_total();

        return float(key_bytes + value_bytes) / float(table_bytes);
    }

    /*! \brief current relative storage density of the hash table
     * \param stream CUDA stream in which this operation is executed in
     * \return storage density
     */
    HOSTQUALIFIER INLINEQUALIFIER
    float relative_storage_density(const cudaStream_t stream = 0) const noexcept
    {
        const index_type key_bytes = num_keys(stream) * sizeof(key_type);
        const index_type value_bytes = num_values(stream) * sizeof(value_type);
        const index_type occupied_bytes =
            num_occupied(stream) * sizeof(key_type) + value_bytes;

        return float(key_bytes + value_bytes) / (occupied_bytes);
    }

    /*! \brief get the key capacity of the hash table
     * \return number of key slots in the hash table
     */
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    index_type capacity() const noexcept
    {
        return table_.capacity();
    }

    /*! \brief get the maximum value capacity of the hash table
     * \return maximum value capacity
     */
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    index_type value_capacity() const noexcept
    {
        return table_.capacity() * bucket_size();
    }

    /*! \brief get the total number of bytes occupied by this data structure
     *  \return bytes
     */
    HOSTQUALIFIER INLINEQUALIFIER
    index_type bytes_total() const noexcept
    {
        return table_.bytes_total() + sizeof(index_type);
    }

    /*! \brief indicates if the hash table is properly initialized
     * \return \c true iff the hash table is properly initialized
     */
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    bool is_initialized() const noexcept
    {
        return is_initialized_;
    }

    /*! \brief get the status of the hash table
     * \param[in] stream CUDA stream in which this operation is executed in
     * \return the status
     */
    HOSTQUALIFIER INLINEQUALIFIER
    status_type peek_status(const cudaStream_t stream = 0) const noexcept
    {
        status_type status = status_type::not_initialized();

        if(status_ != nullptr)
        {
            cudaMemcpyAsync(
                &status,
                status_,
                sizeof(status_type),
                D2H,
                stream);

            cudaStreamSynchronize(stream);
        }

        return status;
    }

    /*! \brief get and reset the status of the hash table
     * \param[in] stream CUDA stream in which this operation is executed in
     * \return the status
     */
    HOSTQUALIFIER INLINEQUALIFIER
    status_type pop_status(const cudaStream_t stream = 0) noexcept
    {
        status_type status = status_type::not_initialized();

        if(status_ != nullptr)
        {
            cudaMemcpyAsync(
                &status,
                status_,
                sizeof(status_type),
                D2H,
                stream);

            assign_status(table_.status(), stream);
        }

        return status;
    }

    /*! \brief checks if \c key is equal to \c EmptyKey
     * \return \c bool
     */
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    static constexpr bool is_empty_key(const key_type key) noexcept
    {
        return (key == empty_key());
    }

    /*! \brief checks if \c key is equal to \c TombstoneKey
     * \return \c bool
     */
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    static constexpr bool is_tombstone_key(const key_type key) noexcept
    {
        return (key == tombstone_key());
    }

    /*! \brief checks if \c key is not equal to \c (EmptyKey||TombstoneKey)
     * \return \c bool
     */
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    static constexpr bool is_valid_key(const key_type key) noexcept
    {
        return (key != empty_key() && key != tombstone_key());
    }

    /*! \brief checks if \c value is equal to \c EmptyValue
     * \return \c bool
     */
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    static constexpr bool is_empty_value(const value_type value) noexcept
    {
        return (value == empty_value());
    }

    /*! \brief checks if \c value is equal not to \c EmptyValue
     * \return \c bool
     */
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    static constexpr bool is_valid_value(const value_type value) noexcept
    {
        return (value != empty_value());
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
    /*! \brief assigns the hash table's status
     * \info \c const on purpose
     * \param[in] status new status
     * \param[in] stream CUDA stream in which this operation is executed in
     */
    HOSTQUALIFIER INLINEQUALIFIER
    void assign_status(
        const status_type status,
        const cudaStream_t stream = 0) const noexcept
    {
        if(status_ != nullptr)
        {
            cudaMemcpyAsync(
                status_,
                &status,
                sizeof(status_type),
                H2D,
                stream);

            cudaStreamSynchronize(stream);
        }
    }

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
        if(status_ != nullptr)
        {
            status_type peeked = peek_status(stream);
            const status_type joined = peeked + status;

            if(joined != peeked)
            {
                assign_status(joined, stream);
            }
        }
    }

    /*! \brief joins additional flags to the hash table's status
     * \info \c const on purpose
     * \param[in] status new status
     */
    DEVICEQUALIFIER INLINEQUALIFIER
    void device_join_status(const status_type status) const noexcept
    {
        if(status_ != nullptr)
        {
            status_->atomic_join(status);
        }
    }

    status_type * status_; //< pointer to status
    TableStorage table_; //< actual key/value storage
    temp_type temp_; //< temporary memory
    key_type seed_; //< random seed
    index_type max_values_per_key_; //< maximum number of values to store per key
    index_type * num_keys_; //< pointer to the count of unique keys
    index_type * num_occupied_; //< pointer to the count of occupied key slots
    bool is_copy_; //< indicates if table is a shallow copy
    bool is_initialized_; //< indicates if table is properly initialized

    template<class Core>
    GLOBALQUALIFIER
    friend void kernels::size(index_type * const, const Core);

    template<class Core>
    GLOBALQUALIFIER
    friend void kernels::num_values(index_type * const, const Core);

    template<class Func, class Core>
    GLOBALQUALIFIER
    friend void kernels::for_each(Func, const Core);

    template<class Func, class Core>
    GLOBALQUALIFIER
    friend void kernels::for_each_unique_key(Func, const Core);

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

}; // class MultiBucketHashTable

} // namespace warpcore

#endif /* WARPCORE_MULTI_BUCKET_HASH_TABLE_CUH */