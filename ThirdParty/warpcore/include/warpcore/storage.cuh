#ifndef WARPCORE_STORAGE_CUH
#define WARPCORE_STORAGE_CUH

namespace warpcore
{

/*! \brief storage classes
 */
namespace storage
{

/*! \brief thread-safe device-sided ring buffer without any overflow checks
 * \tparam T base type
 */
template<class T>
class CyclicStore
{
public:
    using base_type = T;
    using index_type = index_t;
    using status_type = Status;

    /*! \brief constructor
     * \param[in] capacity buffer capacity
     */
    HOSTQUALIFIER INLINEQUALIFIER
    explicit CyclicStore(const index_type capacity) noexcept :
        store_(nullptr),
        capacity_(capacity),
        current_(nullptr),
        status_(status_type::not_initialized()),
        is_copy_(false)
    {
        if(capacity != 0)
        {
            const auto total_bytes = sizeof(T) * capacity_;

            if(helpers::available_gpu_memory() >= total_bytes && capacity_ > 0)
            {
                cudaMalloc(&store_, sizeof(T) * capacity_);
                current_ = new index_type(0);
                status_ = status_type::none();
            }
            else
            {
                status_ += status_type::out_of_memory();
            }
        }
        else
        {
            status_ += status_type::invalid_configuration();
        }
    }

    /*! \brief copy-constructor (shallow)
     *  \param[in] object to be copied
     */
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    CyclicStore(const CyclicStore& o) noexcept :
        store_(o.store_),
        capacity_(o.capacity_),
        current_(o.current_),
        status_(o.status_),
        is_copy_(true)
    {}

    /*! \brief move-constructor
     *  \param[in] object to be moved
     */
    HOSTQUALIFIER INLINEQUALIFIER
    CyclicStore(CyclicStore&& o) noexcept :
        store_(std::move(o.store_)),
        capacity_(std::move(o.capacity_)),
        current_(std::move(o.current_)),
        status_(std::move(o.status_)),
        is_copy_(std::move(o.is_copy_))
    {
        o.is_copy_ = true;
    }

    #ifndef __CUDA_ARCH__
    /*! \brief destructor
     */
    HOSTQUALIFIER INLINEQUALIFIER
    ~CyclicStore() noexcept
    {
        if(!is_copy_)
        {
            if(store_ != nullptr) cudaFree(store_);
            delete current_;
        }
    }
    #endif

    /*! \brief atomically fetches the next slot in the buffer
     *  \return pointer to the next slot in the buffer
     *  \info \c const on purpose
     */
    HOSTQUALIFIER INLINEQUALIFIER
    T * get() const noexcept
    {
        index_type old;
        index_type val;

        do
        {
            old = *current_;
            val = (old == capacity_ - 1) ? 0 : old + 1;
        }while(!__sync_bool_compare_and_swap(current_, old, val));

        return store_ + old;
    }

    /*! \brief get buffer status
     *  \return status
     */
    HOSTQUALIFIER INLINEQUALIFIER
    status_type status() const noexcept
    {
        return status_;
    }

    /*! \brief get buffer capacity
     *  \return capacity
     */
    HOSTQUALIFIER INLINEQUALIFIER
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
        return capacity_ * sizeof(base_type) + sizeof(index_type);
    }

private:
    base_type * store_; //< actual buffer
    const index_type capacity_; //< buffer capacity
    index_type * current_; //< current active buffer slot
    status_type status_; //< buffer status
    bool is_copy_;

}; // class CyclicStore

/*! \brief key/value storage classes
 */
namespace key_value
{

// forward-declaration of friends
template<class Key, class Value>
class SoAStore;

template<class Key, class Value>
class AoSStore;

namespace detail
{

template<class Key, class Value>
class pair_t
{
public:
    Key key;
    Value value;

    DEVICEQUALIFIER
    constexpr pair_t(const pair_t& pair) noexcept = delete;

private:
    DEVICEQUALIFIER
    constexpr pair_t(const Key& key_, const Value& value_) noexcept :
        key(key_), value(value_)
    {}

    DEVICEQUALIFIER
    constexpr pair_t() noexcept : key(), value()
    {}

    friend AoSStore<Key, Value>;
    friend SoAStore<Key, Value>;
};

template<class Key, class Value>
class pair_ref_t
{
public:
    Key& key;
    Value& value;

private:
    DEVICEQUALIFIER
    constexpr pair_ref_t(Key& key_, Value& value_) noexcept :
        key(key_), value(value_)
    {}

    using NKey = std::remove_const_t<Key>;
    using NValue = std::remove_const_t<Value>;

    friend AoSStore<NKey, NValue>;
    friend SoAStore<NKey, NValue>;
};

template<class Key, class Value>
using pair_const_ref_t = pair_ref_t<const Key, const Value>;

} // namespace detail

/*! \brief key/value store with struct-of-arrays memory layout
 * \tparam Key key type
 * \tparam Value value type
 */
template<class Key, class Value>
class SoAStore
{
public:
    using key_type = Key;
    using value_type = Value;
    using status_type = Status;
    using index_type = index_t;
    using tag = tags::key_value_storage;

    /*! \brief constructor
     * \param[in] capacity number of key/value slots
     */
    HOSTQUALIFIER INLINEQUALIFIER
    explicit SoAStore(const index_type capacity) noexcept :
        status_(Status::not_initialized()),
        capacity_(capacity),
        keys_(nullptr),
        values_(nullptr),
        is_copy_(false)
    {
        if(capacity != 0)
        {
            const auto total_bytes = (((sizeof(key_type) + sizeof(value_type)) *
                capacity) + sizeof(status_type));

            if(helpers::available_gpu_memory() >= total_bytes)
            {
                cudaMalloc(&keys_, sizeof(key_type)*capacity);
                cudaMalloc(&values_, sizeof(value_type)*capacity);

                status_ = status_type::none();
            }
            else
            {
                status_ += status_type::out_of_memory();
            }
        }
        else
        {
            status_ += status_type::invalid_configuration();
        }
    }

    /*! \brief copy-constructor (shallow)
     *  \param[in] object to be copied
     */
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    SoAStore(const SoAStore& o) noexcept :
        status_(o.status_),
        capacity_(o.capacity_),
        keys_(o.keys_),
        values_(o.values_),
        is_copy_(true)
    {}

    /*! \brief move-constructor
     *  \param[in] object to be moved
     */
    HOSTQUALIFIER INLINEQUALIFIER
    SoAStore(SoAStore&& o) noexcept :
        status_(std::move(o.status_)),
        capacity_(std::move(o.capacity_)),
        keys_(std::move(o.keys_)),
        values_(std::move(o.values_)),
        is_copy_(std::move(o.is_copy_))
    {
        o.is_copy_ = true;
    }

    #ifndef __CUDA_ARCH__
    /*! \brief destructor
     */
    HOSTQUALIFIER INLINEQUALIFIER
    ~SoAStore() noexcept
    {
        if(!is_copy_)
        {
            if(keys_   != nullptr) cudaFree(keys_);
            if(values_ != nullptr) cudaFree(values_);
        }
    }
    #endif

    /*! \brief initialize keys
     * \param[in] key initializer key
     * \param[in] stream CUDA stream in which this operation is executed in
     */
    HOSTQUALIFIER INLINEQUALIFIER
    void init_keys(const key_type key, const cudaStream_t stream = 0) noexcept
    {
        if(!status_.has_any())
        {
            helpers::lambda_kernel
            <<<SDIV(capacity_, WARPCORE_BLOCKSIZE), WARPCORE_BLOCKSIZE, 0, stream>>>
            ([=, *this] DEVICEQUALIFIER
            {
                const index_type tid = helpers::global_thread_id();

                if(tid < capacity_)
                {
                    keys_[tid] = key;
                }
            });
        }
    }

    /*! \brief initialize values
     * \param[in] value initializer value
     * \param[in] stream CUDA stream in which this operation is executed in
     */
    HOSTQUALIFIER INLINEQUALIFIER
    void init_values(const value_type value, const cudaStream_t stream = 0) noexcept
    {
        if(!status_.has_any())
        {
            helpers::lambda_kernel
            <<<SDIV(capacity_, WARPCORE_BLOCKSIZE), WARPCORE_BLOCKSIZE, 0, stream>>>
            ([=, *this] DEVICEQUALIFIER
            {
                const index_type tid = helpers::global_thread_id();

                if(tid < capacity_)
                {
                    values_[tid] = value;
                }
            });
        }
    }

    /*! \brief initialize key/value pairs
     * \param[in] key initializer key
     * \param[in] value initializer value
     * \param[in] stream CUDA stream in which this operation is executed in
     */
    HOSTQUALIFIER INLINEQUALIFIER
    void init_pairs(
        const key_type key,
        const value_type value,
        const cudaStream_t stream = 0) noexcept
    {
        if(!status_.has_any())
        {
            helpers::lambda_kernel
            <<<SDIV(capacity_, WARPCORE_BLOCKSIZE), WARPCORE_BLOCKSIZE, 0, stream>>>
            ([=, *this] DEVICEQUALIFIER
            {
                const index_type tid = helpers::global_thread_id();

                if(tid < capacity_)
                {
                    keys_[tid] = key;
                    values_[tid] = value;
                }
            });
        }
    }

    /*! \brief accessor
     * \param[in] i index to access
     * \return pair at position \c i
     */
    DEVICEQUALIFIER INLINEQUALIFIER
    detail::pair_ref_t<key_type, value_type> operator[](index_type i) noexcept
    {
        assert(i < capacity_);
        return detail::pair_ref_t<key_type, value_type>{keys_[i], values_[i]};
    }

    /*! \brief const accessor
     * \param[in] i index to access
     * \return pair at position \c i
     */
    DEVICEQUALIFIER INLINEQUALIFIER
    detail::pair_const_ref_t<key_type, value_type> operator[](
        const index_type i) const noexcept
    {
        return detail::pair_const_ref_t<key_type, value_type>{keys_[i], values_[i]};
    }

    /*! \brief get storage status
     * \return status
     */
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    status_type status() const noexcept
    {
        return status_;
    }

    /*! \brief get storage capacity
     * \return capacity
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
        return capacity_ * (sizeof(key_type) + sizeof(value_type));
    }

private:
    status_type status_; //< storage status
    const index_type capacity_; //< storage capacity
    key_type * keys_; //< actual key storage in SoA format
    value_type * values_; //< actual value storage in SoA format
    bool is_copy_; //< indicates if this object is a shallow copy

}; // class SoAStore

/*! \brief key/value store with array-of-structs memory layout
 * \tparam Key key type
 * \tparam Value value type
 */
template<class Key, class Value>
class AoSStore
{
    using pair_t = detail::pair_t<Key, Value>;

public:
    using key_type = Key;
    using value_type = Value;
    using status_type = Status;
    using index_type = index_t;
    using tag = tags::key_value_storage;

    /*! \brief constructor
     * \param[in] capacity number of key/value slots
     */
    HOSTQUALIFIER INLINEQUALIFIER
    explicit AoSStore(const index_type capacity) noexcept :
        status_(status_type::not_initialized()),
        capacity_(capacity),
        pairs_(nullptr),
        is_copy_(false)
    {
        if(capacity != 0)
        {
            const auto total_bytes = sizeof(pair_t) * capacity;

            if(helpers::available_gpu_memory() >= total_bytes)
            {
                cudaMalloc(&pairs_, sizeof(pair_t) * capacity);

                status_ = status_type::none();
            }
            else
            {
                status_ += status_type::out_of_memory();
            }
        }
        else
        {
            status_ += status_type::invalid_configuration();
        }
    }

    /*! \brief copy-constructor (shallow)
     *  \param[in] object to be copied
     */
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    AoSStore(const AoSStore& o) noexcept :
        status_(o.status_),
        capacity_(o.capacity_),
        pairs_(o.pairs_),
        is_copy_(true)
    {}

    /*! \brief move-constructor
     *  \param[in] object to be moved
     */
    HOSTQUALIFIER INLINEQUALIFIER
    AoSStore(AoSStore&& o) noexcept :
        status_(std::move(o.status_)),
        capacity_(std::move(o.capacity_)),
        pairs_(std::move(o.pairs_)),
        is_copy_(std::move(o.is_copy_))
    {
        o.is_copy_ = true;
    }

    #ifndef __CUDA_ARCH__
    /*! \brief destructor
     */
    HOSTQUALIFIER INLINEQUALIFIER
    ~AoSStore() noexcept
    {
        if(!is_copy_)
        {
            if(pairs_ != nullptr) cudaFree(pairs_);
        }
    }
    #endif

    /*! \brief initialize keys
     * \param[in] key initializer key
     * \param[in] stream CUDA stream in which this operation is executed in
     */
    HOSTQUALIFIER INLINEQUALIFIER
    void init_keys(const key_type key, const cudaStream_t stream = 0) noexcept
    {
        if(!status_.has_any())
        {
            helpers::lambda_kernel
            <<<SDIV(capacity_, WARPCORE_BLOCKSIZE), WARPCORE_BLOCKSIZE, 0, stream>>>
            ([=, *this] DEVICEQUALIFIER
            {
                const index_type tid = helpers::global_thread_id();

                if(tid < capacity_)
                {
                    pairs_[tid].key = key;
                }
            });
        }
    }

    /*! \brief initialize values
     * \param[in] value initializer value
     * \param[in] stream CUDA stream in which this operation is executed in
     */
    HOSTQUALIFIER INLINEQUALIFIER
    void init_values(const value_type value, const cudaStream_t stream = 0) noexcept
    {
        if(!status_.has_any())
        {
            helpers::lambda_kernel
            <<<SDIV(capacity_, WARPCORE_BLOCKSIZE), WARPCORE_BLOCKSIZE, 0, stream>>>
            ([=, *this] DEVICEQUALIFIER
            {
                const index_type tid = helpers::global_thread_id();

                if(tid < capacity_)
                {
                    pairs_[tid].value = value;
                }
            });
        }
    }

    /*! \brief initialize key/value pairs
     * \param[in] key initializer key
     * \param[in] value initializer value
     * \param[in] stream CUDA stream in which this operation is executed in
     */
    HOSTQUALIFIER INLINEQUALIFIER
    void init_pairs(
        const key_type key,
        const value_type value,
        const cudaStream_t stream = 0) noexcept
    {
        if(!status_.has_any())
        {
            helpers::lambda_kernel
            <<<SDIV(capacity_, WARPCORE_BLOCKSIZE), WARPCORE_BLOCKSIZE, 0, stream>>>
            ([=, *this] DEVICEQUALIFIER
            {
                const index_type tid = helpers::global_thread_id();

                if(tid < capacity_)
                {
                    pairs_[tid].key = key;
                    pairs_[tid].value = value;
                }
            });
        }
    }

    /*! \brief accessor
     * \param[in] i index to access
     * \return pair at position \c i
     */
    DEVICEQUALIFIER INLINEQUALIFIER
    pair_t& operator[](const index_type i) noexcept
    {
        return pairs_[i];
    }

    /*! \brief const accessor
     * \param[in] i index to access
     * \return pair at position \c i
     */
    DEVICEQUALIFIER INLINEQUALIFIER
    const pair_t& operator[](const index_type i) const noexcept
    {
        return pairs_[i];
    }

    /*! \brief get storage status
     * \return status
     */
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    status_type status() const noexcept
    {
        return status_;
    }

    /*! \brief get storage capacity
     * \return status
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
        return capacity_ * sizeof(pair_t);
    }

private:
    status_type status_; //< storage status
    const index_type capacity_; //< storage capacity
    pair_t * pairs_; //< actual pair storage in AoS format
    bool is_copy_; //< indicates if this object is a shallow copy

}; // class AoSStore

} // namespace key_value

/*! \brief multi-value storage classes
 */
namespace multi_value
{

namespace detail
{
    enum class LinkedListState
    {
        uninitialized = 0,
        initialized   = 1,
        blocking      = 2,
        full          = 3
    };


    template<class Store>
    union Bucket
    {
    private:
        using value_type = typename Store::value_type;
        using info_type =
            packed_types::PackedPair<Store::bucket_index_bits(), Store::bucket_size_bits()>;

        value_type value_;
        info_type info_;

        DEVICEQUALIFIER INLINEQUALIFIER
        constexpr explicit Bucket(
            const info_type info) noexcept : info_{info}
        {}

        DEVICEQUALIFIER INLINEQUALIFIER
        constexpr explicit Bucket(
            const value_type value) noexcept : value_{value}
        {}

    public:
        // FIXME friend
        DEVICEQUALIFIER INLINEQUALIFIER
        constexpr explicit Bucket(
            const index_t previous,
            const index_t bucket_size) noexcept : info_{previous, bucket_size}
        {}

        HOSTDEVICEQUALIFIER INLINEQUALIFIER
        constexpr explicit Bucket() noexcept :
        info_()
        {};

        DEVICEQUALIFIER INLINEQUALIFIER
        Bucket<Store> atomic_exchange_info(const Bucket<Store> bucket) noexcept
        {
            return Bucket<Store>(atomicExch(&info_, bucket.info_));
        }

        DEVICEQUALIFIER INLINEQUALIFIER
        constexpr value_type value() const noexcept
        {
            return value_;
        }

        DEVICEQUALIFIER INLINEQUALIFIER
        constexpr index_t previous() const noexcept
        {
            return info_.first();
        }

        DEVICEQUALIFIER INLINEQUALIFIER
        constexpr index_t bucket_size() const noexcept
        {
            return info_.second();
        }

        DEVICEQUALIFIER INLINEQUALIFIER
        constexpr void value(const value_type& val) noexcept
        {
            value_ = val;
        }

        DEVICEQUALIFIER INLINEQUALIFIER
        constexpr void previous(const index_t prev) noexcept
        {
            info_.first(prev);
        }

        DEVICEQUALIFIER INLINEQUALIFIER
        constexpr void bucket_size(const index_t size) noexcept
        {
            info_.second(size);
        }
    };

    template<class Store>
    class BucketListHandle
    {
        using packed_type = packed_types::PackedQuadruple<
            2,
            Store::bucket_index_bits(),
            Store::value_counter_bits(),
            Store::bucket_size_bits()>;

        HOSTDEVICEQUALIFIER INLINEQUALIFIER
        constexpr explicit BucketListHandle(
            LinkedListState state,
            index_t index,
            index_t counter,
            index_t offset) noexcept : pack_()
        {
            pack_.first(state);
            pack_.second(index);
            pack_.third(counter);
            pack_.fourth(offset);
        };

        HOSTDEVICEQUALIFIER INLINEQUALIFIER
        constexpr explicit BucketListHandle(const packed_type pack) noexcept :
        pack_(pack)
        {};

    public:
        HOSTDEVICEQUALIFIER INLINEQUALIFIER
        constexpr explicit BucketListHandle() noexcept :
        pack_()
        {};

    private:
        DEVICEQUALIFIER INLINEQUALIFIER
        constexpr LinkedListState linked_list_state() const noexcept
        {
            return pack_.template first_as<LinkedListState>();
        }

        DEVICEQUALIFIER INLINEQUALIFIER
        constexpr index_t bucket_index() const noexcept
        {
            return pack_.second();
        }

    public:
        DEVICEQUALIFIER INLINEQUALIFIER
        constexpr index_t value_count() const noexcept
        {
            return pack_.third();
        }

    private:
        DEVICEQUALIFIER INLINEQUALIFIER
        constexpr index_t num_values_tail() const noexcept
        {
            return pack_.fourth();
        }

    public:
        HOSTDEVICEQUALIFIER INLINEQUALIFIER
        static constexpr index_t max_bucket_index() noexcept
        {
            return (index_t{1} << Store::bucket_index_bits()) - 1;
        }

        HOSTDEVICEQUALIFIER INLINEQUALIFIER
        static constexpr index_t max_value_count() noexcept
        {
            return (index_t{1} << Store::value_counter_bits()) - 1;
        }

        HOSTDEVICEQUALIFIER INLINEQUALIFIER
        static constexpr index_t max_bucket_size() noexcept
        {
            return (index_t{1} << Store::bucket_size_bits()) - 1;
        }

    private:
        DEVICEQUALIFIER INLINEQUALIFIER
        constexpr bool is_uninitialized() const noexcept
        {
            return (linked_list_state() == LinkedListState::uninitialized);
        }

        DEVICEQUALIFIER INLINEQUALIFIER
        constexpr bool is_initialized() const noexcept
        {
            return (linked_list_state() == LinkedListState::initialized);
        }

        DEVICEQUALIFIER INLINEQUALIFIER
        constexpr bool is_blocking() const noexcept
        {
            return (linked_list_state() == LinkedListState::blocking);
        }

        DEVICEQUALIFIER INLINEQUALIFIER
        constexpr bool is_full() const noexcept
        {
            return (linked_list_state() == LinkedListState::full);
        }

        HOSTDEVICEQUALIFIER INLINEQUALIFIER
        constexpr bool operator==(
            const BucketListHandle<Store> other) const noexcept
        {
            return pack_ == other.pack_;
        }

        HOSTDEVICEQUALIFIER INLINEQUALIFIER
        constexpr bool operator!=(
            const BucketListHandle<Store> other) const noexcept
        {
            return !(*this == other);
        }

        packed_type pack_;

        DEVICEQUALIFIER INLINEQUALIFIER
        friend BucketListHandle<Store> atomicCAS(
            BucketListHandle<Store> * const address_,
            const BucketListHandle<Store> compare_,
            const BucketListHandle<Store> val_) noexcept
        {
            return BucketListHandle(
                atomicCAS(&(address_->pack_), compare_.pack_, val_.pack_));
        }

        DEVICEQUALIFIER INLINEQUALIFIER
        friend BucketListHandle<Store> atomicExch(
            BucketListHandle<Store> * address_,
            const BucketListHandle<Store> val_) noexcept
        {
            return BucketListHandle(
                atomicExch(&(address_->pack_), val_.pack_));
        }

        friend Store;
    };
} // namespace detail

/*! \brief value store consisting of growing linked buckets of values
 * \tparam Value type to store
 * \tparam BucketIndexBits number of bits used to enumerate bucket IDs
 * \tparam ValueCounterBits number of bits used to count values in a bucket list
 * \tparam bucketSizeBits number of bits used to hold the value capacity of a bucket
 */
template<
    class   Value,
    index_t BucketIndexBits = 32,
    index_t ValueCounterBits = 20,
    index_t BucketSizeBits = 10>
class BucketListStore
{
private:
    static_assert(
        checks::is_valid_value_type<Value>(),
        "Value type must be trivially copyable.");

    static_assert(
        (BucketIndexBits + ValueCounterBits + BucketSizeBits + 2 <= 64),
        "Too many bits for bucket index and value counter and bucket size.");

    using type = BucketListStore<
        Value,
        BucketIndexBits,
        ValueCounterBits,
        BucketSizeBits>;

    friend detail::BucketListHandle<type>;

public:
    using value_type = Value;
    using handle_type = detail::BucketListHandle<type>;
    using index_type = index_t;
    using status_type = Status;
    using bucket_type = detail::Bucket<type>;
    using tag = tags::dynamic_value_storage;

    /*! \brief get number of bits used to enumerate buckets
     * \return number of bits
     */
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    static constexpr index_type bucket_index_bits() noexcept
    {
        return BucketIndexBits;
    };

    /*! \brief get number of bits used to count values in a bucket list
     * \return number of bits
     */
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    static constexpr index_type value_counter_bits() noexcept
    {
        return ValueCounterBits;
    };

    /*! \brief get number of bits used to hold the value capacity of a bucket
     * \return number of bits
     */
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    static constexpr index_type bucket_size_bits() noexcept
    {
        return BucketSizeBits;
    };

private:
    friend bucket_type;

    /*! \brief head bucket identifier
     *  \return identifier
     */
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    static constexpr index_type head() noexcept
    {
        return handle_type::max_bucket_index();
    }

public:
    /*! \brief get uninitialized handle
     *  \return handle
     */
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    static constexpr handle_type uninitialized_handle() noexcept
    {
        return handle_type{detail::LinkedListState::uninitialized, head(), 0, 0};
    }

    /*! \brief get number of values in bucket list
     *  \return value count
     */
    DEVICEQUALIFIER INLINEQUALIFIER
    static constexpr index_type size(const handle_type& handle) noexcept
    {
        return handle.value_count();
    }


    /*! \brief constructor
     * \param[in] max_capacity maximum number of value slots
     * \param[in] bucket_grow_factor factor which determines the growth of each newly allocated bucket
     * \param[in] min_bucket_size value capacity of the first bucket of a bucket list
     * \param[in] max_bucket_size value capacity after which no more growth is allowed for newly allocated buckets
     */
    HOSTQUALIFIER INLINEQUALIFIER
    explicit BucketListStore(
        const index_type max_capacity,
        const float bucket_grow_factor = 1.1,
        const index_type min_bucket_size = 1,
        const index_type max_bucket_size = handle_type::max_bucket_size()) noexcept :
        status_(Status::not_initialized()),
        capacity_(max_capacity),
        bucket_grow_factor_(bucket_grow_factor),
        min_bucket_size_(min_bucket_size),
        max_bucket_size_(max_bucket_size),
        next_free_bucket_(nullptr),
        buckets_(nullptr),
        is_copy_(false)
    {
        if(capacity_ < handle_type::max_bucket_index() &&
            bucket_grow_factor_ >= 1.0 &&
            min_bucket_size_ >= 1 &&
            max_bucket_size_ >= min_bucket_size_ &&
            max_bucket_size_ <= handle_type::max_bucket_size())
        {
            const auto total_bytes =
                sizeof(bucket_type) * capacity_ + sizeof(index_type);

            if(helpers::available_gpu_memory() >= total_bytes)
            {
                cudaMalloc(&buckets_, sizeof(bucket_type) * capacity_);
                cudaMalloc(&next_free_bucket_, sizeof(index_type));

                status_ = status_type::none();
                init();
            }
            else
            {
                status_ += status_type::out_of_memory();
            }
        }
        else
        {
            status_ += status_type::invalid_configuration();
        }
    }

    /*! \brief copy-constructor (shallow)
     *  \param[in] object to be copied
     */
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    BucketListStore(const BucketListStore& o) noexcept :
        status_(o.status_),
        capacity_(o.capacity_),
        bucket_grow_factor_(o.bucket_grow_factor_),
        min_bucket_size_(o.min_bucket_size_),
        max_bucket_size_(o.max_bucket_size_),
        buckets_(o.buckets_),
        next_free_bucket_(o.next_free_bucket_),
        is_copy_(true)
    {}

    /*! \brief move-constructor
     *  \param[in] object to be moved
     */
    HOSTQUALIFIER INLINEQUALIFIER
    BucketListStore(BucketListStore&& o) noexcept :
        status_(std::move(o.status_)),
        capacity_(std::move(o.capacity_)),
        bucket_grow_factor_(std::move(o.bucket_grow_factor_)),
        min_bucket_size_(std::move(o.min_bucket_size_)),
        max_bucket_size_(std::move(o.max_bucket_size_)),
        buckets_(std::move(o.buckets_)),
        next_free_bucket_(std::move(o.next_free_bucket_)),
        is_copy_(std::move(o.is_copy_))
    {
        o.is_copy_ = true;
    }

    #ifndef __CUDA_ARCH__
    /*! \brief destructor
     */
    HOSTQUALIFIER INLINEQUALIFIER
    ~BucketListStore() noexcept
    {
        if(!is_copy_)
        {
            if(buckets_ != nullptr) cudaFree(buckets_);
            if(next_free_bucket_ != nullptr) cudaFree(next_free_bucket_);
        }
    }
    #endif

    /*! \brief (re)initialize storage
     * \param[in] stream CUDA stream in which this operation is executed in
     */
    HOSTQUALIFIER INLINEQUALIFIER
    void init(const cudaStream_t stream = 0) noexcept
    {
        if(!status_.has_not_initialized())
        {
            helpers::lambda_kernel
            <<<SDIV(capacity_, WARPCORE_BLOCKSIZE), WARPCORE_BLOCKSIZE, 0, stream>>>
            ([=, *this] DEVICEQUALIFIER () mutable
            {
                const index_type tid = helpers::global_thread_id();

                if(tid < capacity_)
                {
                    if(tid == 0)
                    {
                        *next_free_bucket_ = 0;
                    }

                    buckets_[tid].previous(head());
                    buckets_[tid].bucket_size(min_bucket_size_);
                }
            });

            status_ = status_type::none();
        }
    }

    /*! \brief append a value to a bucket list
     * \param[in] handle handle to the bucket list
     * \param[in] value value to be inserted
     * \return status
     */
    DEVICEQUALIFIER INLINEQUALIFIER
    status_type append(
        handle_type& handle,
        const value_type& value,
        index_type max_values_per_key) noexcept
    {
        handle_type current_handle =cub::ThreadLoad<cub::LOAD_VOLATILE>(&handle);

        if(current_handle.is_uninitialized())
        {
            // block handle
            const auto old_handle = atomicCAS(
                &handle,
                current_handle,
                handle_type{
                    detail::LinkedListState::blocking,
                    head(),
                    0,
                    0});

            // winner allocates first bucket
            if(old_handle == current_handle)
            {
                const index_type alloc =
                        atomicAdd(next_free_bucket_, min_bucket_size_);

                if(alloc + min_bucket_size_ <= capacity_)
                {
                    buckets_[alloc].value(value);

                    // successfully allocated initial bucket
                    atomicExch(
                    &handle,
                    handle_type{
                        detail::LinkedListState::initialized,
                        alloc,
                        1,
                        1});

                    return Status::none();
                }

                // mark as full
                atomicExch(
                    &handle,
                    handle_type{
                        detail::LinkedListState::full,
                        head(),
                        0,
                        0});

                return status_type::out_of_memory();
            }
        }

        // try to find a slot until there is no more space
        while(true)
        {
            current_handle = cub::ThreadLoad<cub::LOAD_VOLATILE>(&handle);

            if(current_handle.is_blocking())
            {
                //__nanosleep(1000); // why not?
                continue;
            }

            if(current_handle.is_full())
            {
                return status_type::out_of_memory();
            }

            if(current_handle.value_count() == max_values_per_key)
            {
                return status_type::max_values_for_key_reached();
            }

            const auto current_bucket = cub::ThreadLoad<cub::LOAD_VOLATILE>(
                buckets_ + current_handle.bucket_index());

            const auto current_bucket_size =
                (current_handle.value_count() <= min_bucket_size_) ?
                    min_bucket_size_ : current_bucket.bucket_size();

            // if the bucket is already full allocate a new bucket
            if(current_handle.num_values_tail() == current_bucket_size)
            {
                const auto old_handle = atomicCAS(
                    &handle,
                    current_handle,
                    handle_type{
                        detail::LinkedListState::blocking,
                        current_handle.bucket_index(),
                        current_handle.value_count(),
                        current_handle.num_values_tail()});

                // blocking failed -> reload handle
                if(old_handle != current_handle)
                {
                    continue;
                }

                // compute new bucket size
                const index_type new_bucket_size = min(
                    float(max_bucket_size_),
                    ceilf(float(current_bucket_size) * bucket_grow_factor_));

                // get index of next free bucket in pool
                const index_type alloc =
                    atomicAdd(next_free_bucket_, new_bucket_size + 1);

                if(alloc + new_bucket_size + 1 <= capacity_)
                {
                    buckets_[alloc + 1].value(value);

                    const auto old = buckets_[alloc].atomic_exchange_info(
                        bucket_type{current_handle.bucket_index(),
                        new_bucket_size});

                    if(old.bucket_size() != 0)
                    {
                        // bucket allocation successful
                        atomicExch(
                        &handle,
                        handle_type{
                            detail::LinkedListState::initialized,
                            alloc,
                            current_handle.value_count() + 1,
                            1});
                    }

                    return Status::none();
                }
                else
                {
                    // mark as full
                    atomicExch(
                        &handle,
                        handle_type{
                            detail::LinkedListState::full,
                            current_handle.bucket_index(),
                            current_handle.value_count(),
                            current_handle.num_values_tail()});

                    return status_type::out_of_memory();
                }
            }

            const auto old_handle =
                atomicCAS(
                    &handle,
                    current_handle,
                    handle_type{
                    detail::LinkedListState::initialized,
                    current_handle.bucket_index(),
                    current_handle.value_count() + 1,
                    current_handle.num_values_tail() + 1});

            if(old_handle == current_handle)
            {
                const auto i = current_handle.bucket_index();
                const auto j =
                    (current_handle.value_count() + 1 <= min_bucket_size_) ?
                        current_handle.num_values_tail() :
                        current_handle.num_values_tail() + 1;

                buckets_[i + j].value(value);

                return status_type::none();
            }
        }

        return status_type::unknown_error();
    }

    /*! \brief apply a (lambda-)function on each value inside a bucket list
     * \tparam Func function to be executed for each value
     * \param[in] handle handle to the bucket list
     * \param[in] f function which takes the value together whith the index of the value inside the list as parameters
     * \param[in] group cooperative group used for hash table probing
     */
    template<class Func>
    DEVICEQUALIFIER INLINEQUALIFIER
    void for_each(
        Func f, // TODO const
        const handle_type& handle,
        const cg::thread_group& group = cg::this_thread()) const noexcept
    {
        const index_type rank = group.thread_rank();
        const index_type group_size = group.size();
        index_type local_index = rank;

        // return if nothing is to be done
        if(!handle.is_initialized() || handle.bucket_index() == head()) return;

        bucket_type * bucket_ptr = buckets_ + handle.bucket_index();

        const index_type bucket_offset =
            (handle.value_count() <= min_bucket_size_) ? 0 : 1;

        // process first bucket
        while(local_index < handle.num_values_tail())
        {
            f((bucket_ptr + local_index + bucket_offset)->value(), local_index);
            local_index += group_size;
        }

        index_type global_index = local_index;
        local_index -= handle.num_values_tail();

        // while there are more values left, process them, too
        while(global_index < handle.value_count())
        {
            bucket_ptr = buckets_ + bucket_ptr->previous();

            // check if we are at the final bucket
            const bool last =
                (global_index >= (handle.value_count() - min_bucket_size_));
            const auto current_bucket_size =
                last ? min_bucket_size_ : bucket_ptr->bucket_size();
            const index_type bucket_offset =
                last ? 0 : 1;

            // while there are more values to be processed in the current bucket
            while(local_index < current_bucket_size)
            {
                f((bucket_ptr + local_index + bucket_offset)->value(), global_index);

                local_index += group_size;
                global_index += group_size;
            }

            local_index -= bucket_ptr->bucket_size();
        }
    }

    /*! \brief get status
     * \return status
     */
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    status_type status() const noexcept
    {
        return status_;
    }

    /*! \brief get maximum value capacity
     * \return capacity
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
        return capacity_ * sizeof(bucket_type) + sizeof(index_type);
    }

    /*! \brief get load factor
     * \param[in] stream CUDA stream in which this operation is executed in
     * \return load factor
     */
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    float load_factor(const cudaStream_t stream = 0) const noexcept
    {
        index_type load = 0;

        cudaMemcpyAsync(
            &load, next_free_bucket_, sizeof(index_type), D2H, stream);

        cudaStreamSynchronize(stream);

        return float(load) / float(capacity());
    }

     /*! \brief get the number of occupied bytes
     * \param[in] stream CUDA stream in which this operation is executed in
     * \return bytes
     */
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    index_type bytes_occupied(const cudaStream_t stream = 0) const noexcept
    {
         index_type occupied = 0;

         cudaMemcpyAsync(
             &occupied, next_free_bucket_, sizeof(index_type), D2H, stream);

         cudaStreamSynchronize(stream);

         return occupied * sizeof(bucket_type);
    }

    /*! \brief get bucket growth factor
     * \return factor
     */
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    float bucket_grow_factor() const noexcept
    {
        return bucket_grow_factor_;
    }

    /*! \brief get minimum bucket capacity
     * \return capacity
     */
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    index_type min_bucket_size() const noexcept
    {
        return min_bucket_size_;
    }

    /*! \brief get maximum bucket capacity
     * \return capacity
     */
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    index_type max_bucket_size() const noexcept
    {
        return max_bucket_size_;
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
    status_type status_; //< status of the store
    const index_type capacity_; //< value capacity
    const float bucket_grow_factor_; //< grow factor for allocated buckets
    const index_type min_bucket_size_; //< initial bucket size
    const index_type max_bucket_size_; //< bucket size after which no more growth occurs
    bucket_type * buckets_; //< pointer to bucket store
    index_type * next_free_bucket_; //< index of next non-occupied bucket
    bool is_copy_; //< indicates if this object is a shallow copy

}; // class BucketListStore

} // namespace multi_value

} // namespace storage

} // namespace warpcore

#endif /* WARPCORE_STORAGE_CUH */