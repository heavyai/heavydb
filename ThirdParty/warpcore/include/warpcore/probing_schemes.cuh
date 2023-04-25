#ifndef WARPCORE_PROBING_SCHEMES_CUH
#define WARPCORE_PROBING_SCHEMES_CUH

namespace warpcore
{

/*! \brief probing scheme iterators
 */
namespace probing_schemes
{

namespace cg = cooperative_groups;
namespace checks = warpcore::checks;

//TODO add inner (warp-level) probing?

/*! \brief double hashing scheme: \f$hash(k,i) = h_1(k)+i\cdot h_2(k)\f$
 * \tparam Hasher1 first hash function
 * \tparam Hasher1 second hash function
 * \tparam CGSize cooperative group size
 */
template <class Hasher1, class Hasher2, index_t CGSize = 1>
class DoubleHashing
{
    static_assert(
       checks::is_valid_cg_size(CGSize),
        "invalid cooperative group size");

    static_assert(
        std::is_same<
            typename Hasher1::key_type,
            typename Hasher2::key_type>::value,
        "key types of both hashers must be the same");

public:
    using key_type = typename Hasher1::key_type;
    using index_type = index_t;
    using tag = typename std::conditional<
        checks::is_true_permutation_hasher<Hasher1>() &&
        checks::is_true_permutation_hasher<Hasher2>(),
        tags::cycle_free_probing_scheme,
        tags::probing_scheme>::type;

    /*! \brief get cooperative group size
     * \return cooperative group size
     */
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    static constexpr index_type cg_size() noexcept { return CGSize; }

    /*! \brief constructor
     * \param[in] capacity capacity of the underlying hash table
     * \param[in] probing_length number of probing attempts
     * \param[in] group cooperative group
     */
    DEVICEQUALIFIER INLINEQUALIFIER
    explicit DoubleHashing(
        index_type capacity,
        index_type probing_length,
        const cg::thread_block_tile<CGSize>& group) :
        capacity_(capacity),
        probing_length_(SDIV(probing_length, group.size()) * group.size()),
        group_(group)
    {}

    template<class T>
    DEVICEQUALIFIER INLINEQUALIFIER
    index_type begin(T, T) = delete;

    template<class T>
    DEVICEQUALIFIER INLINEQUALIFIER
    index_type begin(T) = delete;

    /*! \brief begin probing sequence
     * \param[in] key key to be probed
     * \param[in] seed random seed
     * \return initial probing index for \c key
     */
    DEVICEQUALIFIER INLINEQUALIFIER
    index_type begin(key_type key, key_type seed = 0) noexcept
    {
        i_ = group_.thread_rank();
        pos_  = Hasher1::hash(key + seed) + group_.thread_rank();
        pos_ = pos_ % capacity_;
        // step size in range [1, capacity-1] * group_size
        base_ = (Hasher2::hash(key + seed + 1) % (capacity_ / group_.size() - 1) + 1) * group_.size();

        return pos_;
    }

    /*! \brief next probing index for \c key
     * \return next probing index for \c key
     */
    DEVICEQUALIFIER INLINEQUALIFIER
    index_type next() noexcept
    {
        i_ += CGSize;
        pos_ = (pos_ + base_) % capacity_;

        return (i_ < probing_length_) ? pos_ : end();
    }

    /*! \brief end specifier of probing sequence
     * \return end specifier
     */
    DEVICEQUALIFIER INLINEQUALIFIER
    static constexpr index_type end() noexcept
    {
        return ~index_type(0);
    }

private:
    const index_type capacity_; //< capacity of the underlying hash table
    const index_type probing_length_; //< max number of probing attempts
    const cg::thread_block_tile<CGSize>& group_; //< cooperative group

    index_type i_; //< current probe count
    index_type pos_; //< current probing position
    index_type base_; //< step size

}; // class DoubleHashing

/*! \brief linear probing scheme: \f$hash(k,i) = h(k)+i\f$
 * \tparam Hasher hash function
 * \tparam CGSize cooperative group size
 */
template <class Hasher, index_t CGSize = 1>
class LinearProbing
{
    static_assert(
        checks::is_valid_cg_size(CGSize),
        "invalid cooperative group size");

public:
    using key_type = typename Hasher::key_type;
    using index_type = index_t;
    using tag = tags::cycle_free_probing_scheme;

    /*! \brief get cooperative group size
     * \return cooperative group size
     */
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    static constexpr index_t cg_size() noexcept { return CGSize; }

    /*! \brief constructor
     * \param[in] capacity capacity of the underlying hash table
     * \param[in] probing_length number of probing attempts
     * \param[in] group cooperative group
     */
    DEVICEQUALIFIER
    explicit LinearProbing(
        index_type capacity,
        index_type probing_length,
        const cg::thread_block_tile<CGSize>& group) :
        capacity_(capacity),
        probing_length_(SDIV(probing_length, group.size()) * group.size()),
        group_(group)
    {}

    template<class T>
    DEVICEQUALIFIER INLINEQUALIFIER
    index_type begin(T, T) = delete;

    template<class T>
    DEVICEQUALIFIER INLINEQUALIFIER
    index_type begin(T) = delete;

    /*! \brief begin probing sequence
     * \param[in] key key to be probed
     * \param[in] seed random seed
     * \return initial probing index for \c key
     */
    DEVICEQUALIFIER INLINEQUALIFIER
    index_type begin(key_type key, key_type seed = 0) noexcept
    {
        i_ = group_.thread_rank();
        pos_ = Hasher::hash(key + seed) + group_.thread_rank();
        pos_ = pos_ % capacity_;

        return pos_;
    }

    /*! \brief next probing index for \c key
     * \return next probing index
     */
    DEVICEQUALIFIER INLINEQUALIFIER
    index_type next() noexcept
    {
        i_ += CGSize;
        pos_ = (pos_ + CGSize) % capacity_;

        return (i_ < probing_length_) ? pos_ : end();
    }

    /*! \brief end specifier of probing sequence
     * \return end specifier
     */
    DEVICEQUALIFIER INLINEQUALIFIER
    static constexpr index_type end() noexcept
    {
        return ~index_type(0);
    }

private:
    const index_type capacity_; //< capacity of the underlying hash table
    const index_type probing_length_; //< number of probing attempts
    const cg::thread_block_tile<CGSize>& group_; //< cooperative group

    index_type i_; //< current probe count
    index_type pos_; //< current probing position

}; // class LinearProbing

/*! \brief quadratic probing scheme: \f$hash(k,i) = h(k)+\frac{1}{2}\cdot i+\frac{1}{2}\cdot i^2\f$
 * \tparam Hasher hash function
 * \tparam CGSize cooperative group size
 */
template <class Hasher, index_t CGSize = 1>
class QuadraticProbing
{
    static_assert(
        checks::is_valid_cg_size(CGSize),
        "invalid cooperative group size");

public:
    using key_type = typename Hasher::key_type;
    using index_type = index_t;
    using tag = tags::probing_scheme;

    /*! \brief get cooperative group size
     * \return cooperative group size
     */
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    static constexpr index_type cg_size() noexcept { return CGSize; }

    /*! \brief constructor
     * \param[in] capacity capacity of the underlying hash table
     * \param[in] probing_length number of probing attempts
     * \param[in] group cooperative group
     */
    DEVICEQUALIFIER
    explicit QuadraticProbing(
        index_type capacity,
        index_type probing_length,
        const cg::thread_block_tile<CGSize>& group) :
        capacity_(capacity),
        probing_length_(SDIV(probing_length, group.size()) * group.size()),
        group_(group)
    {}

    template<class T>
    DEVICEQUALIFIER INLINEQUALIFIER
    index_type begin(T, T) = delete;

    template<class T>
    DEVICEQUALIFIER INLINEQUALIFIER
    index_type begin(T) = delete;

    /*! \brief begin probing sequence
     * \param[in] key key to be probed
     * \param[in] seed random seed
     * \return initial probing index for \c key
     */
    DEVICEQUALIFIER INLINEQUALIFIER
    index_type begin(key_type key, key_type seed = 0) noexcept
    {
        i_ = group_.thread_rank();
        step_ = 1;
        pos_ = Hasher::hash(key + seed) + group_.thread_rank();
        pos_ = pos_ % capacity_;

        return pos_;
    }

    /*! \brief next probing index for \c key
     * \return next probing index
     */
    DEVICEQUALIFIER INLINEQUALIFIER
    index_type next() noexcept
    {
        i_ += CGSize;
        pos_ = (pos_ + step_) % capacity_;
        ++step_;

        return (i_ < probing_length_) ? pos_ : end();
    }

    /*! \brief end specifier of probing sequence
     * \return end specifier
     */
    DEVICEQUALIFIER INLINEQUALIFIER
    static constexpr index_type end() noexcept
    {
        return ~index_type(0);
    }

private:
    const index_type capacity_; //< capacity of the underlying hash table
    const index_type probing_length_; //< number of probing attempts
    const cg::thread_block_tile<CGSize>& group_; //< cooperative group

    index_type i_; //< current probe count
    index_type pos_; //< current probing position
    index_type step_; //< current step size

}; // class QuadraticProbing

} // namespace probing_schemes

} // namespace warpcore

#endif /* WARPCORE_PROBING_SCHEMES_CUH */