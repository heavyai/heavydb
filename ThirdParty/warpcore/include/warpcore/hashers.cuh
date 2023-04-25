#ifndef WARPCORE_HASHERS_CUH
#define WARPCORE_HASHERS_CUH

namespace warpcore
{

/*! \brief hash functions
*/
namespace hashers
{

/*! \brief hash function proposed by NVIDIA
*/
class NvidiaHash
{

public:
    using key_type = std::uint32_t;
    using hash_type = std::uint32_t;
    using tag = tags::hasher;

    /*! \brief deleted hash function for types other than explicitly defined
     * \tparam T key type
     */
    template<class T>
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    static hash_type hash(T) = delete;

    /*! \brief hash function
     * \param[in] x key to be hashed
     * \return hash of \c x
     */
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    static hash_type hash(key_type x) noexcept
    {
        x = (x + 0x7ed55d16) + (x << 12);
        x = (x ^ 0xc761c23c) ^ (x >> 19);
        x = (x + 0x165667b1) + (x <<  5);
        x = (x + 0xd3a2646c) ^ (x <<  9);
        x = (x + 0xfd7046c5) + (x <<  3);
        x = (x ^ 0xb55a4f09) ^ (x >> 16);

        return x;
    }

}; // class NvidiaHash

/*! \brief hash function proposed by Mueller
 */
class MuellerHash
{

public:
    using key_type = std::uint32_t;
    using hash_type = std::uint32_t;
    using tag = tags::true_permutation_hasher;

    /*! \brief deleted hash function for types other than explicitly defined
     * \tparam T key type
     */
    template<class T>
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    static hash_type hash(T) = delete;

    /*! \brief hash function
     * \param[in] x key to be hashed
     * \return hash of \c x
     */
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    static hash_type hash(key_type x) noexcept
    {
        x = ((x >> 16) ^ x) * 0x45d9f3b;
        x = ((x >> 16) ^ x) * 0x45d9f3b;
        x = ((x >> 16) ^ x);

        return x;
    }

}; // class MuellerHash


/*! \brief murmur integer finalizer
 * \tparam K key type (\c std::uint32_t or std::uint64_t)
 */
template<class K>
class MurmurHash
{

public:
    using key_type = K;
    using hash_type = K;
    using tag = tags::true_permutation_hasher;

    /*! \brief hash function
     * \tparam T key type
     * \param[in] x key to be hashed
     * \return hash of \c x
     */
    template<class T>
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    static T hash(T x) noexcept
    {
        static_assert(
            std::is_same<T, key_type>::value,
            "invalid key type");

        return hash_(x);
    }

private:
    /*! \brief hash function
     * \param[in] x key to be hashed
     * \return hash of \c x
     */
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    static std::uint32_t hash_(std::uint32_t x) noexcept
    {
        x ^= x >> 16;
        x *= 0x85ebca6b;
        x ^= x >> 13;
        x *= 0xc2b2ae35;
        x ^= x >> 16;
        return x;
    }

    /*! \brief hash function
     * \param[in] x key to be hashed
     * \return hash of \c x
    */
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    static std::uint64_t hash_(std::uint64_t x) noexcept
    {
        x ^= x >> 33;
        x *= 0xff51afd7ed558ccd;
        x ^= x >> 33;
        x *= 0xc4ceb9fe1a85ec53;
        x ^= x >> 33;
        return x;
    }

}; // class MurmurHash

/*! \brief identity hash
 * \tparam K key type
 * \tparam H hash type
 */
template<class K, class H = std::uint32_t>
class IdentityMap
{

public:
    using key_type  = K;
    using hash_type = H;
    using tag = tags::true_permutation_hasher;

    static_assert(
        std::is_same<hash_type, std::uint32_t>::value ||
        std::is_same<hash_type, std::uint64_t>::value,
        "invalid hash type");

    static_assert(
        std::is_convertible<key_type, hash_type>::value,
        "key type not convertible to hash type");

    /*! \brief hash function
     * \param[in] x key to be hashed
     * \return hash of \c x
     */
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    static hash_type hash(key_type x) noexcept
    {
        return hash_type{x};
    }

}; // class IdentityMap

} // namespace hashers

}  // namespace warpcore

#endif /* WARPCORE_HASHERS_CUH */