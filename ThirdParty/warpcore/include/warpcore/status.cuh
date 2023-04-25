#ifndef WARPCORE_STATUS_CUH
#define WARPCORE_STATUS_CUH

namespace warpcore
{

// forward declaration of handlers which need to be friends with Status
namespace status_handlers
{
    template<status_base_t Ignore>
    class ReturnBoolean;
}

/*! \brief status/error indicator
 */
class Status
{

public:
    using base_type = status_base_t;

    static_assert(
        std::is_same<base_type, std::uint32_t>::value ||
        std::is_same<base_type, std::uint64_t>::value,
        "unsupported base type");

    Status() noexcept = default;
    constexpr Status(const Status&) noexcept = default;
    constexpr Status(Status&& s) noexcept = default;

    // TODO needed?
    HOSTDEVICEQUALIFIER
    constexpr operator base_type() noexcept { return status_; }

    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    constexpr base_type base() const noexcept { return status_; }

    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    static constexpr Status none() noexcept { return Status(base_type(0)); }
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    static constexpr Status all() noexcept { return Status(mask); }

    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    static constexpr Status unknown_error() noexcept { return Status(one << 0); }
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    static constexpr Status probing_length_exceeded() noexcept { return Status(one << 1); }
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    static constexpr Status invalid_configuration() noexcept { return Status(one << 2); }
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    static constexpr Status invalid_key() noexcept { return Status(one << 3); }
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    static constexpr Status duplicate_key() noexcept { return Status(one << 4); }
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    static constexpr Status key_not_found() noexcept { return Status(one << 5); }
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    static constexpr Status index_overflow() noexcept { return Status(one << 6); }
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    static constexpr Status out_of_memory() noexcept { return Status(one << 7); }
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    static constexpr Status not_initialized() noexcept { return Status(one << 8); }
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    static constexpr Status dry_run() noexcept { return Status(one << 9); }
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    static constexpr Status invalid_phase_overlap() noexcept { return Status(one << 10); }
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    static constexpr Status max_values_for_key_reached() noexcept { return Status(one << 11); }
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    static constexpr Status invalid_value() noexcept { return Status(one << 12); }

    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    static constexpr Status error_mask() noexcept
    {
        return
            unknown_error() +
            invalid_configuration() +
            index_overflow() +
            out_of_memory() +
            not_initialized() +
            invalid_phase_overlap();
    }

    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    static constexpr Status warning_mask() noexcept
    {
        return all() - error_mask();
    }

    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    constexpr Status get_errors() const noexcept { return Status(status_ & error_mask().status_); }
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    constexpr Status get_warnings() const noexcept { return Status(status_ & warning_mask().status_); }

    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    constexpr bool has_any(const Status& s = all()) const noexcept { return status_ & s.status_; }
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    constexpr bool has_any_errors() const noexcept { return has_any(error_mask()); }
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    constexpr bool has_any_warnings() const noexcept { return has_any(warning_mask()); }
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    constexpr bool has_all(const Status& s = all()) const noexcept { return (status_ & s.status_) == s.status_; }

    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    constexpr bool has_unknown_error() const noexcept { return status_ & unknown_error().status_; }
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    constexpr bool has_probing_length_exceeded() const noexcept { return status_ & probing_length_exceeded().status_; }
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    constexpr bool has_invalid_configuration() const noexcept { return status_ & invalid_configuration().status_; }
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    constexpr bool has_invalid_key() const noexcept { return status_ & invalid_key().status_; }
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    constexpr bool has_duplicate_key() const noexcept { return status_ & duplicate_key().status_; }
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    constexpr bool has_key_not_found() const noexcept { return status_ & key_not_found().status_; }
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    constexpr bool has_index_overflow() const noexcept { return status_ & index_overflow().status_; }
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    constexpr bool has_out_of_memory() const noexcept { return status_ & out_of_memory().status_; }
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    constexpr bool has_not_initialized() const noexcept { return status_ & not_initialized().status_; }
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    constexpr bool has_dry_run() const noexcept { return status_ & dry_run().status_; }
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    constexpr bool has_invalid_phase_overlap() const noexcept { return status_ & invalid_phase_overlap().status_; }
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    constexpr bool has_max_values_for_key_reached() const noexcept { return status_ & max_values_for_key_reached().status_; }
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    constexpr bool has_invalid_value() const noexcept { return status_ & invalid_value().status_; }

    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    constexpr Status& operator=(const Status& a) noexcept
    {
        status_ = a.status_;
        return *this;
    }

    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    constexpr Status operator~() const noexcept
    {
        return Status((~status_) ^ mask);
    }

    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    constexpr Status& operator+=(const Status& a) noexcept
    {
        status_ |= a.status_;
        return *this;
    }

    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    constexpr Status operator+(const Status& a) noexcept
    {
        return Status(status_ | a.status_);
    }

    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    constexpr Status operator-(const Status& a) noexcept
    {
        return Status(status_ - (status_ & a.status_));
    }

    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    constexpr Status& operator-=(const Status& a) noexcept
    {
        status_ -= status_ & a.status_;
        return *this;
    }

    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    constexpr bool operator==(const Status& a) const noexcept
    {
        return status_ == a.status_;
    }

    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    constexpr bool operator!=(const Status& a) const noexcept
    {
        return status_ != a.status_;
    }

    DEVICEQUALIFIER INLINEQUALIFIER
    void atomic_assign(Status s) noexcept
    {
        atomicExch(&status_, s.status_);
    }

    DEVICEQUALIFIER INLINEQUALIFIER
    void atomic_join(Status s) noexcept
    {
        if(!has_all(s))
        {
            atomicOr(&status_, s.status_);
        }
    }

    template<class Group>
    DEVICEQUALIFIER INLINEQUALIFIER
    Status group_shuffle(const Group& group, index_t src) noexcept
    {
        return Status(group.shfl(status_, src));
    }

private:
    HOSTDEVICEQUALIFIER
    explicit constexpr Status(base_type s) noexcept : status_{s} {}

    static constexpr base_type one = 1;
    static constexpr base_type mask = 0xFFF; // INFO change when adding new status

    base_type status_;

    // some handlers need to access a private constructor
    template<base_type> friend class status_handlers::ReturnBoolean;

}; // class Status

template<class OStream>
OStream& operator<<(OStream& os, Status status)
{
    std::vector<std::string> msg;

    if(status.has_duplicate_key())
        msg.push_back("duplicate key");
    if(status.has_unknown_error())
        msg.push_back("unknown error");
    if(status.has_probing_length_exceeded())
        msg.push_back("probing length exceeded");
    if(status.has_invalid_configuration())
        msg.push_back("invalid configuration");
    if(status.has_invalid_key())
        msg.push_back("invalid key");
    if(status.has_key_not_found())
        msg.push_back("key not found");
    if(status.has_index_overflow())
        msg.push_back("index overflow");
    if(status.has_out_of_memory())
        msg.push_back("out of memory");
    if(status.has_not_initialized())
        msg.push_back("not initialized");
    if(status.has_dry_run())
        msg.push_back("dry run");
    if(status.has_invalid_phase_overlap())
        msg.push_back("invalid phase overlap");
    if(status.has_max_values_for_key_reached())
        msg.push_back("max values for key reached");
    if(status.has_invalid_value())
        msg.push_back("invalid value");
    // if(!status.has_any())
    //     msg.push_back("none");

    switch(msg.size())
    {
        case 0: os << "[]"; break;
        case 1: os << "[" << msg[0] << "]"; break;
        default:
            os << "[";

            std::uint64_t i = 0;
            while(i < msg.size() - 1)
            {
                os << msg[i] << ", ";
                i++;
            }

            os << msg[i] << "]";
            break;
    }

    return os;
}

/*! \brief status handlers to use for per key status information
 */
namespace status_handlers
{

/*! \brief do not use per key status handling
 */
class ReturnNothing
{
public:
    using base_type = void;
    using tag = tags::status_handler;

    DEVICEQUALIFIER INLINEQUALIFIER
    static void handle(
        Status /* status */,
        base_type * /* out */,
        index_t /* index */) noexcept
    {
        // no op
    }
}; // class ReturnNothing

/*! \brief get status per key
 */
class ReturnStatus
{
public:
    using base_type = Status;
    using tag = tags::status_handler;

    DEVICEQUALIFIER INLINEQUALIFIER
    static void handle(
        Status status,
        base_type * out,
        index_t index) noexcept
    {
        if(out != nullptr){
            out[index] = status;
        }
    }
}; // class ReturnStatus

/*! \brief get boolean error indicator per key
 */
template<Status::base_type Ignore = Status::none()>
class ReturnBoolean
{
public:
    using base_type = bool;
    using tag = tags::status_handler;

    DEVICEQUALIFIER INLINEQUALIFIER
    static void handle(
        Status status,
        base_type * out,
        index_t index) noexcept
    {
        if(out != nullptr){
            out[index] = status.has_any(~Status(Ignore));
        }
    }
}; // class ReturnBoolean

} // namespace status_handlers

}  // namespace warpcore

#endif /* WARPCORE_STATUS_CUH */
