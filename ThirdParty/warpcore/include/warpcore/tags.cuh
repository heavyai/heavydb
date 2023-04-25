#ifndef WARPCORE_TAGS_CUH
#define WARPCORE_TAGS_CUH

namespace warpcore
{

/*! \brief type tags
 */
namespace tags
{

    struct hasher {};
    struct true_permutation_hasher {};
    struct probing_scheme {};
    struct cycle_free_probing_scheme {};
    struct key_value_storage {};
    struct static_value_storage {};
    struct dynamic_value_storage {};
    struct status_handler {};

} // namespace tags

} // namespace warpcore

#endif /* WARPCORE_TAGS_CUH */