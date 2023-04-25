#ifndef WARPCORE_GPU_ENGINE_CUH
#define WARPCORE_GPU_ENGINE_CUH

namespace warpcore
{

/*! \brief CUDA kernels
 */
namespace kernels
{

template<
    class T,
    T Val = 0>
GLOBALQUALIFIER
void memset(
    T * const arr,
    const index_t num)
{
    const index_t tid = helpers::global_thread_id();

    if(tid < num)
    {
        arr[tid] = Val;
    }
}

template<
    class Func,
    class Core>
GLOBALQUALIFIER
void for_each(
    Func f,
    const Core core)
{
    const index_t tid = helpers::global_thread_id();

    if(tid < core.capacity())
    {
        auto&& pair = core.table_[tid];
        if(core.is_valid_key(pair.key))
        {
            f(pair.key, pair.value);
        }
    }
}

template<
    class Func,
    class Core>
GLOBALQUALIFIER
void for_each_unique_key(
    Func f,
    const Core core)
{
    using index_type = typename Core::index_type;
    using probing_scheme_type = typename Core::probing_scheme_type;

    const index_t tid = helpers::global_thread_id();
    const index_t gid = tid / Core::cg_size();
    const auto group =
        cg::tiled_partition<Core::cg_size()>(cg::this_thread_block());

    if(gid < core.capacity())
    {
        // for valid entry in table check if this entry is the first of its key
        auto search_key = core.table_[gid].key;
        if(core.is_valid_key(search_key))
        {
            probing_scheme_type iter(core.capacity(), core.capacity(), group);

            for(index_type i = iter.begin(search_key, core.seed_); i != iter.end(); i = iter.next())
            {
                const auto table_key = core.table_[i].key;
                const auto hit = (table_key == search_key);
                const auto hit_mask = group.ballot(hit);

                const auto leader = ffs(hit_mask) - 1;

                // check if search_key is the first entry for this key
                if(group.thread_rank() == leader && i == gid)
                {
                    f(table_key);
                }

                if(group.any(hit))
                {
                    return;
                }
            }
        }
    }
}

template<
    class Func,
    class Core,
    class StatusHandler = defaults::status_handler_t>
GLOBALQUALIFIER
void for_each(
    Func f,
    const typename Core::key_type * const keys_in,
    const index_t num_in,
    const Core core,
    const index_t probing_length = defaults::probing_length(),
    typename StatusHandler::base_type * const status_out = nullptr)
{
    const index_t tid = helpers::global_thread_id();
    const index_t gid = tid / Core::cg_size();
    const auto group =
        cg::tiled_partition<Core::cg_size()>(cg::this_thread_block());

    if(gid < num_in)
    {
        index_t num_values;

        const auto status =
            core.for_each(f, keys_in[gid], num_values, group, probing_length);

        if(group.thread_rank() == 0)
        {
            StatusHandler::handle(status, status_out, gid);
        }
    }
}

namespace bloom_filter
{

template<class Core>
GLOBALQUALIFIER
void insert(
    const typename Core::key_type * const keys_in,
    const index_t num_in,
    Core core)
{
    const index_t tid = helpers::global_thread_id();
    const index_t gid = tid / Core::cg_size();
    const auto group =
        cg::tiled_partition<Core::cg_size()>(cg::this_thread_block());

    if(gid < num_in)
    {
        core.insert(keys_in[gid], group);
    }
}


template<class Core>
GLOBALQUALIFIER
void retrieve(
    const typename Core::key_type * const keys_in,
    const index_t num_in,
    typename Core::value_type * const values_out,
    const Core core)
{
    const index_t tid = blockDim.x * blockIdx.x + threadIdx.x;
    const index_t gid = tid / Core::cg_size();
    const auto group =
        cg::tiled_partition<Core::cg_size()>(cg::this_thread_block());

    if(gid < num_in)
    {
        typename Core::value_type value = core.retrieve(keys_in[gid], group);

        if(group.thread_rank() == 0)
        {
            values_out[gid] = value;
        }
    }
}


} // namespace bloom_filter


template<class Core, class StatusHandler = defaults::status_handler_t>
GLOBALQUALIFIER
void insert(
    const typename Core::key_type * const keys_in,
    const index_t num_in,
    Core core,
    const index_t probing_length = defaults::probing_length(),
    typename StatusHandler::base_type * const status_out = nullptr)
{
    const index_t tid = helpers::global_thread_id();
    const index_t gid = tid / Core::cg_size();
    const auto group =
        cg::tiled_partition<Core::cg_size()>(cg::this_thread_block());

    if(gid < num_in)
    {
        const auto status =
            core.insert(keys_in[gid], group, probing_length);

        if(group.thread_rank() == 0)
        {
            StatusHandler::handle(status, status_out, gid);
        }
    }
}

template<class Core, class StatusHandler = defaults::status_handler_t>
GLOBALQUALIFIER
void insert(
    const typename Core::key_type * const keys_in,
    const typename Core::value_type * const values_in,
    const index_t num_in,
    Core core,
    const index_t probing_length = defaults::probing_length(),
    typename StatusHandler::base_type * const status_out = nullptr)
{
    const index_t tid = helpers::global_thread_id();
    const index_t gid = tid / Core::cg_size();
    const auto group =
        cg::tiled_partition<Core::cg_size()>(cg::this_thread_block());

    if(gid < num_in)
    {
        const auto status =
            core.insert(keys_in[gid], values_in[gid], group, probing_length);

        if(group.thread_rank() == 0)
        {
            StatusHandler::handle(status, status_out, gid);
        }
    }
}

template<class Core, class StatusHandler = defaults::status_handler_t>
GLOBALQUALIFIER
void retrieve(
    const typename Core::key_type * const keys_in,
    const index_t num_in,
    typename Core::value_type * const values_out,
    const Core core,
    const index_t probing_length = defaults::probing_length(),
    typename StatusHandler::base_type * const status_out = nullptr)
{
    const index_t tid = helpers::global_thread_id();
    const index_t gid = tid / Core::cg_size();
    const auto group =
        cg::tiled_partition<Core::cg_size()>(cg::this_thread_block());

    if(gid < num_in)
    {
        typename Core::value_type value_out;

        const auto status =
            core.retrieve(keys_in[gid], value_out, group, probing_length);

        if(group.thread_rank() == 0)
        {
            if(!status.has_any())
            {
                values_out[gid] = value_out;
            }

            StatusHandler::handle(status, status_out, gid);
        }
    }
}

/*
template<class Core, class StatusHandler>
GLOBALQUALIFIER
void retrieve(
    const typename Core::key_type * const keys_in,
    const index_t num_in,
    typename Core::key_type * const keys_out,
    typename Core::value_type * const values_out,
    index_t * const num_out,
    const index_t probing_length,
    const Core core,
    typename StatusHandler::base_type * const status_out)
{
    const index_t tid = helpers::global_thread_id();
    const index_t gid = tid / Core::cg_size();
    const auto group =
        cg::tiled_partition<Core::cg_size()>(cg::this_thread_block());

    if(gid < num_in)
    {
        const typename Core::key_type key_in = keys_in[gid];
        typename Core::value_type value_out;

        const auto status =
            core.retrieve(key_in, value_out, group, probing_length);

        if(group.thread_rank() == 0)
        {
            if(!status.has_any())
            {
                const auto i = helpers::atomicAggInc(num_out);
                keys_out[i] = key_in;
                values_out[i] = value_out;
            }

            StatusHandler::handle(status, status_out, gid);
        }
    }
}
*/

template<class Core, class StatusHandler = defaults::status_handler_t>
GLOBALQUALIFIER
void retrieve(
    const typename Core::key_type * const keys_in,
    const index_t num_in,
    const index_t * const begin_offsets_in,
    const index_t * const end_offsets_in,
    typename Core::value_type * const values_out,
    const Core core,
    const index_t probing_length = defaults::probing_length(),
    typename StatusHandler::base_type * const status_out = nullptr)
{
    const index_t tid = helpers::global_thread_id();
    const index_t gid = tid / Core::cg_size();
    const auto group =
        cg::tiled_partition<Core::cg_size()>(cg::this_thread_block());

    using status_type = typename Core::status_type;

    if(gid < num_in)
    {
        index_t num_out;

        auto status = core.retrieve(
            keys_in[gid],
            values_out + begin_offsets_in[gid],
            num_out,
            group,
            probing_length);

        if(group.thread_rank() == 0)
        {
            const auto num_prev =
                end_offsets_in[gid] - begin_offsets_in[gid];

            if(num_prev != num_out)
            {
                //printf("%llu %llu\n", num_prev, num_out);
                core.device_join_status(status_type::invalid_phase_overlap());
                status += status_type::invalid_phase_overlap();
            }

            StatusHandler::handle(status, status_out, gid);
        }
    }
}

template<class Core, class StatusHandler = defaults::status_handler_t>
GLOBALQUALIFIER
void erase(
    const typename Core::key_type * const keys_in,
    const index_t num_in,
    Core core,
    const index_t probing_length = defaults::probing_length(),
    typename StatusHandler::base_type * const status_out = nullptr)
{
    const index_t tid = helpers::global_thread_id();
    const index_t gid = tid / Core::cg_size();
    const auto group =
        cg::tiled_partition<Core::cg_size()>(cg::this_thread_block());

    if(gid < num_in)
    {
        const auto status =
            core.erase(keys_in[gid], group, probing_length);

        if(group.thread_rank() == 0)
        {
            StatusHandler::handle(status, status_out, gid);
        }
    }
}

template<class Core>
GLOBALQUALIFIER
void size(
    index_t * const num_out,
    const Core core)
{
    __shared__ index_t smem;

    const index_t tid = helpers::global_thread_id();
    const auto block = cg::this_thread_block();

    if(tid < core.capacity())
    {
        const bool empty = !core.is_valid_key(core.table_[tid].key);

        if(block.thread_rank() == 0)
        {
            smem = 0;
        }

        block.sync();

        if(!empty)
        {
            const auto active_threads = cg::coalesced_threads();

            if(active_threads.thread_rank() == 0)
            {
                atomicAdd(&smem, active_threads.size());
            }
        }

        block.sync();

        if(block.thread_rank() == 0 && smem != 0)
        {
            atomicAdd(num_out, smem);
        }
    }
}

// for Core = MultiBucketHashTable
template<class Core>
GLOBALQUALIFIER
void num_values(
    index_t * const num_out,
    const Core core)
{
    __shared__ index_t smem;

    const index_t tid = helpers::global_thread_id();
    const auto block = cg::this_thread_block();

    if(tid < core.capacity())
    {
        const bool empty = !core.is_valid_key(core.table_[tid].key);

        if(block.thread_rank() == 0)
        {
            smem = 0;
        }

        block.sync();

        index_t value_count = 0;
        if(!empty)
        {
            const auto bucket = core.table_[tid].value;
            #pragma unroll
            for(int b = 0; b < core.bucket_size(); ++b) {
                const auto& value = bucket[b];
                if(value != core.empty_value())
                    ++value_count;
            }

            // TODO warp reduce
            atomicAdd(&smem, value_count);
        }

        block.sync();

        if(block.thread_rank() == 0 && smem != 0)
        {
            atomicAdd(num_out, smem);
        }
    }
}

template<class Core, class StatusHandler = defaults::status_handler_t>
GLOBALQUALIFIER
void num_values(
    const typename Core::key_type * const keys_in,
    const index_t num_in,
    index_t * const num_out,
    index_t * const num_per_key_out,
    const Core core,
    const index_t probing_length = defaults::probing_length(),
    typename StatusHandler::base_type * const status_out = nullptr)
{
    const index_t tid = helpers::global_thread_id();
    const index_t gid = tid / Core::cg_size();
    const auto group =
        cg::tiled_partition<Core::cg_size()>(cg::this_thread_block());

    if(gid < num_in)
    {
        index_t num = 0;

        const auto status =
            core.num_values(keys_in[gid], num, group, probing_length);

        if(group.thread_rank() == 0)
        {
            if(num_per_key_out != nullptr)
            {
                num_per_key_out[gid] = num;
            }

            if(num != 0)
            {
                atomicAdd(num_out, num);
            }

            StatusHandler::handle(status, status_out, gid);
        }
    }
}

} // namespace kernels

} // namespace warpcore

#endif /* WARPCORE_GPU_ENGINE_CUH */