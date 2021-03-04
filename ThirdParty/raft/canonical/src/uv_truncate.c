#include <string.h>
#include <unistd.h>

#include "assert.h"
#include "byte.h"
#include "heap.h"
#include "uv.h"
#include "uv_encoding.h"

#if 0
#define tracef(...) Tracef(c->uv->tracer, __VA_ARGS__)
#else
#define tracef(...)
#endif

/* Track a truncate request. */
struct uvTruncate
{
    struct uv *uv;
    struct UvBarrier barrier;
    raft_index index;
    int status;
};

/* Execute a truncate request in a thread. */
static void uvTruncateWorkCb(uv_work_t *work)
{
    struct uvTruncate *truncate = work->data;
    struct uv *uv = truncate->uv;
    struct uvSnapshotInfo *snapshots;
    struct uvSegmentInfo *segments;
    struct uvSegmentInfo *segment;
    size_t n_snapshots;
    size_t n_segments;
    size_t i;
    size_t j;
    char errmsg[RAFT_ERRMSG_BUF_SIZE];
    int rv;

    /* Load all segments on disk. */
    rv = UvList(uv, &snapshots, &n_snapshots, &segments, &n_segments, errmsg);
    if (rv != 0) {
        goto err;
    }
    if (snapshots != NULL) {
        HeapFree(snapshots);
    }
    assert(segments != NULL);

    /* Find the segment that contains the truncate point. */
    segment = NULL; /* Suppress warnings. */
    for (i = 0; i < n_segments; i++) {
        segment = &segments[i];
        if (segment->is_open) {
            continue;
        }
        if (truncate->index >= segment->first_index &&
            truncate->index <= segment->end_index) {
            break;
        }
    }
    assert(i < n_segments);

    /* If the truncate index is not the first of the segment, we need to
     * truncate it. */
    if (truncate->index > segment->first_index) {
        rv = uvSegmentTruncate(uv, segment, truncate->index);
        if (rv != 0) {
            goto err_after_list;
        }
    }

    /* Remove all closed segments past the one containing the truncate index. */
    for (j = i; j < n_segments; j++) {
        segment = &segments[j];
        if (segment->is_open) {
            continue;
        }
        rv = UvFsRemoveFile(uv->dir, segment->filename, errmsg);
        if (rv != 0) {
            tracef("unlink segment %s: %s", segment->filename, errmsg);
            rv = RAFT_IOERR;
            goto err_after_list;
        }
    }
    rv = UvFsSyncDir(uv->dir, errmsg);
    if (rv != 0) {
        tracef("sync data directory: %s", errmsg);
        rv = RAFT_IOERR;
        goto err_after_list;
    }

    HeapFree(segments);
    truncate->status = 0;

    return;

err_after_list:
    HeapFree(segments);
err:
    assert(rv != 0);
    truncate->status = rv;
}

static void uvTruncateAfterWorkCb(uv_work_t *work, int status)
{
    struct uvTruncate *truncate = work->data;
    struct uv *uv = truncate->uv;
    assert(status == 0);
    if (truncate->status != 0) {
        uv->errored = true;
    }
    uv->truncate_work.data = NULL;
    HeapFree(truncate);
    UvUnblock(uv);
}

static void uvTruncateBarrierCb(struct UvBarrier *barrier)
{
    struct uvTruncate *truncate = barrier->data;
    struct uv *uv = truncate->uv;
    int rv;

    /* If we're closing, don't perform truncation at all and abort here. */
    if (uv->closing) {
        HeapFree(truncate);
        return;
    }

    assert(QUEUE_IS_EMPTY(&uv->append_writing_reqs));
    assert(QUEUE_IS_EMPTY(&uv->finalize_reqs));
    assert(uv->finalize_work.data == NULL);
    assert(uv->truncate_work.data == NULL);

    uv->truncate_work.data = truncate;
    rv = uv_queue_work(uv->loop, &uv->truncate_work, uvTruncateWorkCb,
                       uvTruncateAfterWorkCb);
    if (rv != 0) {
        tracef("truncate index %lld: %s", truncate->index, uv_strerror(rv));
        uv->truncate_work.data = NULL;
        uv->errored = true;
    }
}

int UvTruncate(struct raft_io *io, raft_index index)
{
    struct uv *uv;
    struct uvTruncate *truncate;
    int rv;

    uv = io->impl;
    assert(!uv->closing);

    /* We should truncate only entries that we were requested to append in the
     * first place. */
    assert(index > 0);
    assert(index < uv->append_next_index);

    truncate = HeapMalloc(sizeof *truncate);
    if (truncate == NULL) {
        rv = RAFT_NOMEM;
        goto err;
    }
    truncate->uv = uv;
    truncate->index = index;
    truncate->barrier.data = truncate;

    /* Make sure that we wait for any inflight writes to finish and then close
     * the current segment. */
    rv = UvBarrier(uv, index, &truncate->barrier, uvTruncateBarrierCb);
    if (rv != 0) {
        goto err_after_req_alloc;
    }

    return 0;

err_after_req_alloc:
    HeapFree(truncate);
err:
    assert(rv != 0);
    return rv;
}

#undef tracef
