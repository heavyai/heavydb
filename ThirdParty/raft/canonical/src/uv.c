#include "../include/raft/uv.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include "../include/raft.h"
#include "assert.h"
#include "byte.h"
#include "configuration.h"
#include "entry.h"
#include "heap.h"
#include "snapshot.h"
#include "tracing.h"
#include "uv.h"
#include "uv_encoding.h"
#include "uv_os.h"

/* Set to 1 to enable tracing. */
#if 0
#define tracef(...) Tracef(c->uv->tracer, __VA_ARGS__)
#else
#define tracef(...)
#endif

/* Retry to connect to peer servers every second.
 *
 * TODO: implement an exponential backoff instead.  */
#define CONNECT_RETRY_DELAY 1000

/* Implementation of raft_io->config. */
static int uvInit(struct raft_io *io, raft_id id, const char *address)
{
    struct uv *uv;
    size_t direct_io;
    struct uvMetadata metadata;
    int rv;
    uv = io->impl;
    uv->id = id;

    rv = UvFsCheckDir(uv->dir, io->errmsg);
    if (rv != 0) {
        return rv;
    }

    /* Probe file system capabilities */
    rv = UvFsProbeCapabilities(uv->dir, &direct_io, &uv->async_io, io->errmsg);
    if (rv != 0) {
        return rv;
    }
    uv->direct_io = direct_io != 0;
    uv->block_size = direct_io != 0 ? direct_io : 4096;

    rv = uvMetadataLoad(uv->dir, &metadata, io->errmsg);
    if (rv != 0) {
        return rv;
    }
    uv->metadata = metadata;

    rv = uv->transport->init(uv->transport, id, address);
    if (rv != 0) {
        ErrMsgTransfer(uv->transport->errmsg, io->errmsg, "transport");
        return rv;
    }
    uv->transport->data = uv;

    rv = uv_timer_init(uv->loop, &uv->timer);
    assert(rv == 0); /* This should never fail */
    uv->timer.data = uv;

    return 0;
}

/* Periodic timer callback */
static void uvTickTimerCb(uv_timer_t *timer)
{
    struct uv *uv;
    uv = timer->data;
    if (uv->tick_cb != NULL) {
        uv->tick_cb(uv->io);
    }
}

/* Implementation of raft_io->start. */
static int uvStart(struct raft_io *io,
                   unsigned msecs,
                   raft_io_tick_cb tick_cb,
                   raft_io_recv_cb recv_cb)
{
    struct uv *uv;
    int rv;
    uv = io->impl;
    uv->state = UV__ACTIVE;
    uv->tick_cb = tick_cb;
    uv->recv_cb = recv_cb;
    rv = UvRecvStart(uv);
    if (rv != 0) {
        return rv;
    }
    rv = uv_timer_start(&uv->timer, uvTickTimerCb, msecs, msecs);
    assert(rv == 0);
    return 0;
}

void uvMaybeFireCloseCb(struct uv *uv)
{
    if (!uv->closing) {
        return;
    }

    if (uv->transport->data != NULL) {
        return;
    }
    if (uv->timer.data != NULL) {
        return;
    }
    if (!QUEUE_IS_EMPTY(&uv->append_segments)) {
        return;
    }
    if (!QUEUE_IS_EMPTY(&uv->finalize_reqs)) {
        return;
    }
    if (uv->finalize_work.data != NULL) {
        return;
    }
    if (uv->prepare_inflight != NULL) {
        return;
    }
    if (uv->barrier != NULL) {
        return;
    }
    if (uv->snapshot_put_work.data != NULL) {
        return;
    }
    if (!QUEUE_IS_EMPTY(&uv->snapshot_get_reqs)) {
        return;
    }
    if (!QUEUE_IS_EMPTY(&uv->aborting)) {
        return;
    }

    assert(uv->truncate_work.data == NULL);

    if (uv->close_cb != NULL) {
        uv->close_cb(uv->io);
    }
}

static void uvTickTimerCloseCb(uv_handle_t *handle)
{
    struct uv *uv = handle->data;
    assert(uv->closing);
    uv->timer.data = NULL;
    uvMaybeFireCloseCb(uv);
}

static void uvTransportCloseCb(struct raft_uv_transport *transport)
{
    struct uv *uv = transport->data;
    assert(uv->closing);
    uv->transport->data = NULL;
    uvMaybeFireCloseCb(uv);
}

/* Implementation of raft_io->stop. */
static void uvClose(struct raft_io *io, raft_io_close_cb cb)
{
    struct uv *uv;
    uv = io->impl;
    assert(!uv->closing);
    uv->close_cb = cb;
    uv->closing = true;
    UvSendClose(uv);
    UvRecvClose(uv);
    uvAppendClose(uv);
    if (uv->transport->data != NULL) {
        uv->transport->close(uv->transport, uvTransportCloseCb);
    }
    if (uv->timer.data != NULL) {
        uv_close((uv_handle_t *)&uv->timer, uvTickTimerCloseCb);
    }
    uvMaybeFireCloseCb(uv);
}

/* Filter the given segment list to find the most recent contiguous chunk of
 * closed segments that overlaps with the given snapshot last index. */
static int uvFilterSegments(struct uv *uv,
                            raft_index last_index,
                            const char *snapshot_filename,
                            struct uvSegmentInfo **segments,
                            size_t *n)
{
    struct uvSegmentInfo *segment;
    size_t i; /* First valid closed segment. */
    size_t j; /* Last valid closed segment. */

    /* If there are not segments at all, or only open segments, there's nothing
     * to do. */
    if (*segments == NULL || (*segments)[0].is_open) {
        return 0;
    }

    /* Find the index of the most recent closed segment. */
    for (j = 0; j < *n; j++) {
        segment = &(*segments)[j];
        if (segment->is_open) {
            break;
        }
    }
    assert(j > 0);
    j--;

    segment = &(*segments)[j];
    tracef("most recent closed segment is %s", segment->filename);

    /* If the end index of the last closed segment is lower than the last
     * snapshot index, there might be no entry that we can keep. We return an
     * empty segment list, unless there is at least one open segment, in that
     * case we keep everything hoping that they contain all the entries since
     * the last closed segment (TODO: we should encode the starting entry in the
     * open segment). */
    if (segment->end_index < last_index) {
        if (!(*segments)[*n - 1].is_open) {
            tracef(
                "discarding all closed segments, since most recent is behind "
                "last snapshot");
            raft_free(*segments);
            *segments = NULL;
            *n = 0;
            return 0;
        }
        tracef(
            "most recent closed segment %s is behind last snapshot, "
            "yet there are open segments",
            segment->filename);
    }

    /* Now scan the segments backwards, searching for the longest list of
     * contiguous closed segments. */
    if (j >= 1) {
        for (i = j; i > 0; i--) {
            struct uvSegmentInfo *newer;
            struct uvSegmentInfo *older;
            newer = &(*segments)[i];
            older = &(*segments)[i - 1];
            if (older->end_index != newer->first_index - 1) {
                tracef("discarding non contiguous segment %s", older->filename);
                break;
            }
        }
    } else {
        i = j;
    }

    /* Make sure that the first index of the first valid closed segment is not
     * greater than the snapshot's last index plus one (so there are no
     * missing entries). */
    segment = &(*segments)[i];
    if (segment->first_index > last_index + 1) {
        ErrMsgPrintf(uv->io->errmsg,
                     "closed segment %s is past last snapshot %s",
                     segment->filename, snapshot_filename);
        return RAFT_CORRUPT;
    }

    if (i != 0) {
        size_t new_n = *n - i;
        struct uvSegmentInfo *new_segments;
        new_segments = raft_malloc(new_n * sizeof *new_segments);
        if (new_segments == NULL) {
            return RAFT_NOMEM;
        }
        memcpy(new_segments, &(*segments)[i], new_n * sizeof *new_segments);
        raft_free(*segments);
        *segments = new_segments;
        *n = new_n;
    }

    return 0;
}

/* Load the last snapshot (if any) and all entries contained in all segment
 * files of the data directory. */
static int uvLoadSnapshotAndEntries(struct uv *uv,
                                    struct raft_snapshot **snapshot,
                                    raft_index *start_index,
                                    struct raft_entry *entries[],
                                    size_t *n)
{
    struct uvSnapshotInfo *snapshots;
    struct uvSegmentInfo *segments;
    size_t n_snapshots;
    size_t n_segments;
    int rv;

    *snapshot = NULL;
    *start_index = 1;
    *entries = NULL;
    *n = 0;

    /* List available snapshots and segments. */
    rv = UvList(uv, &snapshots, &n_snapshots, &segments, &n_segments,
                uv->io->errmsg);
    if (rv != 0) {
        goto err;
    }

    /* Load the most recent snapshot, if any. */
    if (snapshots != NULL) {
        char snapshot_filename[UV__FILENAME_LEN];
        *snapshot = HeapMalloc(sizeof **snapshot);
        if (*snapshot == NULL) {
            rv = RAFT_NOMEM;
            goto err;
        }
        rv = UvSnapshotLoad(uv, &snapshots[n_snapshots - 1], *snapshot,
                            uv->io->errmsg);
        if (rv != 0) {
            HeapFree(*snapshot);
            *snapshot = NULL;
            goto err;
        }
        uvSnapshotFilenameOf(&snapshots[n_snapshots - 1], snapshot_filename);
        tracef("most recent snapshot at %lld", (*snapshot)->index);
        HeapFree(snapshots);
        snapshots = NULL;

        /* Update the start index. If there are closed segments on disk let's
         * make sure that the first index of the first closed segment is not
         * greater than the snapshot's last index plus one (so there are no
         * missing entries), and update the start index accordingly. */
        rv = uvFilterSegments(uv, (*snapshot)->index, snapshot_filename,
                              &segments, &n_segments);
        if (rv != 0) {
            goto err;
        }
        if (segments != NULL) {
            if (segments[0].is_open) {
                *start_index = 1;
            } else {
                *start_index = segments[0].first_index;
            }
        } else {
            *start_index = (*snapshot)->index + 1;
        }
    }

    /* Read data from segments, closing any open segments. */
    if (segments != NULL) {
        raft_index last_index;
        rv = uvSegmentLoadAll(uv, *start_index, segments, n_segments, entries,
                              n);
        if (rv != 0) {
            goto err;
        }

        /* Check if all entries that we loaded are actually behind the last
         * snapshot. This can happen if the last closed segment was behind the
         * last snapshot and there were open segments, but the entries in the
         * open segments turned out to be behind the snapshot as well.  */
        last_index = *start_index + *n - 1;
        if (*snapshot != NULL && last_index < (*snapshot)->index) {
            ErrMsgPrintf(uv->io->errmsg,
                         "last entry on disk has index %llu, which is behind "
                         "last snapshot's index %llu",
                         last_index, (*snapshot)->index);
            rv = RAFT_CORRUPT;
            goto err;
        }

        raft_free(segments);
        segments = NULL;
    }

    return 0;

err:
    assert(rv != 0);
    if (*snapshot != NULL) {
        snapshotDestroy(*snapshot);
        *snapshot = NULL;
    }
    if (snapshots != NULL) {
        raft_free(snapshots);
    }
    if (segments != NULL) {
        raft_free(segments);
    }
    if (*entries != NULL) {
        entryBatchesDestroy(*entries, *n);
        *entries = NULL;
        *n = 0;
    }
    return rv;
}

/* Implementation of raft_io->load. */
static int uvLoad(struct raft_io *io,
                  raft_term *term,
                  raft_id *voted_for,
                  struct raft_snapshot **snapshot,
                  raft_index *start_index,
                  struct raft_entry **entries,
                  size_t *n_entries)
{
    struct uv *uv;
    raft_index last_index;
    int rv;
    uv = io->impl;

    *term = uv->metadata.term;
    *voted_for = uv->metadata.voted_for;
    *snapshot = NULL;

    rv =
        uvLoadSnapshotAndEntries(uv, snapshot, start_index, entries, n_entries);
    if (rv != 0) {
        return rv;
    }
    tracef("start index %lld, %zu entries", *start_index, *n_entries);
    if (*snapshot == NULL) {
        tracef("no snapshot");
    }

    last_index = *start_index + *n_entries - 1;

    /* Set the index of the next entry that will be appended. */
    uv->append_next_index = last_index + 1;

    return 0;
}

/* Implementation of raft_io->set_term. */
static int uvSetTerm(struct raft_io *io, const raft_term term)
{
    struct uv *uv;
    int rv;
    uv = io->impl;
    uv->metadata.version++;
    uv->metadata.term = term;
    uv->metadata.voted_for = 0;
    rv = uvMetadataStore(uv, &uv->metadata);
    if (rv != 0) {
        return rv;
    }
    return 0;
}

/* Implementation of raft_io->set_term. */
static int uvSetVote(struct raft_io *io, const raft_id server_id)
{
    struct uv *uv;
    int rv;
    uv = io->impl;
    uv->metadata.version++;
    uv->metadata.voted_for = server_id;
    rv = uvMetadataStore(uv, &uv->metadata);
    if (rv != 0) {
        return rv;
    }
    return 0;
}

/* Implementation of raft_io->bootstrap. */
static int uvBootstrap(struct raft_io *io,
                       const struct raft_configuration *configuration)
{
    struct uv *uv;
    int rv;
    uv = io->impl;

    /* We shouldn't have written anything else yet. */
    if (uv->metadata.term != 0) {
        ErrMsgPrintf(io->errmsg, "metadata contains term %lld",
                     uv->metadata.term);
        return RAFT_CANTBOOTSTRAP;
    }

    /* Write the term */
    rv = uvSetTerm(io, 1);
    if (rv != 0) {
        return rv;
    }

    /* Create the first closed segment file, containing just one entry. */
    rv = uvSegmentCreateFirstClosed(uv, configuration);
    if (rv != 0) {
        return rv;
    }

    return 0;
}

/* Implementation of raft_io->recover. */
static int uvRecover(struct raft_io *io, const struct raft_configuration *conf)
{
    struct uv *uv = io->impl;
    struct raft_snapshot *snapshot;
    raft_index start_index;
    raft_index next_index;
    struct raft_entry *entries;
    size_t n_entries;
    int rv;

    /* Load the current state. This also closes any leftover open segment. */
    rv = uvLoadSnapshotAndEntries(uv, &snapshot, &start_index, &entries,
                                  &n_entries);
    if (rv != 0) {
        return rv;
    }

    /* We don't care about the actual data, just index of the last entry. */
    if (snapshot != NULL) {
        snapshotDestroy(snapshot);
    }
    if (entries != NULL) {
        entryBatchesDestroy(entries, n_entries);
    }

    assert(start_index > 0);
    next_index = start_index + n_entries;

    rv = uvSegmentCreateClosedWithConfiguration(uv, next_index, conf);
    if (rv != 0) {
        return rv;
    }

    return 0;
}

/* Implementation of raft_io->time. */
static raft_time uvTime(struct raft_io *io)
{
    struct uv *uv;
    uv = io->impl;
    return uv_now(uv->loop);
}

/* Implementation of raft_io->random. */
static int uvRandom(struct raft_io *io, int min, int max)
{
    (void)io;
    return min + (abs(rand()) % (max - min));
}

int raft_uv_init(struct raft_io *io,
                 struct uv_loop_s *loop,
                 const char *dir,
                 struct raft_uv_transport *transport)
{
    struct uv *uv;
    void *data;
    int rv;

    assert(io != NULL);
    assert(loop != NULL);
    assert(dir != NULL);
    assert(transport != NULL);

    data = io->data;
    memset(io, 0, sizeof *io);
    io->data = data;

    /* Ensure that the given path doesn't exceed our static buffer limit. */
    if (!UV__DIR_HAS_VALID_LEN(dir)) {
        ErrMsgPrintf(io->errmsg, "directory path too long");
        return RAFT_NAMETOOLONG;
    }

    /* Allocate the raft_io_uv object */
    uv = raft_malloc(sizeof *uv);
    if (uv == NULL) {
        rv = RAFT_NOMEM;
        goto err;
    }
    memset(uv, 0, sizeof(struct uv));

    uv->io = io;
    uv->loop = loop;
    strcpy(uv->dir, dir);
    uv->transport = transport;
    uv->transport->data = NULL;
    uv->tracer = &NoopTracer;
    uv->id = 0; /* Set by raft_io->config() */
    uv->state = UV__PRISTINE;
    uv->errored = false;
    uv->direct_io = false;
    uv->async_io = false;
    uv->segment_size = UV__MAX_SEGMENT_SIZE;
    uv->block_size = 0;
    QUEUE_INIT(&uv->clients);
    QUEUE_INIT(&uv->servers);
    uv->connect_retry_delay = CONNECT_RETRY_DELAY;
    uv->prepare_inflight = NULL;
    QUEUE_INIT(&uv->prepare_reqs);
    QUEUE_INIT(&uv->prepare_pool);
    uv->prepare_next_counter = 1;
    uv->append_next_index = 1;
    QUEUE_INIT(&uv->append_segments);
    QUEUE_INIT(&uv->append_pending_reqs);
    QUEUE_INIT(&uv->append_writing_reqs);
    uv->barrier = NULL;
    QUEUE_INIT(&uv->finalize_reqs);
    uv->finalize_work.data = NULL;
    uv->truncate_work.data = NULL;
    QUEUE_INIT(&uv->snapshot_get_reqs);
    uv->snapshot_put_work.data = NULL;
    uv->timer.data = NULL;
    uv->tick_cb = NULL; /* Set by raft_io->start() */
    uv->recv_cb = NULL; /* Set by raft_io->start() */
    QUEUE_INIT(&uv->aborting);
    uv->closing = false;
    uv->close_cb = NULL;

    /* Set the raft_io implementation. */
    io->version = 1; /* future-proof'ing */
    io->impl = uv;
    io->init = uvInit;
    io->close = uvClose;
    io->start = uvStart;
    io->load = uvLoad;
    io->bootstrap = uvBootstrap;
    io->recover = uvRecover;
    io->set_term = uvSetTerm;
    io->set_vote = uvSetVote;
    io->append = UvAppend;
    io->truncate = UvTruncate;
    io->send = UvSend;
    io->snapshot_put = UvSnapshotPut;
    io->snapshot_get = UvSnapshotGet;
    io->time = uvTime;
    io->random = uvRandom;

    return 0;

err:
    assert(rv != 0);
    if (rv == RAFT_NOMEM) {
        ErrMsgOom(io->errmsg);
    }
    return rv;
}

void raft_uv_close(struct raft_io *io)
{
    struct uv *uv;
    uv = io->impl;
    raft_free(uv);
}

void raft_uv_set_segment_size(struct raft_io *io, size_t size)
{
    struct uv *uv;
    uv = io->impl;
    uv->segment_size = size;
}

void raft_uv_set_block_size(struct raft_io *io, size_t size)
{
    struct uv *uv;
    uv = io->impl;
    uv->block_size = size;
}

void raft_uv_set_connect_retry_delay(struct raft_io *io, unsigned msecs)
{
    struct uv *uv;
    uv = io->impl;
    uv->connect_retry_delay = msecs;
}

void raft_uv_set_tracer(struct raft_io *io, struct raft_tracer *tracer)
{
    struct uv *uv;
    uv = io->impl;
    uv->tracer = tracer;
}

#undef tracef
