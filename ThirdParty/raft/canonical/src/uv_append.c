#include "assert.h"
#include "byte.h"
#include "heap.h"
#include "queue.h"
#include "uv.h"
#include "uv_encoding.h"
#include "uv_writer.h"

/* The happy path for an append request is:
 *
 * - If there is a current segment and it is has enough spare capacity to hold
 *   the entries in the request, then queue the request, linking it to the
 *   current segment.
 *
 * - If there is no current segment, or it hasn't enough spare capacity to hold
 *   the entries in the request, then request a new open segment to be prepared,
 *   queue the request and link it to the newly requested segment.
 *
 * - Wait for any pending write against the current segment to complete, and
 *   also for the prepare request if we asked for a new segment. Also wait for
 *   any in progress barrier to be removed.
 *
 * - Submit a write request for the entries in this append request. The write
 *   request might contain other append requests targeted to the current segment
 *   that might have accumulated in the meantime, if we have been waiting for a
 *   segment to be prepared, or for the previous write to complete or for a
 *   barrier to be removed.
 *
 * - Wait for the write request to finish and fire the append request's
 *   callback.
 *
 * Possible failure modes are:
 *
 * - The request to prepare a new segment fails.
 * - The write request fails.
 * - The request to finalize a new segment fails to be submitted.
 *
 * In all these cases we mark the instance as errored and fire the relevant
 * callbacks.
 **/

/* An open segment being written or waiting to be written. */
struct uvAliveSegment
{
    struct uv *uv;                  /* Our writer */
    struct uvPrepare prepare;       /* Prepare segment file request */
    struct UvWriter writer;         /* Writer to perform async I/O */
    struct UvWriterReq write;       /* Write request */
    unsigned long long counter;     /* Open segment counter */
    raft_index first_index;         /* Index of the first entry written */
    raft_index pending_last_index;  /* Index of the last entry written */
    size_t size;                    /* Total number of bytes used */
    unsigned next_block;            /* Next segment block to write */
    struct uvSegmentBuffer pending; /* Buffer for data yet to be written */
    uv_buf_t buf;                   /* Write buffer for current write */
    raft_index last_index;          /* Last entry actually written */
    size_t written;                 /* Number of bytes actually written */
    queue queue;                    /* Segment queue */
    struct UvBarrier *barrier;      /* Barrier waiting on this segment */
    bool finalize;                  /* Finalize the segment after writing */
};

struct uvAppend
{
    struct raft_io_append *req;       /* User request */
    const struct raft_entry *entries; /* Entries to write */
    unsigned n;                       /* Number of entries */
    struct uvAliveSegment *segment;   /* Segment to write to */
    queue queue;
};

static void uvAliveSegmentWriterCloseCb(struct UvWriter *writer)
{
    struct uvAliveSegment *segment = writer->data;
    struct uv *uv = segment->uv;
    uvSegmentBufferClose(&segment->pending);
    HeapFree(segment);
    uvMaybeFireCloseCb(uv);
}

/* Submit a request to close the current open segment. */
static void uvAliveSegmentFinalize(struct uvAliveSegment *s)
{
    struct uv *uv = s->uv;
    int rv;

    rv = UvFinalize(uv, s->counter, s->written, s->first_index, s->last_index);
    if (rv != 0) {
        uv->errored = true;
        /* We failed to submit the finalize request, but let's still close the
         * file handle and release the segment memory. */
    }

    QUEUE_REMOVE(&s->queue);
    UvWriterClose(&s->writer, uvAliveSegmentWriterCloseCb);
}

/* Flush the append requests in the given queue, firing their callbacks with the
 * given status. */
static void uvAppendFinishRequestsInQueue(struct uv *uv, queue *q, int status)
{
    queue queue_copy;
    struct uvAppend *append;
    QUEUE_INIT(&queue_copy);
    while (!QUEUE_IS_EMPTY(q)) {
        queue *head;
        head = QUEUE_HEAD(q);
        append = QUEUE_DATA(head, struct uvAppend, queue);
        /* Rollback the append next index if the result was unsuccessful. */
        if (status != 0) {
            uv->append_next_index -= append->n;
        }
        QUEUE_REMOVE(head);
        QUEUE_PUSH(&queue_copy, head);
    }
    while (!QUEUE_IS_EMPTY(&queue_copy)) {
        queue *head;
        struct raft_io_append *req;
        head = QUEUE_HEAD(&queue_copy);
        append = QUEUE_DATA(head, struct uvAppend, queue);
        QUEUE_REMOVE(head);
        req = append->req;
        HeapFree(append);
        req->cb(req, status);
    }
}

/* Flush the append requests in the writing queue, firing their callbacks with
 * the given status. */
static void uvAppendFinishWritingRequests(struct uv *uv, int status)
{
    uvAppendFinishRequestsInQueue(uv, &uv->append_writing_reqs, status);
}

/* Flush the append requests in the pending queue, firing their callbacks with
 * the given status. */
static void uvAppendFinishPendingRequests(struct uv *uv, int status)
{
    uvAppendFinishRequestsInQueue(uv, &uv->append_pending_reqs, status);
}

/* Return the segment currently being written, or NULL when no segment has been
 * written yet. */
static struct uvAliveSegment *uvGetCurrentAliveSegment(struct uv *uv)
{
    queue *head;
    if (QUEUE_IS_EMPTY(&uv->append_segments)) {
        return NULL;
    }
    head = QUEUE_HEAD(&uv->append_segments);
    return QUEUE_DATA(head, struct uvAliveSegment, queue);
}

/* Extend the segment's write buffer by encoding the entries in the given
 * request into it. IOW, previous data in the write buffer will be retained, and
 * data for these new entries will be appended. */
static int uvAliveSegmentEncodeEntriesToWriteBuf(struct uvAliveSegment *segment,
                                                 struct uvAppend *append)
{
    int rv;
    assert(append->segment == segment);

    /* If this is the very first write to the segment, we need to include the
     * format version */
    if (segment->pending.n == 0 && segment->next_block == 0) {
        rv = uvSegmentBufferFormat(&segment->pending);
        if (rv != 0) {
            return rv;
        }
    }

    rv = uvSegmentBufferAppend(&segment->pending, append->entries, append->n);
    if (rv != 0) {
        return rv;
    }

    segment->pending_last_index += append->n;

    return 0;
}

static int uvAppendMaybeStart(struct uv *uv);
static void uvAliveSegmentWriteCb(struct UvWriterReq *write, const int status)
{
    struct uvAliveSegment *s = write->data;
    struct uv *uv = s->uv;
    unsigned n_blocks;
    int rv;

    assert(uv->state != UV__CLOSED);

    assert(s->buf.len % uv->block_size == 0);
    assert(s->buf.len >= uv->block_size);

    /* Check if the write was successful. */
    if (status != 0) {
        Tracef(uv->tracer, "write: %s", uv->io->errmsg);
        uv->errored = true;
        goto out;
    }

    s->written = s->next_block * uv->block_size + s->pending.n;
    s->last_index = s->pending_last_index;

    /* Update our write markers.
     *
     * We have four cases:
     *
     * - The data fit completely in the leftover space of the first block that
     *   we wrote and there is more space left. In this case we just keep the
     *   scheduled marker unchanged.
     *
     * - The data fit completely in the leftover space of the first block that
     *   we wrote and there is no space left. In this case we advance the
     *   current block counter, reset the first write block and set the
     *   scheduled marker to 0.
     *
     * - The data did not fit completely in the leftover space of the first
     *   block that we wrote, so we wrote more than one block. The last block
     *   that we wrote was not filled completely and has leftover space. In this
     *   case we advance the current block counter and copy the memory used for
     *   the last block to the head of the write arena list, updating the
     *   scheduled marker accordingly.
     *
     * - The data did not fit completely in the leftover space of the first
     *   block that we wrote, so we wrote more than one block. The last block
     *   that we wrote was filled exactly and has no leftover space. In this
     *   case we advance the current block counter, reset the first buffer and
     *   set the scheduled marker to 0.
     */
    n_blocks =
        (unsigned)(s->buf.len / uv->block_size); /* Number of blocks written. */
    if (s->pending.n < uv->block_size) {
        /* Nothing to do */
        assert(n_blocks == 1);
    } else if (s->pending.n == uv->block_size) {
        assert(n_blocks == 1);
        s->next_block++;
        uvSegmentBufferReset(&s->pending, 0);
    } else {
        assert(s->pending.n > uv->block_size);
        assert(s->buf.len > uv->block_size);

        if (s->pending.n % uv->block_size > 0) {
            s->next_block += n_blocks - 1;
            uvSegmentBufferReset(&s->pending, n_blocks - 1);
        } else {
            s->next_block += n_blocks;
            uvSegmentBufferReset(&s->pending, 0);
        }
    }

out:
    /* Fire the callbacks of all requests that were fulfilled with this
     * write. */
    uvAppendFinishWritingRequests(uv, status);

    /* During the closing sequence we should have already canceled all pending
     * request. */
    if (uv->closing) {
        assert(QUEUE_IS_EMPTY(&uv->append_pending_reqs));
        assert(s->finalize);
        uvAliveSegmentFinalize(s);
        return;
    }

    /* Possibly process waiting requests. */
    if (!QUEUE_IS_EMPTY(&uv->append_pending_reqs)) {
        rv = uvAppendMaybeStart(uv);
        if (rv != 0) {
            uv->errored = true;
        }
    } else if (s->finalize) {
        /* If there are no more append_pending_reqs, this segment
         * must be finalized here in case we don't receive AppendEntries
         * RPCs anymore (could happen during a Snapshot install, causing
         * the BarrierCb to never fire) */
        uvAliveSegmentFinalize(s);
    }
}

/* Submit a file write request to append the entries encoded in the write buffer
 * of the given segment. */
static int uvAliveSegmentWrite(struct uvAliveSegment *s)
{
    int rv;
    assert(s->counter != 0);
    assert(s->pending.n > 0);
    uvSegmentBufferFinalize(&s->pending, &s->buf);
    rv = UvWriterSubmit(&s->writer, &s->write, &s->buf, 1,
                        s->next_block * s->uv->block_size,
                        uvAliveSegmentWriteCb);
    if (rv != 0) {
        return rv;
    }
    return 0;
}

/* Start writing all pending append requests for the current segment, unless we
 * are already writing, or the segment itself has not yet been prepared or we
 * are blocked on a barrier. If there are no more requests targeted at the
 * current segment, make sure it's marked to be finalize and try with the next
 * segment. */
static int uvAppendMaybeStart(struct uv *uv)
{
    struct uvAliveSegment *segment;
    struct uvAppend *append;
    unsigned n_reqs;
    queue *head;
    queue q;
    int rv;

    assert(!uv->closing);
    assert(!QUEUE_IS_EMPTY(&uv->append_pending_reqs));

    /* If we are already writing, let's wait. */
    if (!QUEUE_IS_EMPTY(&uv->append_writing_reqs)) {
        return 0;
    }

start:
    segment = uvGetCurrentAliveSegment(uv);
    assert(segment != NULL);
    /* If the preparer isn't done yet, let's wait. */
    if (segment->counter == 0) {
        return 0;
    }

    /* If there's a barrier in progress, and it's not waiting for this segment
     * to be finalized, let's wait. */
    if (uv->barrier != NULL && segment->barrier != uv->barrier) {
        return 0;
    }

    /* If there's no barrier in progress and this segment is marked with a
     * barrier, it means that this was a pending barrier, which we can become
     * the current barrier now. */
    if (uv->barrier == NULL && segment->barrier != NULL) {
        uv->barrier = segment->barrier;
    }

    /* Let's add to the segment's write buffer all pending requests targeted to
     * this segment. */
    QUEUE_INIT(&q);

    n_reqs = 0;
    while (!QUEUE_IS_EMPTY(&uv->append_pending_reqs)) {
        head = QUEUE_HEAD(&uv->append_pending_reqs);
        append = QUEUE_DATA(head, struct uvAppend, queue);
        assert(append->segment != NULL);
        if (append->segment != segment) {
            break; /* Not targeted to this segment */
        }
        QUEUE_REMOVE(head);
        QUEUE_PUSH(&q, head);
        n_reqs++;
        rv = uvAliveSegmentEncodeEntriesToWriteBuf(segment, append);
        if (rv != 0) {
            goto err;
        }
    }

    /* If we have no more requests for this segment, let's check if it has been
     * marked for closing, and in that case finalize it and possibly trigger a
     * write against the next segment (unless there is a truncate request, in
     * that case we need to wait for it). Otherwise it must mean we have
     * exhausted the queue of pending append requests. */
    if (n_reqs == 0) {
        assert(QUEUE_IS_EMPTY(&uv->append_writing_reqs));
        if (segment->finalize) {
            uvAliveSegmentFinalize(segment);
            if (!QUEUE_IS_EMPTY(&uv->append_pending_reqs)) {
                goto start;
            }
        }
        assert(QUEUE_IS_EMPTY(&uv->append_pending_reqs));
        return 0;
    }

    while (!QUEUE_IS_EMPTY(&q)) {
        head = QUEUE_HEAD(&q);
        QUEUE_REMOVE(head);
        QUEUE_PUSH(&uv->append_writing_reqs, head);
    }

    rv = uvAliveSegmentWrite(segment);
    if (rv != 0) {
        goto err;
    }

    return 0;

err:
    assert(rv != 0);
    return rv;
}

/* Invoked when a newly added open segment becomes ready for writing, after the
 * associated UvPrepare request completes (either synchronously or
 * asynchronously). */
static int uvAliveSegmentReady(struct uv *uv,
                               uv_file fd,
                               uvCounter counter,
                               struct uvAliveSegment *segment)
{
    int rv;
    rv = UvWriterInit(&segment->writer, uv->loop, fd, uv->direct_io,
                      uv->async_io, 1, uv->io->errmsg);
    if (rv != 0) {
        ErrMsgWrapf(uv->io->errmsg, "setup writer for open-%llu", counter);
        return rv;
    }
    segment->counter = counter;
    return 0;
}

static void uvAliveSegmentPrepareCb(struct uvPrepare *req, int status)
{
    struct uvAliveSegment *segment = req->data;
    struct uv *uv = segment->uv;
    int rv;

    assert(segment->counter == 0);
    assert(segment->written == 0);

    /* If we have been closed, let's discard the segment. */
    if (uv->closing) {
        QUEUE_REMOVE(&segment->queue);
        assert(status == RAFT_CANCELED); /* UvPrepare cancels pending reqs */
        uvSegmentBufferClose(&segment->pending);
        HeapFree(segment);
        return;
    }

    if (status != 0) {
        rv = status;
        goto err;
    }

    assert(req->counter > 0);
    assert(req->fd >= 0);

    /* There must be pending appends that were waiting for this prepare
     * requests. */
    assert(!QUEUE_IS_EMPTY(&uv->append_pending_reqs));

    rv = uvAliveSegmentReady(uv, req->fd, req->counter, segment);
    if (rv != 0) {
        goto err;
    }

    rv = uvAppendMaybeStart(uv);
    if (rv != 0) {
        goto err;
    }

    return;

err:
    QUEUE_REMOVE(&segment->queue);
    HeapFree(segment);
    uv->errored = true;
    uvAppendFinishPendingRequests(uv, rv);
}

/* Initialize a new open segment object. */
static void uvAliveSegmentInit(struct uvAliveSegment *s, struct uv *uv)
{
    s->uv = uv;
    s->prepare.data = s;
    s->writer.data = s;
    s->write.data = s;
    s->counter = 0;
    s->first_index = uv->append_next_index;
    s->pending_last_index = s->first_index - 1;
    s->last_index = 0;
    s->size = sizeof(uint64_t) /* Format version */;
    s->next_block = 0;
    uvSegmentBufferInit(&s->pending, uv->block_size);
    s->written = 0;
    s->barrier = NULL;
    s->finalize = false;
}

/* Add a new active open segment, since the append request being submitted does
 * not fit in the last segment we scheduled writes for, or no segment had been
 * previously requested at all. */
static int uvAppendPushAliveSegment(struct uv *uv)
{
    struct uvAliveSegment *segment;
    uv_file fd;
    uvCounter counter;
    int rv;

    segment = HeapMalloc(sizeof *segment);
    if (segment == NULL) {
        rv = RAFT_NOMEM;
        goto err;
    }
    uvAliveSegmentInit(segment, uv);

    QUEUE_PUSH(&uv->append_segments, &segment->queue);

    rv = UvPrepare(uv, &fd, &counter, &segment->prepare,
                   uvAliveSegmentPrepareCb);
    if (rv != 0) {
        goto err_after_alloc;
    }

    /* If we've been returned a ready prepared segment right away, start writing
     * to it immediately. */
    if (fd != -1) {
        rv = uvAliveSegmentReady(uv, fd, counter, segment);
        if (rv != 0) {
            goto err_after_prepare;
        }
    }
    return 0;

err_after_prepare:
    UvOsClose(fd);
    UvFinalize(uv, counter, 0, 0, 0);
err_after_alloc:
    QUEUE_REMOVE(&segment->queue);
    HeapFree(segment);
err:
    assert(rv != 0);
    return rv;
}

/* Return the last segment that we have requested to prepare. */
static struct uvAliveSegment *uvGetLastAliveSegment(struct uv *uv)
{
    queue *tail;
    if (QUEUE_IS_EMPTY(&uv->append_segments)) {
        return NULL;
    }
    tail = QUEUE_TAIL(&uv->append_segments);
    return QUEUE_DATA(tail, struct uvAliveSegment, queue);
}

/* Return #true if the remaining capacity of the given segment is equal or
 * greater than @size. */
static bool uvAliveSegmentHasEnoughSpareCapacity(struct uvAliveSegment *s,
                                                 size_t size)
{
    return s->size + size <= s->uv->segment_size;
}

/* Add @size bytes to the number of bytes that the segment will hold. The actual
 * write will happen when the previous write completes, if any. */
static void uvAliveSegmentReserveSegmentCapacity(struct uvAliveSegment *s,
                                                 size_t size)
{
    s->size += size;
}

/* Return the number of bytes needed to store the batch of entries of this
 * append request on disk. */
static size_t uvAppendSize(struct uvAppend *a)
{
    size_t size = sizeof(uint32_t) * 2; /* CRC checksums */
    unsigned i;
    size += uvSizeofBatchHeader(a->n); /* Batch header */
    for (i = 0; i < a->n; i++) {       /* Entries data */
        size += bytePad64(a->entries[i].buf.len);
    }
    return size;
}

/* Enqueue an append entries request, assigning it to the appropriate active
 * open segment. */
static int uvAppendEnqueueRequest(struct uv *uv, struct uvAppend *append)
{
    struct uvAliveSegment *segment;
    size_t size;
    bool fits;
    int rv;

    assert(append->entries != NULL);
    assert(append->n > 0);
    assert(uv->append_next_index > 0);

    size = uvAppendSize(append);

    /* If we have no segments yet, it means this is the very first append, and
     * we need to add a new segment. Otherwise we check if the last segment has
     * enough room for this batch of entries. */
    segment = uvGetLastAliveSegment(uv);
    if (segment == NULL || segment->finalize) {
        fits = false;
    } else {
        fits = uvAliveSegmentHasEnoughSpareCapacity(segment, size);
        if (!fits) {
            segment->finalize = true; /* Finalize when all writes are done */
        }
    }

    /* If there's no segment or if this batch does not fit in this segment, we
     * need to add a new one. */
    if (!fits) {
        rv = uvAppendPushAliveSegment(uv);
        if (rv != 0) {
            goto err;
        }
    }

    segment = uvGetLastAliveSegment(uv); /* Get the last added segment */
    uvAliveSegmentReserveSegmentCapacity(segment, size);

    append->segment = segment;
    QUEUE_PUSH(&uv->append_pending_reqs, &append->queue);
    uv->append_next_index += append->n;

    return 0;

err:
    assert(rv != 0);
    return rv;
}

int UvAppend(struct raft_io *io,
             struct raft_io_append *req,
             const struct raft_entry entries[],
             unsigned n,
             raft_io_append_cb cb)
{
    struct uv *uv;
    struct uvAppend *append;
    int rv;

    uv = io->impl;
    assert(!uv->closing);

    append = HeapMalloc(sizeof *append);
    if (append == NULL) {
        rv = RAFT_NOMEM;
        goto err;
    }
    append->req = req;
    append->entries = entries;
    append->n = n;
    req->cb = cb;

    rv = uvAppendEnqueueRequest(uv, append);
    if (rv != 0) {
        goto err_after_req_alloc;
    }

    assert(append->segment != NULL);
    assert(!QUEUE_IS_EMPTY(&uv->append_pending_reqs));

    /* Try to write immediately. */
    rv = uvAppendMaybeStart(uv);
    if (rv != 0) {
        return rv;
    }

    return 0;

err_after_req_alloc:
    HeapFree(append);
err:
    assert(rv != 0);
    return rv;
}

/* Finalize the current segment as soon as all its pending or inflight append
 * requests get completed. */
static void uvFinalizeCurrentAliveSegmentOnceIdle(struct uv *uv)
{
    struct uvAliveSegment *s;
    queue *head;
    bool has_pending_reqs;
    bool has_writing_reqs;

    s = uvGetCurrentAliveSegment(uv);
    if (s == NULL) {
        return;
    }

    /* Check if there are pending append requests targeted to the current
     * segment. */
    has_pending_reqs = false;
    QUEUE_FOREACH(head, &uv->append_pending_reqs)
    {
        struct uvAppend *r = QUEUE_DATA(head, struct uvAppend, queue);
        if (r->segment == s) {
            has_pending_reqs = true;
            break;
        }
    }
    has_writing_reqs = !QUEUE_IS_EMPTY(&uv->append_writing_reqs);

    /* If there is no pending append request or inflight write against the
     * current segment, we can submit a request for it to be closed
     * immediately. Otherwise, we set the finalize flag.
     *
     * TODO: is it actually possible to have pending requests with no writing
     * requests? Probably no. */
    if (!has_pending_reqs && !has_writing_reqs) {
        uvAliveSegmentFinalize(s);
    } else {
        s->finalize = true;
    }
}

bool UvBarrierReady(struct uv *uv)
{
    if (uv->barrier == NULL) {
        return true;
    }

    queue *head;
    QUEUE_FOREACH(head, &uv->append_segments)
    {
        struct uvAliveSegment *segment;
        segment = QUEUE_DATA(head, struct uvAliveSegment, queue);
        if (segment->barrier == uv->barrier) {
                return false;
        }
    }
    return true;
}

int UvBarrier(struct uv *uv,
              raft_index next_index,
              struct UvBarrier *barrier,
              UvBarrierCb cb)
{
    queue *head;

    assert(!uv->closing);

    /* The next entry will be appended at this index. */
    uv->append_next_index = next_index;

    /* Arrange for all open segments not already involved in other barriers to
     * be finalized as soon as their append requests get completed and mark them
     * as involved in this specific barrier request.  */
    QUEUE_FOREACH(head, &uv->append_segments)
    {
        struct uvAliveSegment *segment;
        segment = QUEUE_DATA(head, struct uvAliveSegment, queue);
        if (segment->barrier != NULL) {
            continue;
        }
        segment->barrier = barrier;
        if (segment == uvGetCurrentAliveSegment(uv)) {
            uvFinalizeCurrentAliveSegmentOnceIdle(uv);
            continue;
        }
        segment->finalize = true;
    }

    barrier->cb = cb;

    if (uv->barrier == NULL) {
        uv->barrier = barrier;
        /* If there's no pending append-related activity, we can fire the
         * callback immediately.
         *
         * TODO: find a way to avoid invoking this synchronously. */
        if (QUEUE_IS_EMPTY(&uv->append_segments) &&
            QUEUE_IS_EMPTY(&uv->finalize_reqs) &&
            uv->finalize_work.data == NULL) {
            barrier->cb(barrier);
        }
    }

    return 0;
}

void UvUnblock(struct uv *uv)
{
    assert(uv->barrier != NULL);
    uv->barrier = NULL;
    if (uv->closing) {
        uvMaybeFireCloseCb(uv);
        return;
    }
    if (!QUEUE_IS_EMPTY(&uv->append_pending_reqs)) {
        int rv;
        rv = uvAppendMaybeStart(uv);
        if (rv != 0) {
            uv->errored = true;
        }
    }
}

/* Fire all pending barrier requests, the barrier callback will notice that
 * we're closing and abort there. */
static void uvBarrierClose(struct uv *uv)
{
    struct UvBarrier *barrier = NULL;
    queue *head;
    assert(uv->closing);
    QUEUE_FOREACH(head, &uv->append_segments)
    {
        struct uvAliveSegment *segment;
        segment = QUEUE_DATA(head, struct uvAliveSegment, queue);
        if (segment->barrier != NULL && segment->barrier != barrier) {
            barrier = segment->barrier;
            barrier->cb(barrier);
            if (segment->barrier == uv->barrier) {
                uv->barrier = NULL;
            }
        }
        segment->barrier = NULL;
    }

    /* There might still still be a current barrier set on uv->barrier, meaning
     * that the open segment it was associated with has started to be finalized
     * and is not anymore in the append_segments queue. Let's cancel that
     * too. */
    if (uv->barrier != NULL) {
        uv->barrier->cb(uv->barrier);
        uv->barrier = NULL;
    }
}

void uvAppendClose(struct uv *uv)
{
    struct uvAliveSegment *segment;
    assert(uv->closing);

    uvBarrierClose(uv);
    UvPrepareClose(uv);

    uvAppendFinishPendingRequests(uv, RAFT_CANCELED);

    uvFinalizeCurrentAliveSegmentOnceIdle(uv);

    /* Also finalize the segments that we didn't write at all and are just
     * sitting in the append_segments queue waiting for writes against the
     * current segment to complete. */
    while (!QUEUE_IS_EMPTY(&uv->append_segments)) {
        segment = uvGetLastAliveSegment(uv);
        assert(segment != NULL);
        if (segment == uvGetCurrentAliveSegment(uv)) {
            break; /* We reached the head of the queue */
        }
        assert(segment->written == 0);
        uvAliveSegmentFinalize(segment);
    }
}
