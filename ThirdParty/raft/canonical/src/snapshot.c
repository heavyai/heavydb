#include "snapshot.h"

#include <stdint.h>
#include <string.h>

#include "assert.h"
#include "configuration.h"
#include "err.h"
#include "log.h"
#include "tracing.h"

/* Set to 1 to enable tracing. */
#if 1
#define tracef(...) Tracef(r->tracer, __VA_ARGS__)
#else
#define tracef(...)
#endif

void snapshotClose(struct raft_snapshot *s)
{
    unsigned i;
    configurationClose(&s->configuration);
    for (i = 0; i < s->n_bufs; i++) {
        raft_free(s->bufs[i].base);
    }
    raft_free(s->bufs);
}

void snapshotDestroy(struct raft_snapshot *s)
{
    snapshotClose(s);
    raft_free(s);
}

int snapshotRestore(struct raft *r, struct raft_snapshot *snapshot)
{
    int rv;

    assert(snapshot->n_bufs == 1);

    rv = r->fsm->restore(r->fsm, &snapshot->bufs[0]);
    if (rv != 0) {
        tracef("restore snapshot %llu: %s", snapshot->index,
               errCodeToString(rv));
        return rv;
    }

    configurationClose(&r->configuration);
    r->configuration = snapshot->configuration;
    r->configuration_index = snapshot->configuration_index;

    r->commit_index = snapshot->index;
    r->last_applied = snapshot->index;
    r->last_stored = snapshot->index;

    /* Don't free the snapshot data buffer, as ownership has been transferred to
     * the fsm. */
    raft_free(snapshot->bufs);

    return 0;
}

int snapshotCopy(const struct raft_snapshot *src, struct raft_snapshot *dst)
{
    int rv;
    unsigned i;
    size_t size;
    uint8_t *cursor;

    dst->term = src->term;
    dst->index = src->index;

    rv = configurationCopy(&src->configuration, &dst->configuration);
    if (rv != 0) {
        return rv;
    }

    size = 0;
    for (i = 0; i < src->n_bufs; i++) {
        size += src->bufs[i].len;
    }

    dst->bufs = raft_malloc(sizeof *dst->bufs);
    assert(dst->bufs != NULL);

    dst->bufs[0].base = raft_malloc(size);
    dst->bufs[0].len = size;
    if (dst->bufs[0].base == NULL) {
        return RAFT_NOMEM;
    }

    cursor = dst->bufs[0].base;

    for (i = 0; i < src->n_bufs; i++) {
        memcpy(cursor, src->bufs[i].base, src->bufs[i].len);
        cursor += src->bufs[i].len;
    }

    dst->n_bufs = 1;

    return 0;
}

#undef tracef
