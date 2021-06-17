#include "../include/raft.h"

#include <string.h>

#include "assert.h"
#include "byte.h"
#include "configuration.h"
#include "convert.h"
#include "election.h"
#include "err.h"
#include "heap.h"
#include "log.h"
#include "membership.h"
#include "tracing.h"

#define DEFAULT_ELECTION_TIMEOUT 1000 /* One second */
#define DEFAULT_HEARTBEAT_TIMEOUT 100 /* One tenth of a second */
#define DEFAULT_INSTALL_SNAPSHOT_TIMEOUT (2 * DEFAULT_HEARTBEAT_TIMEOUT)
#define DEFAULT_SNAPSHOT_THRESHOLD 1024
#define DEFAULT_SNAPSHOT_TRAILING 2048

/* Number of milliseconds after which a server promotion will be aborted if the
 * server hasn't caught up with the logs yet. */
#define DEFAULT_MAX_CATCH_UP_ROUNDS 10
#define DEFAULT_MAX_CATCH_UP_ROUND_DURATION (5 * 1000)

int raft_init(struct raft *r,
              struct raft_io *io,
              struct raft_fsm *fsm,
              const raft_id id,
              const char *address)
{
    int rv;
    assert(r != NULL);
    r->io = io;
    r->io->data = r;
    r->fsm = fsm;
    r->tracer = &NoopTracer;
    r->id = id;
    /* Make a copy of the address */
    r->address = HeapMalloc(strlen(address) + 1);
    if (r->address == NULL) {
        rv = RAFT_NOMEM;
        goto err;
    }
    strcpy(r->address, address);
    r->current_term = 0;
    r->voted_for = 0;
    logInit(&r->log);
    raft_configuration_init(&r->configuration);
    r->configuration_index = 0;
    r->configuration_uncommitted_index = 0;
    r->election_timeout = DEFAULT_ELECTION_TIMEOUT;
    r->heartbeat_timeout = DEFAULT_HEARTBEAT_TIMEOUT;
    r->install_snapshot_timeout = DEFAULT_INSTALL_SNAPSHOT_TIMEOUT;
    r->commit_index = 0;
    r->last_applied = 0;
    r->last_stored = 0;
    r->state = RAFT_UNAVAILABLE;
    r->transfer = NULL;
    r->snapshot.pending.term = 0;
    r->snapshot.threshold = DEFAULT_SNAPSHOT_THRESHOLD;
    r->snapshot.trailing = DEFAULT_SNAPSHOT_TRAILING;
    r->snapshot.put.data = NULL;
    r->close_cb = NULL;
    memset(r->errmsg, 0, sizeof r->errmsg);
    r->pre_vote = false;
    r->max_catch_up_rounds = DEFAULT_MAX_CATCH_UP_ROUNDS;
    r->max_catch_up_round_duration = DEFAULT_MAX_CATCH_UP_ROUND_DURATION;
    rv = r->io->init(r->io, r->id, r->address);
    if (rv != 0) {
        ErrMsgTransfer(r->io->errmsg, r->errmsg, "io");
        goto err_after_address_alloc;
    }
    return 0;

err_after_address_alloc:
    HeapFree(r->address);
err:
    assert(rv != 0);
    return rv;
}

static void ioCloseCb(struct raft_io *io)
{
    struct raft *r = io->data;
    raft_free(r->address);
    logClose(&r->log);
    raft_configuration_close(&r->configuration);
    if (r->close_cb != NULL) {
        r->close_cb(r);
    }
}

void raft_close(struct raft *r, void (*cb)(struct raft *r))
{
    assert(r->close_cb == NULL);
    if (r->state != RAFT_UNAVAILABLE) {
        convertToUnavailable(r);
    }
    r->close_cb = cb;
    r->io->close(r->io, ioCloseCb);
}

void raft_set_election_timeout(struct raft *r, const unsigned msecs)
{
    r->election_timeout = msecs;
}

void raft_set_heartbeat_timeout(struct raft *r, const unsigned msecs)
{
    r->heartbeat_timeout = msecs;
}

void raft_set_install_snapshot_timeout(struct raft *r, const unsigned msecs)
{
    r->install_snapshot_timeout = msecs;
}

void raft_set_snapshot_threshold(struct raft *r, unsigned n)
{
    r->snapshot.threshold = n;
}

void raft_set_snapshot_trailing(struct raft *r, unsigned n)
{
    r->snapshot.trailing = n;
}

void raft_set_max_catch_up_rounds(struct raft *r, unsigned n)
{
    r->max_catch_up_rounds = n;
}

void raft_set_max_catch_up_round_duration(struct raft *r, unsigned msecs)
{
    r->max_catch_up_round_duration = msecs;
}

void raft_set_pre_vote(struct raft *r, bool enabled)
{
    r->pre_vote = enabled;
}

const char *raft_errmsg(struct raft *r)
{
    return r->errmsg;
}

int raft_bootstrap(struct raft *r, const struct raft_configuration *conf)
{
    int rv;

    if (r->state != RAFT_UNAVAILABLE) {
        return RAFT_BUSY;
    }

    rv = r->io->bootstrap(r->io, conf);
    if (rv != 0) {
        return rv;
    }

    return 0;
}

int raft_recover(struct raft *r, const struct raft_configuration *conf)
{
    int rv;

    if (r->state != RAFT_UNAVAILABLE) {
        return RAFT_BUSY;
    }

    rv = r->io->recover(r->io, conf);
    if (rv != 0) {
        return rv;
    }

    return 0;
}

const char *raft_strerror(int errnum)
{
    return errCodeToString(errnum);
}

void raft_configuration_init(struct raft_configuration *c)
{
    configurationInit(c);
}

void raft_configuration_close(struct raft_configuration *c)
{
    configurationClose(c);
}

int raft_configuration_add(struct raft_configuration *c,
                           const raft_id id,
                           const char *address,
                           const int role)
{
    return configurationAdd(c, id, address, role);
}

int raft_configuration_encode(const struct raft_configuration *c,
                              struct raft_buffer *buf)
{
    return configurationEncode(c, buf);
}

unsigned long long raft_digest(const char *text, unsigned long long n)
{
    struct byteSha1 sha1;
    uint8_t value[20];
    uint64_t n64 = byteFlip64((uint64_t)n);
    uint64_t digest;

    byteSha1Init(&sha1);
    byteSha1Update(&sha1, (const uint8_t *)text, (uint32_t)strlen(text));
    byteSha1Update(&sha1, (const uint8_t *)&n64, (uint32_t)(sizeof n64));
    byteSha1Digest(&sha1, value);

    memcpy(&digest, value + (sizeof value - sizeof digest), sizeof digest);

    return byteFlip64(digest);
}
