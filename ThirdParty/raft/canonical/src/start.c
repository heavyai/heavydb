#include "../include/raft.h"
#include "assert.h"
#include "configuration.h"
#include "convert.h"
#include "entry.h"
#include "err.h"
#include "log.h"
#include "recv.h"
#include "snapshot.h"
#include "tick.h"
#include "tracing.h"

/* Set to 1 to enable tracing. */
#if 0
#define tracef(...) Tracef(r->tracer, __VA_ARGS__)
#else
#define tracef(...)
#endif

/* Restore the most recent configuration. */
static int restoreMostRecentConfiguration(struct raft *r,
                                          struct raft_entry *entry,
                                          raft_index index)
{
    struct raft_configuration configuration;
    int rv;
    raft_configuration_init(&configuration);
    rv = configurationDecode(&entry->buf, &configuration);
    if (rv != 0) {
        raft_configuration_close(&configuration);
        return rv;
    }
    raft_configuration_close(&r->configuration);
    r->configuration = configuration;
    r->configuration_index = index;
    return 0;
}

/* Restore the entries that were loaded from persistent storage. The most recent
 * configuration entry will be restored as well, if any.
 *
 * Note that we don't care whether the most recent configuration entry was
 * actually committed or not. We don't allow more than one pending uncommitted
 * configuration change at a time, plus
 *
 *   when adding or removing just a single server, it is safe to switch directly
 *   to the new configuration.
 *
 * and
 *
 *   The new configuration takes effect on each server as soon as it is added to
 *   that server's log: the C_new entry is replicated to the C_new servers, and
 *   a majority of the new configuration is used to determine the C_new entry's
 *   commitment. This means that servers do notwait for configuration entries to
 *   be committed, and each server always uses the latest configuration found in
 *   its log.
 *
 * as explained in section 4.1.
 *
 * TODO: we should probably set configuration_uncommitted_index as well, since we
 * can't be sure a configuration change has been committed and we need to be
 * ready to roll back to the last committed configuration.
 */
static int restoreEntries(struct raft *r,
                          raft_index snapshot_index,
                          raft_term snapshot_term,
                          raft_index start_index,
                          struct raft_entry *entries,
                          size_t n)
{
    struct raft_entry *conf = NULL;
    raft_index conf_index;
    size_t i;
    int rv;
    logStart(&r->log, snapshot_index, snapshot_term, start_index);
    r->last_stored = start_index - 1;
    for (i = 0; i < n; i++) {
        struct raft_entry *entry = &entries[i];
        rv = logAppend(&r->log, entry->term, entry->type, &entry->buf,
                       entry->batch);
        if (rv != 0) {
            goto err;
        }
        r->last_stored++;
        if (entry->type == RAFT_CHANGE) {
            conf = entry;
            conf_index = r->last_stored;
        }
    }
    if (conf != NULL) {
        rv = restoreMostRecentConfiguration(r, conf, conf_index);
        if (rv != 0) {
            goto err;
        }
    }
    raft_free(entries);
    return 0;

err:
    if (logNumEntries(&r->log) > 0) {
        logDiscard(&r->log, r->log.offset + 1);
    }
    return rv;
}

/* If we're the only voting server in the configuration, automatically
 * self-elect ourselves and convert to leader without waiting for the election
 * timeout. */
static int maybeSelfElect(struct raft *r)
{
    const struct raft_server *server;
    int rv;
    server = configurationGet(&r->configuration, r->id);
    if (server == NULL || server->role != RAFT_VOTER ||
        configurationVoterCount(&r->configuration) > 1) {
        return 0;
    }
    /* Converting to candidate will notice that we're the only voter and
     * automatically convert to leader. */
    rv = convertToCandidate(r, false /* disrupt leader */);
    if (rv != 0) {
        return rv;
    }
    assert(r->state == RAFT_LEADER);
    return 0;
}

int raft_start(struct raft *r)
{
    struct raft_snapshot *snapshot;
    raft_index snapshot_index = 0;
    raft_term snapshot_term = 0;
    raft_index start_index;
    struct raft_entry *entries;
    size_t n_entries;
    int rv;

    assert(r != NULL);
    assert(r->state == RAFT_UNAVAILABLE);
    assert(r->heartbeat_timeout != 0);
    assert(r->heartbeat_timeout < r->election_timeout);
    assert(r->install_snapshot_timeout != 0);
    assert(r->install_snapshot_timeout < r->election_timeout);
    assert(logNumEntries(&r->log) == 0);
    assert(logSnapshotIndex(&r->log) == 0);
    assert(r->last_stored == 0);

    tracef("starting");
    rv = r->io->load(r->io, &r->current_term, &r->voted_for, &snapshot,
                     &start_index, &entries, &n_entries);
    if (rv != 0) {
        ErrMsgTransfer(r->io->errmsg, r->errmsg, "io");
        return rv;
    }
    assert(start_index >= 1);

    /* If we have a snapshot, let's restore it. */
    if (snapshot != NULL) {
        tracef("restore snapshot with last index %llu and last term %llu",
               snapshot->index, snapshot->term);
        rv = snapshotRestore(r, snapshot);
        if (rv != 0) {
            snapshotDestroy(snapshot);
            entryBatchesDestroy(entries, n_entries);
            return rv;
        }
        snapshot_index = snapshot->index;
        snapshot_term = snapshot->term;
        raft_free(snapshot);
    } else if (n_entries > 0) {
        /* If we don't have a snapshot and the on-disk log is not empty, then
         * the first entry must be a configuration entry. */
        assert(start_index == 1);
        assert(entries[0].type == RAFT_CHANGE);

        /* As a small optimization, bump the commit index to 1 since we require
         * the first entry to be the same on all servers. */
        r->commit_index = 1;
        r->last_applied = 1;
    }

    /* Append the entries to the log, possibly restoring the last
     * configuration. */
    tracef("restore %lu entries starting at %llu", n_entries, start_index);
    rv = restoreEntries(r, snapshot_index, snapshot_term, start_index, entries,
                        n_entries);
    if (rv != 0) {
        entryBatchesDestroy(entries, n_entries);
        return rv;
    }

    /* Start the I/O backend. The tickCb function is expected to fire every
     * r->heartbeat_timeout milliseconds and recvCb whenever an RPC is
     * received. */
    rv = r->io->start(r->io, r->heartbeat_timeout, tickCb, recvCb);
    if (rv != 0) {
        return rv;
    }

    /* By default we start as followers. */
    convertToFollower(r);

    /* If there's only one voting server, and that is us, it's safe to convert
     * to leader right away. If that is not us, we're either joining the cluster
     * or we're simply configured as non-voter, and we'll stay follower. */
    rv = maybeSelfElect(r);
    if (rv != 0) {
        return rv;
    }

    return 0;
}

#undef tracef
