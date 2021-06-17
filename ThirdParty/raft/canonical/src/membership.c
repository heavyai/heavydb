#include "membership.h"

#include "../include/raft.h"
#include "assert.h"
#include "configuration.h"
#include "err.h"
#include "log.h"
#include "progress.h"

int membershipCanChangeConfiguration(struct raft *r)
{
    int rv;

    if (r->state != RAFT_LEADER || r->transfer != NULL) {
        rv = RAFT_NOTLEADER;
        goto err;
    }

    if (r->configuration_uncommitted_index != 0) {
        rv = RAFT_CANTCHANGE;
        goto err;
    }

    if (r->leader_state.promotee_id != 0) {
        rv = RAFT_CANTCHANGE;
        goto err;
    }

    /* In order to become leader at all we are supposed to have committed at
     * least the initial configuration at index 1. */
    assert(r->configuration_index > 0);

    /* The index of the last committed configuration can't be greater than the
     * last log index. */
    assert(logLastIndex(&r->log) >= r->configuration_index);

    /* No catch-up round should be in progress. */
    assert(r->leader_state.round_number == 0);
    assert(r->leader_state.round_index == 0);
    assert(r->leader_state.round_start == 0);

    return 0;

err:
    assert(rv != 0);
    ErrMsgFromCode(r->errmsg, rv);
    return rv;
}

bool membershipUpdateCatchUpRound(struct raft *r)
{
    unsigned server_index;
    raft_index match_index;
    raft_index last_index;
    raft_time now = r->io->time(r->io);
    raft_time round_duration;
    bool is_up_to_date;
    bool is_fast_enough;

    assert(r->state == RAFT_LEADER);
    assert(r->leader_state.promotee_id != 0);

    server_index =
        configurationIndexOf(&r->configuration, r->leader_state.promotee_id);
    assert(server_index < r->configuration.n);

    match_index = progressMatchIndex(r, server_index);

    /* If the server did not reach the target index for this round, it did not
     * catch up. */
    if (match_index < r->leader_state.round_index) {
        return false;
    }

    last_index = logLastIndex(&r->log);
    round_duration = now - r->leader_state.round_start;

    is_up_to_date = match_index == last_index;
    is_fast_enough = round_duration < r->election_timeout;

    /* If the server's log is fully up-to-date or the round that just terminated
     * was fast enough, then the server as caught up. */
    if (is_up_to_date || is_fast_enough) {
        r->leader_state.round_number = 0;
        r->leader_state.round_index = 0;
        r->leader_state.round_start = 0;

        return true;
    }

    /* If we get here it means that this catch-up round is complete, but there
     * are more entries to replicate, or it was not fast enough. Let's start a
     * new round. */
    r->leader_state.round_number++;
    r->leader_state.round_index = last_index;
    r->leader_state.round_start = now;

    return false;
}

int membershipUncommittedChange(struct raft *r,
                                const raft_index index,
                                const struct raft_entry *entry)
{
    struct raft_configuration configuration;
    int rv;

    assert(r != NULL);
    assert(r->state == RAFT_FOLLOWER);
    assert(entry != NULL);
    assert(entry->type == RAFT_CHANGE);

    raft_configuration_init(&configuration);

    rv = configurationDecode(&entry->buf, &configuration);
    if (rv != 0) {
        goto err;
    }

    raft_configuration_close(&r->configuration);

    r->configuration = configuration;
    r->configuration_uncommitted_index = index;

    return 0;

err:
    assert(rv != 0);
    return rv;
}

int membershipRollback(struct raft *r)
{
    const struct raft_entry *entry;
    int rv;

    assert(r != NULL);
    assert(r->state == RAFT_FOLLOWER);
    assert(r->configuration_uncommitted_index > 0);

    /* Fetch the last committed configuration entry. */
    assert(r->configuration_index != 0);

    entry = logGet(&r->log, r->configuration_index);

    assert(entry != NULL);

    /* Replace the current configuration with the last committed one. */
    raft_configuration_close(&r->configuration);
    raft_configuration_init(&r->configuration);

    rv = configurationDecode(&entry->buf, &r->configuration);
    if (rv != 0) {
        return rv;
    }

    r->configuration_uncommitted_index = 0;

    return 0;
}

void membershipLeadershipTransferInit(struct raft *r,
                                      struct raft_transfer *req,
                                      raft_id id,
                                      raft_transfer_cb cb)
{
    req->cb = cb;
    req->id = id;
    req->start = r->io->time(r->io);
    req->send.data = NULL;
    r->transfer = req;
}

int membershipLeadershipTransferStart(struct raft *r)
{
    const struct raft_server *server;
    struct raft_message message;
    int rv;
    assert(r->transfer->send.data == NULL);
    server = configurationGet(&r->configuration, r->transfer->id);
    assert(server != NULL);
    message.type = RAFT_IO_TIMEOUT_NOW;
    message.server_id = server->id;
    message.server_address = server->address;
    message.timeout_now.term = r->current_term;
    message.timeout_now.last_log_index = logLastIndex(&r->log);
    message.timeout_now.last_log_term = logLastTerm(&r->log);
    r->transfer->send.data = r;
    rv = r->io->send(r->io, &r->transfer->send, &message, NULL);
    if (rv != 0) {
        ErrMsgTransferf(r->io->errmsg, r->errmsg, "send timeout now to %llu",
                        server->id);
        return rv;
    }
    return 0;
}

void membershipLeadershipTransferClose(struct raft *r)
{
    struct raft_transfer *req = r->transfer;
    raft_transfer_cb cb = req->cb;
    r->transfer = NULL;
    if (cb != NULL) {
        cb(req);
    }
}
