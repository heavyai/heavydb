#include "recv_request_vote.h"

#include "assert.h"
#include "election.h"
#include "recv.h"
#include "tracing.h"

/* Set to 1 to enable tracing. */
#if 1
#define tracef(...) Tracef(r->tracer, __VA_ARGS__)
#else
#define tracef(...)
#endif

static void requestVoteSendCb(struct raft_io_send *req, int status)
{
    (void)status;
    raft_free(req);
}

int recvRequestVote(struct raft *r,
                    const raft_id id,
                    const char *address,
                    const struct raft_request_vote *args)
{
    struct raft_io_send *req;
    struct raft_message message;
    struct raft_request_vote_result *result = &message.request_vote_result;
    bool has_leader;
    int match;
    int rv;

    assert(r != NULL);
    assert(id > 0);
    assert(args != NULL);

    result->vote_granted = false;

    /* Reject the request if we have a leader.
     *
     * From Section 4.2.3:
     *
     *   [Removed] servers should not be able to disrupt a leader whose cluster
     *   is receiving heartbeats. [...] If a server receives a RequestVote
     *   request within the minimum election timeout of hearing from a current
     *   leader, it does not update its term or grant its vote
     *
     * From Section 4.2.3:
     *
     *   This change conflicts with the leadership transfer mechanism as
     *   described in Chapter 3, in which a server legitimately starts an
     *   election without waiting an election timeout. In that case, RequestVote
     *   messages should be processed by other servers even when they believe a
     *   current cluster leader exists. Those RequestVote requests can include a
     *   special flag to indicate this behavior ("I have permission to disrupt
     *   the leader - it told me to!").
     */
    has_leader =
        r->state == RAFT_LEADER ||
        (r->state == RAFT_FOLLOWER && r->follower_state.current_leader.id != 0);
    if (has_leader && !args->disrupt_leader) {
        tracef("local server has a leader -> reject ");
        goto reply;
    }

    /* If this is a pre-vote request, don't actually increment our term or
     * persist the vote. */
    if (args->pre_vote) {
        recvCheckMatchingTerms(r, args->term, &match);
    } else {
        rv = recvEnsureMatchingTerms(r, args->term, &match);
        if (rv != 0) {
            return rv;
        }
    }

    /* From Figure 3.1:
     *
     *   RequestVote RPC: Receiver implementation: Reply false if
     *   term < currentTerm.
     *
     */
    if (match < 0) {
        tracef("local term is higher -> reject ");
        goto reply;
    }

    /* Unless this is a pre-vote request, at this point our term must be the
     * same as the request term (otherwise we would have rejected the request or
     * bumped our term). */
    if (!args->pre_vote) {
        assert(r->current_term == args->term);
    }

    rv = electionVote(r, args, &result->vote_granted);
    if (rv != 0) {
        return rv;
    }

reply:
    result->term = r->current_term;

    message.type = RAFT_IO_REQUEST_VOTE_RESULT;
    message.server_id = id;
    message.server_address = address;

    req = raft_malloc(sizeof *req);
    if (req == NULL) {
        return RAFT_NOMEM;
    }
    req->data = r;

    rv = r->io->send(r->io, req, &message, requestVoteSendCb);
    if (rv != 0) {
        raft_free(req);
        return rv;
    }

    return 0;
}

#undef tracef
