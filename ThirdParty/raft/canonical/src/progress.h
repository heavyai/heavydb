/* Track replication progress on followers. */

#ifndef PROGRESS_H_
#define PROGRESS_H_

#include "../include/raft.h"

/* Possible values for the state field of struct raft_progress. */
enum {
    PROGRESS__PROBE = 0, /* At most one AppendEntries per heartbeat interval */
    PROGRESS__PIPELINE,  /* Optimistically stream AppendEntries */
    PROGRESS__SNAPSHOT   /* Sending a snapshot */
};

/* Create and initialize the array of progress objects used by the leader to *
 * track followers. The match index will be set to zero, and the next index to
 * the current last index plus 1. */
int progressBuildArray(struct raft *r);

/* Re-build the progress array against a new configuration.
 *
 * Progress information for servers existing both in the new and in the current
 * configuration will remain unchanged.
 *
 * Progress information for servers existing only in the new configuration will
 * be initialized as in progressBuildArray().*/
int progressRebuildArray(struct raft *r,
                         const struct raft_configuration *configuration);

/* Whether the log of the i'th server in the configuration up-to-date with
 * ours. */
bool progressIsUpToDate(struct raft *r, unsigned i);

/* Whether a new AppendEntries or InstallSnapshot message should be sent to the
 * i'th server at this time.
 *
 * See the docstring of replicationProgress() for details about how the decision
 * is taken. */
bool progressShouldReplicate(struct raft *r, unsigned i);

/* Return the index of the next entry that should be sent to the i'th server. */
raft_index progressNextIndex(struct raft *r, unsigned i);

/* Return the index of the most recent entry that the i'th server has reported
 * as replicated. */
raft_index progressMatchIndex(struct raft *r, unsigned i);

/* Update the last_send timestamp after an AppendEntries request has been
 * sent. */
void progressUpdateLastSend(struct raft *r, unsigned i);

/* Reset to false the recent_recv flag of the server at the given index,
 * returning the previous value.
 *
 * To be called once every election_timeout milliseconds. */
bool progressResetRecentRecv(struct raft *r, unsigned i);

/* Set to true the recent_recv flag of the server at the given index.
 *
 * To be called whenever we receive an AppendEntries RPC result */
void progressMarkRecentRecv(struct raft *r, unsigned i);

/* Convert to the i'th server to snapshot mode. */
void progressToSnapshot(struct raft *r, unsigned i);

/* Convert to probe mode. */
void progressToProbe(struct raft *r, unsigned i);

/* Convert to pipeline mode. */
void progressToPipeline(struct raft *r, unsigned i);

/* Abort snapshot mode and switch to back to probe.
 *
 * Called after sending the snapshot has failed or timed out. */
void progressAbortSnapshot(struct raft *r, unsigned i);

/* Return the progress mode code for the i'th server. */
int progressState(struct raft *r, unsigned i);

/* Optimistically update the next index of the given server.
 *
 * Called in pipeline mode after sending new entries. */
void progressOptimisticNextIndex(struct raft *r,
                                 unsigned i,
                                 raft_index next_index);

/* Return false if the given @index comes from an outdated message. Otherwise
 * update the progress and returns true. To be called when receiving a
 * successful AppendEntries RPC response. */
bool progressMaybeUpdate(struct raft *r, unsigned i, raft_index last_index);

/* Return false if the given rejected index comes from an out of order
 * message. Otherwise decrease the progress next index to min(rejected,
 * last_index) and returns true. To be called when receiving an unsuccessful
 * AppendEntries RPC response. */
bool progressMaybeDecrement(struct raft *r,
                            unsigned i,
                            raft_index rejected,
                            raft_index last_index);

/* Return true if match_index is equal or higher than the snapshot_index. */
bool progressSnapshotDone(struct raft *r, unsigned i);

#endif /* PROGRESS_H_ */
