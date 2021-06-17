/* Modify and inspect @raft_configuration objects. */

#ifndef CONFIGURATION_H_
#define CONFIGURATION_H_

#include "../include/raft.h"

/* Initialize an empty configuration. */
void configurationInit(struct raft_configuration *c);

/* Release all memory used by the given configuration. */
void configurationClose(struct raft_configuration *c);

/* Add a server to the given configuration. */
int configurationAdd(struct raft_configuration *c,
                     raft_id id,
                     const char *address,
                     int role);

/* Return the number of servers with the RAFT_VOTER role. */
unsigned configurationVoterCount(const struct raft_configuration *c);

/* Return the index of the server with the given ID (relative to the c->servers
 * array). If there's no server with the given ID, return the number of
 * servers. */
unsigned configurationIndexOf(const struct raft_configuration *c, raft_id id);

/* Return the index of the RAFT_VOTER server with the given ID (relative to the
 * sub array of c->servers that has only voting servers). If there's no server
 * with the given ID, or if it's not flagged as voting, return the number of
 * servers. */
unsigned configurationIndexOfVoter(const struct raft_configuration *c,
                                   raft_id id);

/* Get the server with the given ID, or #NULL if no matching server is found. */
const struct raft_server *configurationGet(const struct raft_configuration *c,
                                           raft_id id);

/* Remove a server from a raft configuration. The given ID must match the one of
 * an existing server in the configuration. */
int configurationRemove(struct raft_configuration *c, raft_id id);

/* Add all servers in c1 to c2 (which must be empty). */
int configurationCopy(const struct raft_configuration *src,
                      struct raft_configuration *dst);

/* Number of bytes needed to encode the given configuration object. */
size_t configurationEncodedSize(const struct raft_configuration *c);

/* Encode the given configuration object to the given pre-allocated buffer,
 * which is assumed to be at least configurationEncodedSize(c) bytes. */
void configurationEncodeToBuf(const struct raft_configuration *c, void *buf);

/* Encode the given configuration object. The memory of the returned buffer is
 * allocated using raft_malloc(), and client code is responsible for releasing
 * it when no longer needed. */
int configurationEncode(const struct raft_configuration *c,
                        struct raft_buffer *buf);

/* Populate a configuration object by decoding the given serialized payload. */
int configurationDecode(const struct raft_buffer *buf,
                        struct raft_configuration *c);

#endif /* CONFIGURATION_H_ */
