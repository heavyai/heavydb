#ifndef REQUEST_H_
#define REQUEST_H_

#include "../include/raft.h"

/* Abstract request type */
struct request
{
    /* Must be kept in sync with RAFT__REQUEST in raft.h */
    void *data;
    int type;
    raft_index index;
    void *queue[2];
};

#endif /* REQUEST_H_ */
