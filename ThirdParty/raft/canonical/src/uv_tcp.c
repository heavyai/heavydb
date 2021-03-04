#include "uv_tcp.h"

#include <string.h>

#include "../include/raft.h"
#include "../include/raft/uv.h"
#include "assert.h"
#include "err.h"

/* Implementation of raft_uv_transport->init. */
static int uvTcpInit(struct raft_uv_transport *transport,
                     raft_id id,
                     const char *address)
{
    struct UvTcp *t = transport->impl;
    int rv;
    assert(id > 0);
    assert(address != NULL);
    t->id = id;
    t->address = address;
    rv = uv_tcp_init(t->loop, &t->listener);
    if (rv != 0) {
        return rv;
    }
    t->listener.data = t;
    return 0;
}

/* Implementation of raft_uv_transport->close. */
static void uvTcpClose(struct raft_uv_transport *transport,
                       raft_uv_transport_close_cb cb)
{
    struct UvTcp *t = transport->impl;
    assert(!t->closing);
    t->closing = true;
    t->close_cb = cb;
    UvTcpListenClose(t);
    UvTcpConnectClose(t);
}

void UvTcpMaybeFireCloseCb(struct UvTcp *t)
{
    if (!t->closing) {
        return;
    }

    assert(QUEUE_IS_EMPTY(&t->accepting));
    assert(QUEUE_IS_EMPTY(&t->connecting));

    if (t->listener.data != NULL) {
        return;
    }
    if (!QUEUE_IS_EMPTY(&t->aborting)) {
        return;
    }

    if (t->close_cb != NULL) {
        t->close_cb(t->transport);
    }
}

int raft_uv_tcp_init(struct raft_uv_transport *transport,
                     struct uv_loop_s *loop)
{
    struct UvTcp *t;
    void *data = transport->data;
    memset(transport, 0, sizeof *transport);
    transport->data = data;
    t = raft_malloc(sizeof *t);
    if (t == NULL) {
        ErrMsgOom(transport->errmsg);
        return RAFT_NOMEM;
    }
    t->transport = transport;
    t->loop = loop;
    t->id = 0;
    t->address = NULL;
    t->listener.data = NULL;
    t->accept_cb = NULL;
    QUEUE_INIT(&t->accepting);
    QUEUE_INIT(&t->connecting);
    QUEUE_INIT(&t->aborting);
    t->closing = false;
    t->close_cb = NULL;

    transport->impl = t;
    transport->init = uvTcpInit;
    transport->close = uvTcpClose;
    transport->listen = UvTcpListen;
    transport->connect = UvTcpConnect;

    return 0;
}

void raft_uv_tcp_close(struct raft_uv_transport *transport)
{
    struct UvTcp *t = transport->impl;
    raft_free(t);
}
