#include <string.h>

#include "assert.h"
#include "byte.h"
#include "err.h"
#include "heap.h"
#include "uv_ip.h"
#include "uv_tcp.h"

/* The happy path of a connection request is:
 *
 * - Create a TCP handle and submit a TCP connect request.
 * - Once connected over TCP, submit a write request for the handshake.
 * - Once the write completes, fire the connection request callback.
 *
 * Possible failure modes are:
 *
 * - The transport get closed, close the TCP handle and and fire the request
 *   callback with RAFT_CANCELED.
 *
 * - Either the TCP connect or the write request fails: close the TCP handle and
 *   fire the request callback with RAFT_NOCONNECTION.
 */

/* Hold state for a single connection request. */
struct uvTcpConnect
{
    struct UvTcp *t;             /* Transport implementation */
    struct raft_uv_connect *req; /* User request */
    uv_buf_t handshake;          /* Handshake data */
    struct uv_tcp_s *tcp;        /* TCP connection socket handle */
    struct uv_connect_s connect; /* TCP connection request */
    struct uv_write_s write;     /* TCP handshake request */
    int status;                  /* Returned to the request callback */
    queue queue;                 /* Pending connect queue */
};

/* Encode an handshake message into the given buffer. */
static int uvTcpEncodeHandshake(raft_id id, const char *address, uv_buf_t *buf)
{
    void *cursor;
    size_t address_len = bytePad64(strlen(address) + 1);
    buf->len = sizeof(uint64_t) + /* Protocol version. */
               sizeof(uint64_t) + /* Server ID. */
               sizeof(uint64_t) /* Size of the address buffer */;
    buf->len += address_len;
    buf->base = HeapMalloc(buf->len);
    if (buf->base == NULL) {
        return RAFT_NOMEM;
    }
    cursor = buf->base;
    bytePut64(&cursor, UV__TCP_HANDSHAKE_PROTOCOL);
    bytePut64(&cursor, id);
    bytePut64(&cursor, address_len);
    strcpy(cursor, address);
    return 0;
}

/* Finish the connect request, releasing its memory and firing the connect
 * callback. */
static void uvTcpConnectFinish(struct uvTcpConnect *connect)
{
    struct uv_stream_s *stream = (struct uv_stream_s *)connect->tcp;
    struct raft_uv_connect *req = connect->req;
    int status = connect->status;
    QUEUE_REMOVE(&connect->queue);
    HeapFree(connect->handshake.base);
    raft_free(connect);
    req->cb(req, stream, status);
}

/* The TCP connection handle has been closed in consequence of an error or
 * because the transport is closing. */
static void uvTcpConnectUvCloseCb(struct uv_handle_s *handle)
{
    struct uvTcpConnect *connect = handle->data;
    struct UvTcp *t = connect->t;
    assert(connect->status != 0);
    assert(handle == (struct uv_handle_s *)connect->tcp);
    HeapFree(connect->tcp);
    connect->tcp = NULL;
    uvTcpConnectFinish(connect);
    UvTcpMaybeFireCloseCb(t);
}

/* Abort a connection request. */
static void uvTcpConnectAbort(struct uvTcpConnect *connect)
{
    QUEUE_REMOVE(&connect->queue);
    QUEUE_PUSH(&connect->t->aborting, &connect->queue);
    uv_close((struct uv_handle_s *)connect->tcp, uvTcpConnectUvCloseCb);
}

/* The handshake TCP write completes. Fire the connect callback. */
static void uvTcpConnectUvWriteCb(struct uv_write_s *write, int status)
{
    struct uvTcpConnect *connect = write->data;
    struct UvTcp *t = connect->t;

    if (t->closing) {
        connect->status = RAFT_CANCELED;
        return;
    }

    if (status != 0) {
        assert(status != UV_ECANCELED); /* t->closing would have been true */
        connect->status = RAFT_NOCONNECTION;
        uvTcpConnectAbort(connect);
        return;
    }

    uvTcpConnectFinish(connect);
}

/* The TCP connection is established. Write the handshake data. */
static void uvTcpConnectUvConnectCb(struct uv_connect_s *req, int status)
{
    struct uvTcpConnect *connect = req->data;
    struct UvTcp *t = connect->t;
    int rv;

    if (t->closing) {
        connect->status = RAFT_CANCELED;
        return;
    }

    if (status != 0) {
        assert(status != UV_ECANCELED); /* t->closing would have been true */
        connect->status = RAFT_NOCONNECTION;
        ErrMsgPrintf(t->transport->errmsg, "uv_tcp_connect(): %s",
                     uv_strerror(status));
        goto err;
    }

    rv = uv_write(&connect->write, (struct uv_stream_s *)connect->tcp,
                  &connect->handshake, 1, uvTcpConnectUvWriteCb);
    if (rv != 0) {
        /* UNTESTED: what are the error conditions? perhaps ENOMEM */
        connect->status = RAFT_NOCONNECTION;
        goto err;
    }

    return;

err:
    uvTcpConnectAbort(connect);
}

/* Create a new TCP handle and submit a connection request to the event loop. */
static int uvTcpConnectStart(struct uvTcpConnect *r, const char *address)
{
    struct UvTcp *t = r->t;
    struct sockaddr_in addr;
    int rv;

    rv = uvIpParse(address, &addr);
    if (rv != 0) {
        goto err;
    }

    /* Initialize the handshake buffer. */
    rv = uvTcpEncodeHandshake(t->id, t->address, &r->handshake);
    if (rv != 0) {
        assert(rv == RAFT_NOMEM);
        ErrMsgOom(r->t->transport->errmsg);
        goto err;
    }

    r->tcp = HeapMalloc(sizeof *r->tcp);
    if (r->tcp == NULL) {
        ErrMsgOom(t->transport->errmsg);
        rv = RAFT_NOMEM;
        goto err_after_encode_handshake;
    }

    rv = uv_tcp_init(r->t->loop, r->tcp);
    assert(rv == 0);
    r->tcp->data = r;

    rv = uv_tcp_connect(&r->connect, r->tcp, (struct sockaddr *)&addr,
                        uvTcpConnectUvConnectCb);
    if (rv != 0) {
        /* UNTESTED: since parsing succeed, this should fail only because of
         * lack of system resources */
        ErrMsgPrintf(t->transport->errmsg, "uv_tcp_connect(): %s",
                     uv_strerror(rv));
        rv = RAFT_NOCONNECTION;
        goto err_after_tcp_init;
    }

    return 0;

err_after_tcp_init:
    uv_close((uv_handle_t *)r->tcp, (uv_close_cb)HeapFree);
err_after_encode_handshake:
    HeapFree(r->handshake.base);
err:
    return rv;
}

int UvTcpConnect(struct raft_uv_transport *transport,
                 struct raft_uv_connect *req,
                 raft_id id,
                 const char *address,
                 raft_uv_connect_cb cb)
{
    struct UvTcp *t = transport->impl;
    struct uvTcpConnect *r;
    int rv;
    (void)id;
    assert(!t->closing);

    /* Create and initialize a new TCP connection request object */
    r = HeapMalloc(sizeof *r);
    if (r == NULL) {
        rv = RAFT_NOMEM;
        ErrMsgOom(transport->errmsg);
        goto err;
    }
    r->t = t;
    r->req = req;
    r->status = 0;
    r->write.data = r;
    r->connect.data = r;

    req->cb = cb;

    /* Keep track of the pending request */
    QUEUE_PUSH(&t->connecting, &r->queue);

    /* Start connecting */
    rv = uvTcpConnectStart(r, address);
    if (rv != 0) {
        goto err_after_alloc;
    }

    return 0;

err_after_alloc:
    QUEUE_REMOVE(&r->queue);
    HeapFree(r);
err:
    return rv;
}

void UvTcpConnectClose(struct UvTcp *t)
{
    while (!QUEUE_IS_EMPTY(&t->connecting)) {
        struct uvTcpConnect *connect;
        queue *head;
        head = QUEUE_HEAD(&t->connecting);
        connect = QUEUE_DATA(head, struct uvTcpConnect, queue);
        uvTcpConnectAbort(connect);
    }
}
