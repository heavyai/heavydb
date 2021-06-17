#ifndef RAFT_UV_H
#define RAFT_UV_H

#include <uv.h>

#include "../raft.h"

struct raft_uv_transport;

/**
 * Configure the given @raft_io instance to use a libuv-based I/O
 * implementation.
 *
 * The @dir path will be copied, and its memory can possibly be released once
 * this function returns.
 *
 * Return #RAFT_NAMETOOLONG if @dir exceeds the size of the internal buffer
 * that should hold it
 *
 * Return #RAFT_NOTFOUND if @dir does not exist.
 *
 * Return #RAFT_INVALID if @dir exists but it's not a directory.
 *
 * The implementation of metadata and log persistency is virtually the same as
 * the one found in LogCabin [0].
 *
 * The disk files consist of metadata files, closed segments, and open
 * segments. Metadata files are used to track Raft metadata, such as the
 * server's current term, vote, and log's start index. Segments contain
 * contiguous entries that are part of the log. Closed segments are never
 * written to again (but may be renamed and truncated if a suffix of the log is
 * truncated). Open segments are where newly appended entries go. Once an open
 * segment reaches the maximum allowed size, it is closed and a new one is used.
 *
 * Metadata files are named "metadata1" and "metadata2". The code alternates
 * between these so that there is always at least one readable metadata file.
 * On boot, the readable metadata file with the higher version number is used.
 *
 * The format of a metadata file is:
 *
 * [8 bytes] Format (currently 1).
 * [8 bytes] Incremental version number.
 * [8 bytes] Current term.
 * [8 bytes] ID of server we voted for.
 *
 * Closed segments are named by the format string "%lu-%lu" with their
 * start and end indexes, both inclusive. Closed segments always contain at
 * least one entry; the end index is always at least as large as the start
 * index. Closed segment files may occasionally include data past their
 * filename's end index (these are ignored but a warning is logged). This can
 * happen if the suffix of the segment is truncated and a crash occurs at an
 * inopportune time (the segment file is first renamed, then truncated, and a
 * crash occurs in between).
 *
 * Open segments are named by the format string "open-%lu" with a unique
 * number. These should not exist when the server shuts down cleanly, but they
 * exist while the server is running and may be left around during a crash.
 * Open segments either contain entries which come after the last closed
 * segment or are full of zeros. When the server crashes while appending to an
 * open segment, the end of that file may be corrupt. We can't distinguish
 * between a corrupt file and a partially written entry. The code assumes it's
 * a partially written entry, logs a warning, and ignores it.
 *
 * Truncating a suffix of the log will remove all entries that are no longer
 * part of the log. Truncating a prefix of the log will only remove complete
 * segments that are before the new log start index. For example, if a
 * segment has entries 10 through 20 and the prefix of the log is truncated to
 * start at entry 15, that entire segment will be retained.
 *
 * Each segment file starts with a segment header, which currently contains
 * just an 8-byte version number for the format of that segment. The current
 * format (version 1) is just a concatenation of serialized entry batches.
 *
 * Each batch has the following format:
 *
 * [4 bytes] CRC32 checksum of the batch header, little endian.
 * [4 bytes] CRC32 checksum of the batch data, little endian.
 * [  ...  ] Batch (as described in @raft_decode_entries_batch).
 *
 * [0] https://github.com/logcabin/logcabin/blob/master/Storage/SegmentedLog.h
 */
RAFT_API int raft_uv_init(struct raft_io *io,
                          struct uv_loop_s *loop,
                          const char *dir,
                          struct raft_uv_transport *transport);

/**
 * Release any memory allocated internally.
 */
RAFT_API void raft_uv_close(struct raft_io *io);

/**
 * Set the block size that will be used for direct I/O.
 *
 * The default is to automatically detect the appropriate block size.
 */
RAFT_API void raft_uv_set_block_size(struct raft_io *io, size_t size);

/**
 * Set the maximum initial size of newly created open segments.
 *
 * If the given size is not a multiple of the block size, the actual size will
 * be reduced to the closest multiple.
 *
 * The default is 8 megabytes.
 */
RAFT_API void raft_uv_set_segment_size(struct raft_io *io, size_t size);

/**
 * Set how many milliseconds to wait between subsequent retries when
 * establishing a connection with another server. The default is 1000
 * milliseconds.
 */
RAFT_API void raft_uv_set_connect_retry_delay(struct raft_io *io, unsigned msecs);

/**
 * Emit low-level debug messages using the given tracer.
 */
RAFT_API void raft_uv_set_tracer(struct raft_io *io,
                                 struct raft_tracer *tracer);

/**
 * Callback invoked by the transport implementation when a new incoming
 * connection has been established.
 *
 * No references to @address must be kept after this function returns.
 *
 * Ownership of @stream is transferred to user code, which is responsible of
 * uv_close()'ing it and then releasing its memory.
 */
typedef void (*raft_uv_accept_cb)(struct raft_uv_transport *t,
                                  raft_id id,
                                  const char *address,
                                  struct uv_stream_s *stream);

/**
 * Callback invoked by the transport implementation after a connect request has
 * completed. If status is #0, then @stream will point to a valid handle, which
 * user code is then responsible to uv_close() and then release.
 */
struct raft_uv_connect;
typedef void (*raft_uv_connect_cb)(struct raft_uv_connect *req,
                                   struct uv_stream_s *stream,
                                   int status);

/**
 * Handle to a connect request.
 */
struct raft_uv_connect
{
    void *data;            /* User data */
    raft_uv_connect_cb cb; /* Callback */
};

/**
 * Callback invoked by the transport implementation after a close request is
 * completed.
 */
typedef void (*raft_uv_transport_close_cb)(struct raft_uv_transport *t);

/**
 * Interface to establish outgoing connections to other Raft servers and to
 * accept incoming connections from them.
 */
struct raft_uv_transport
{
    /**
     * User defined data.
     */
    void *data;

    /**
     * Implementation-defined state.
     */
    void *impl;

    /**
     * Human-readable message providing diagnostic information about the last
     * error occurred.
     */
    char errmsg[RAFT_ERRMSG_BUF_SIZE];

    /**
     * Initialize the transport with the given server's identity.
     */
    int (*init)(struct raft_uv_transport *t, raft_id id, const char *address);

    /**
     * Start listening for incoming connections.
     *
     * Once a new connection is accepted, the @cb callback passed in the
     * initializer must be invoked with the relevant details of the connecting
     * Raft server.
     */
    int (*listen)(struct raft_uv_transport *t, raft_uv_accept_cb cb);

    /**
     * Connect to the server with the given ID and address.
     *
     * The @cb callback must be invoked when the connection has been established
     * or the connection attempt has failed. The memory pointed by @req can be
     * released only after @cb has fired.
     */
    int (*connect)(struct raft_uv_transport *t,
                   struct raft_uv_connect *req,
                   raft_id id,
                   const char *address,
                   raft_uv_connect_cb cb);

    /**
     * Close the transport.
     *
     * The implementation must:
     *
     * - Stop accepting incoming connections. The @cb callback passed to @listen
     *   must not be invoked anymore.
     *
     * - Cancel all pending @connect requests.
     *
     * - Invoke the @cb callback passed to this method once it's safe to release
     *   the memory of the transport object.
     */
    void (*close)(struct raft_uv_transport *t, raft_uv_transport_close_cb cb);
};

/**
 * Init a transport interface that uses TCP sockets.
 */
RAFT_API int raft_uv_tcp_init(struct raft_uv_transport *t,
                              struct uv_loop_s *loop);

/**
 * Release any memory allocated internally.
 */
RAFT_API void raft_uv_tcp_close(struct raft_uv_transport *t);

#endif /* RAFT_UV_H */
