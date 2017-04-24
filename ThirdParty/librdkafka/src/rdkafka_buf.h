/*
 * librdkafka - Apache Kafka C library
 *
 * Copyright (c) 2012-2015, Magnus Edenhill
 * All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met: 
 * 
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer. 
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution. 
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE 
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE 
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE 
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR 
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF 
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS 
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN 
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */
#pragma once

#include "rdkafka_int.h"
#include "rdcrc32.h"
#include "rdlist.h"

typedef struct rd_kafka_broker_s rd_kafka_broker_t;

#define RD_KAFKA_HEADERS_IOV_CNT   2

/* Align X (upwards) to STRIDE, which must be power of 2. */
#define _ALIGN(X,STRIDE) (((X) + ((STRIDE) - 1)) & -(STRIDE))


/**
 * Temporary buffer with memory aligned writes to accommodate
 * effective and platform safe struct writes.
 */
typedef struct rd_tmpabuf_s {
	size_t size;
	size_t of;
	char  *buf;
	int    failed;
	int    assert_on_fail;
} rd_tmpabuf_t;

/**
 * @brief Allocate new tmpabuf with \p size bytes pre-allocated.
 */
static RD_UNUSED void
rd_tmpabuf_new (rd_tmpabuf_t *tab, size_t size, int assert_on_fail) {
	tab->buf = rd_malloc(size);
	tab->size = size;
	tab->of = 0;
	tab->failed = 0;
	tab->assert_on_fail = assert_on_fail;
}

/**
 * @brief Free memory allocated by tmpabuf
 */
static RD_UNUSED void
rd_tmpabuf_destroy (rd_tmpabuf_t *tab) {
	rd_free(tab->buf);
}

/**
 * @returns 1 if a previous operation failed.
 */
static RD_UNUSED RD_INLINE int
rd_tmpabuf_failed (rd_tmpabuf_t *tab) {
	return tab->failed;
}

/**
 * @brief Allocate \p size bytes for writing, returning an aligned pointer
 *        to the memory.
 * @returns the allocated pointer (within the tmpabuf) on success or
 *          NULL if the requested number of bytes + alignment is not available
 *          in the tmpabuf.
 */
static RD_UNUSED void *
rd_tmpabuf_alloc0 (const char *func, int line, rd_tmpabuf_t *tab, size_t size) {
	void *ptr;

	if (unlikely(tab->failed))
		return NULL;

	if (unlikely(tab->of + size > tab->size)) {
		if (tab->assert_on_fail) {
			fprintf(stderr,
				"%s: %s:%d: requested size %zd + %zd > %zd\n",
				__FUNCTION__, func, line, tab->of, size,
				tab->size);
			assert(!*"rd_tmpabuf_alloc: not enough size in buffer");
		}
		return NULL;
	}

        ptr = (void *)(tab->buf + tab->of);
	tab->of += _ALIGN(size, 8);

	return ptr;
}

#define rd_tmpabuf_alloc(tab,size) \
	rd_tmpabuf_alloc0(__FUNCTION__,__LINE__,tab,size)

/**
 * @brief Write \p buf of \p size bytes to tmpabuf memory in an aligned fashion.
 *
 * @returns the allocated and written-to pointer (within the tmpabuf) on success
 *          or NULL if the requested number of bytes + alignment is not available
 *          in the tmpabuf.
 */
static RD_UNUSED void *
rd_tmpabuf_write0 (const char *func, int line,
		   rd_tmpabuf_t *tab, const void *buf, size_t size) {
	void *ptr = rd_tmpabuf_alloc0(func, line, tab, size);

	if (ptr)
		memcpy(ptr, buf, size);

	return ptr;
}
#define rd_tmpabuf_write(tab,buf,size) \
	rd_tmpabuf_write0(__FUNCTION__, __LINE__, tab, buf, size)


/**
 * @brief Wrapper for rd_tmpabuf_write() that takes a nul-terminated string.
 */
static RD_UNUSED char *
rd_tmpabuf_write_str0 (const char *func, int line,
		       rd_tmpabuf_t *tab, const char *str) {
	return rd_tmpabuf_write0(func, line, tab, str, strlen(str)+1);
}
#define rd_tmpabuf_write_str(tab,str) \
	rd_tmpabuf_write_str0(__FUNCTION__, __LINE__, tab, str)



/**
 *
 * Read buffer interface
 *
 * Memory reading helper macros to be used when parsing network responses.
 *
 * Assumptions:
 *   - an 'err:' goto-label must be available for error bailouts.
 */

#define rd_kafka_buf_parse_fail(rkbuf,...) do {				\
                if (log_decode_errors) {                                \
			rd_kafka_assert(NULL, rkbuf->rkbuf_rkb);	\
                        rd_rkb_log(rkbuf->rkbuf_rkb, LOG_WARNING, "PROTOERR", \
                                   "Protocol parse failure at %s:%i " \
				   "(incorrect broker.version.fallback?)",   \
                                   __FUNCTION__, __LINE__);             \
                        rd_rkb_log(rkbuf->rkbuf_rkb, LOG_WARNING,	\
				   "PROTOERR", __VA_ARGS__);		\
                }                                                       \
                goto err;                                               \
	} while (0)



/**
 * Returns the number of remaining bytes available to read.
 */
#define rd_kafka_buf_remain(rkbuf) (int)(rkbuf->rkbuf_wof - rkbuf->rkbuf_of)

/**
 * Checks that at least 'len' bytes remain to be read in buffer, else fails.
 */
#define rd_kafka_buf_check_len(rkbuf,len) do {				\
		int _LEN = (int)(len);					\
		if (unlikely(_LEN > rd_kafka_buf_remain(rkbuf))) {	\
			rd_kafka_buf_parse_fail(rkbuf, \
						"expected %i bytes > %i " \
						"remaining bytes",	\
						_LEN,			\
						(int)rd_kafka_buf_remain(rkbuf)); \
			goto err;					\
		}							\
	} while (0)

/**
 * Skip (as in read and ignore) the next 'len' bytes.
 */
#define rd_kafka_buf_skip(rkbuf, len) do {	\
		rd_kafka_buf_check_len(rkbuf, len);		\
		rkbuf->rkbuf_of += (len);			\
	} while (0)



/**
 * Read 'len' bytes and copy to 'dstptr'
 */
#define rd_kafka_buf_read(rkbuf,dstptr,len) do {			\
		rd_kafka_buf_check_len(rkbuf, len);			\
		memcpy((dstptr), rkbuf->rkbuf_rbuf+rkbuf->rkbuf_of, (len)); \
		rkbuf->rkbuf_of += (len);				\
	} while (0)

/**
 * Read a 16,32,64-bit integer and store it in 'dstptr' (which must be aligned).
 */
#define rd_kafka_buf_read_i64(rkbuf,dstptr) do {			\
		rd_kafka_buf_read(rkbuf, dstptr, 8);			\
		*(int64_t *)(dstptr) = be64toh(*(int64_t *)(dstptr));	\
	} while (0)

#define rd_kafka_buf_read_i32(rkbuf,dstptr) do {			\
		rd_kafka_buf_read(rkbuf, dstptr, 4);			\
		*(int32_t *)(dstptr) = be32toh(*(int32_t *)(dstptr));	\
	} while (0)

/* Same as .._read_i32 but does a direct assignment.
 * dst is assumed to be a scalar, not pointer. */
#define rd_kafka_buf_read_i32a(rkbuf, dst) do {				\
                int32_t _v;                                             \
		rd_kafka_buf_read(rkbuf, &_v, 4);			\
		dst = (int32_t) be32toh(_v);				\
	} while (0)

#define rd_kafka_buf_read_i16(rkbuf, dstptr) do {			\
		rd_kafka_buf_read(rkbuf, dstptr, 2);			\
		*(int16_t *)(dstptr) = be16toh(*(int16_t *)(dstptr));	\
	} while (0)

#define rd_kafka_buf_read_i16a(rkbuf, dst) do {				\
                int16_t _v;                                             \
		rd_kafka_buf_read(rkbuf, &_v, 2);			\
                dst = (int16_t)be16toh(_v);				\
	} while (0)

#define rd_kafka_buf_read_i8(rkbuf, dst) rd_kafka_buf_read(rkbuf, dst, 1)


/* Read Kafka String representation (2+N).
 * The kstr data will be updated to point to the rkbuf. */
#define rd_kafka_buf_read_str(rkbuf, kstr) do {				\
		int _ksize;						\
		rd_kafka_buf_read_i16a(rkbuf, (kstr)->len);		\
		_ksize = RD_KAFKAP_STR_LEN(kstr);			\
		(kstr)->str = RD_KAFKAP_STR_IS_NULL(kstr) ?		\
			NULL : ((const char *)rkbuf->rkbuf_rbuf+rkbuf->rkbuf_of); \
		rd_kafka_buf_skip(rkbuf, _ksize);			\
	} while (0)

/* Read Kafka String representation (2+N) and write it to the \p tmpabuf
 * with a trailing nul byte. */
#define rd_kafka_buf_read_str_tmpabuf(rkbuf, tmpabuf, dst) do {		\
                rd_kafkap_str_t _kstr;					\
		size_t _slen;						\
		char *_dst;						\
		rd_kafka_buf_read_str(rkbuf, &_kstr);			\
		_slen = RD_KAFKAP_STR_LEN(&_kstr);			\
		if (!(_dst =						\
		      rd_tmpabuf_write(tmpabuf, _kstr.str, _slen+1)))	\
			rd_kafka_buf_parse_fail(			\
				rkbuf,					\
				"Not enough room in tmpabuf: "		\
				"%"PRIusz"+%"PRIusz			\
				" > %"PRIusz,				\
				(tmpabuf)->of, _slen+1, (tmpabuf)->size); \
		_dst[_slen] = '\0';					\
		dst = (void *)_dst;					\
	} while (0)

/**
 * Skip a string.
 */
#define rd_kafka_buf_skip_str(rkbuf) do {			\
		int16_t _slen;					\
		rd_kafka_buf_read_i16(rkbuf, &_slen);		\
		rd_kafka_buf_skip(rkbuf, RD_KAFKAP_STR_LEN0(_slen));	\
	} while (0)

/* Read Kafka Bytes representation (4+N).
 *  The 'kbytes' will be updated to point to rkbuf data */
#define rd_kafka_buf_read_bytes(rkbuf, kbytes) do {		   \
		int _klen;						\
		rd_kafka_buf_read_i32a(rkbuf, _klen);			\
		(kbytes)->len = _klen;					\
		(kbytes)->data = RD_KAFKAP_BYTES_IS_NULL(kbytes) ?	\
			NULL :						\
			(const void *)(rkbuf->rkbuf_rbuf+rkbuf->rkbuf_of); \
		rd_kafka_buf_skip(rkbuf, RD_KAFKAP_BYTES_LEN0(_klen));	\
	} while (0)



/**
 * Response handling callback.
 *
 * NOTE: Callbacks must check for 'err == RD_KAFKA_RESP_ERR__DESTROY'
 *       which indicates that some entity is terminating (rd_kafka_t, broker,
 *       toppar, queue, etc) and the callback may not be called in the
 *       correct thread. In this case the callback must perform just
 *       the most minimal cleanup and dont trigger any other operations.
 *
 * NOTE: rkb, reply and request may be NULL, depending on error situation.
 */
typedef void (rd_kafka_resp_cb_t) (rd_kafka_t *rk,
				   rd_kafka_broker_t *rkb,
                                   rd_kafka_resp_err_t err,
                                   rd_kafka_buf_t *reply,
                                   rd_kafka_buf_t *request,
                                   void *opaque);

struct rd_kafka_buf_s { /* rd_kafka_buf_t */
	TAILQ_ENTRY(rd_kafka_buf_s) rkbuf_link;

	int32_t rkbuf_corrid;

	rd_ts_t rkbuf_ts_retry;    /* Absolute send retry time */

	int     rkbuf_flags; /* RD_KAFKA_OP_F */
	struct msghdr rkbuf_msg;
	struct iovec *rkbuf_iov;
	int           rkbuf_iovcnt;
	int     rkbuf_connid;      /* broker connection id (used when buffer
				    * was partially sent). */
	size_t  rkbuf_of;          /* send: send offset,
				    * recv: parse offset */
	size_t  rkbuf_len;         /* send: total length,
				    * recv: total expected length */
	size_t  rkbuf_size;        /* allocated size */

	char   *rkbuf_buf;         /* Main buffer */
	char   *rkbuf_buf2;        /* Aux buffer (payload receive buffer) */

	char   *rkbuf_rbuf;        /* Read buffer, points to rkbuf_buf or buf2*/
        char   *rkbuf_wbuf;        /* Write buffer pointer (into rkbuf_buf). */
        size_t  rkbuf_wof;         /* Write buffer offset */
	size_t  rkbuf_wof_init;    /* Initial write offset for current iov */

	rd_crc32_t rkbuf_crc;      /* Current CRC calculation */

	struct rd_kafkap_reqhdr rkbuf_reqhdr;   /* Request header.
                                                 * These fields are encoded
                                                 * and written to output buffer
                                                 * on buffer finalization. */
	struct rd_kafkap_reshdr rkbuf_reshdr;   /* Response header.
                                                 * Decoded fields are copied
                                                 * here from the buffer
                                                 * to provide an ease-of-use
                                                 * interface to the header */

	int32_t rkbuf_expected_size;  /* expected size of message */

        rd_kafka_replyq_t   rkbuf_replyq;       /* Enqueue response on replyq */
        rd_kafka_replyq_t   rkbuf_orig_replyq;  /* Original replyq to be used
                                                 * for retries from inside
                                                 * the rkbuf_cb() callback
                                                 * since rkbuf_replyq will
                                                 * have been reset. */
        rd_kafka_resp_cb_t *rkbuf_cb;           /* Response callback */
        struct rd_kafka_buf_s *rkbuf_response;  /* Response buffer */

        rd_kafka_resp_err_t       rkbuf_err;
        struct rd_kafka_broker_s *rkbuf_rkb;

	rd_refcnt_t rkbuf_refcnt;
	void   *rkbuf_opaque;

	int     rkbuf_retries;            /* Retries so far. */
#define RD_KAFKA_BUF_NO_RETRIES  1000000  /* Do not retry */

        int     rkbuf_features;   /* Required feature(s) that must be
                                   * supported by broker. */

	rd_ts_t rkbuf_ts_enq;
	rd_ts_t rkbuf_ts_sent;    /* Initially: Absolute time of transmission,
				   * after response: RTT. */
	rd_ts_t rkbuf_ts_timeout;

        int64_t rkbuf_offset;     /* Used by OffsetCommit */

	rd_list_t *rkbuf_rktp_vers;    /* Toppar + Op Version map.
					* Used by FetchRequest. */

	rd_kafka_msgq_t rkbuf_msgq;

        union {
                struct {
                        rd_list_t *topics;  /* Requested topics (char *) */
                        char *reason;       /* Textual reason */
                        rd_kafka_op_t *rko; /* Originating rko with replyq
                                             * (if any) */
                        int all_topics;     /* Full/All topics requested */

                        int *decr;          /* Decrement this integer by one
                                             * when request is complete:
                                             * typically points to metadata
                                             * cache's full_.._sent.
                                             * Will be performed with
                                             * decr_lock held. */
                        mtx_t *decr_lock;

                } Metadata;
        } rkbuf_u;
};


typedef struct rd_kafka_bufq_s {
	TAILQ_HEAD(, rd_kafka_buf_s) rkbq_bufs;
	rd_atomic32_t  rkbq_cnt;
	rd_atomic32_t  rkbq_msg_cnt;
} rd_kafka_bufq_t;

#define rd_kafka_bufq_cnt(rkbq) rd_atomic32_get(&(rkbq)->rkbq_cnt)


#define rd_kafka_buf_keep(rkbuf) rd_refcnt_add(&(rkbuf)->rkbuf_refcnt)
#define rd_kafka_buf_destroy(rkbuf)                                     \
        rd_refcnt_destroywrapper(&(rkbuf)->rkbuf_refcnt,                \
                                 rd_kafka_buf_destroy_final(rkbuf))

void rd_kafka_buf_destroy_final (rd_kafka_buf_t *rkbuf);
void rd_kafka_buf_auxbuf_add (rd_kafka_buf_t *rkbuf, void *auxbuf);
void rd_kafka_buf_alloc_recvbuf (rd_kafka_buf_t *kbuf, size_t size);
void rd_kafka_buf_rewind(rd_kafka_buf_t *rkbuf, int iovindex, size_t new_of,
	size_t new_of_init);
struct iovec *rd_kafka_buf_iov_next (rd_kafka_buf_t *rkbuf);
void rd_kafka_buf_push0 (rd_kafka_buf_t *rkbuf, const void *buf, size_t len,
			 int allow_crc_calc, int auto_push);
#define rd_kafka_buf_push(rkbuf,buf,len) \
	rd_kafka_buf_push0(rkbuf,buf,len,1/*allow_crc*/, 1)
void rd_kafka_buf_autopush (rd_kafka_buf_t *rkbuf);
rd_kafka_buf_t *rd_kafka_buf_new_growable (const rd_kafka_t *rk, int16_t ApiKey,
                                           int iovcnt, size_t init_size);
rd_kafka_buf_t *rd_kafka_buf_new0 (const rd_kafka_t *rk, int16_t ApiKey,
                                   int iovcnt, size_t size, int flags);
#define rd_kafka_buf_new(rk,ApiKey,iovcnt,size) \
        rd_kafka_buf_new0(rk,ApiKey,iovcnt,size,0)
rd_kafka_buf_t *rd_kafka_buf_new_shadow (const void *ptr, size_t size);
void rd_kafka_bufq_enq (rd_kafka_bufq_t *rkbufq, rd_kafka_buf_t *rkbuf);
void rd_kafka_bufq_deq (rd_kafka_bufq_t *rkbufq, rd_kafka_buf_t *rkbuf);
void rd_kafka_bufq_init(rd_kafka_bufq_t *rkbufq);
void rd_kafka_bufq_concat (rd_kafka_bufq_t *dst, rd_kafka_bufq_t *src);
void rd_kafka_bufq_purge (rd_kafka_broker_t *rkb,
                          rd_kafka_bufq_t *rkbufq,
                          rd_kafka_resp_err_t err);
void rd_kafka_bufq_connection_reset (rd_kafka_broker_t *rkb,
				     rd_kafka_bufq_t *rkbufq);
void rd_kafka_bufq_dump (rd_kafka_broker_t *rkb, const char *fac,
			 rd_kafka_bufq_t *rkbq);

int rd_kafka_buf_retry (rd_kafka_broker_t *rkb, rd_kafka_buf_t *rkbuf);

void rd_kafka_buf_handle_op (rd_kafka_op_t *rko, rd_kafka_resp_err_t err);
void rd_kafka_buf_callback (rd_kafka_t *rk,
			    rd_kafka_broker_t *rkb, rd_kafka_resp_err_t err,
                            rd_kafka_buf_t *response, rd_kafka_buf_t *request);



/**
 *
 * Write buffer interface
 *
 */

/**
 * Set request API type version
 */
static RD_UNUSED RD_INLINE void
rd_kafka_buf_ApiVersion_set (rd_kafka_buf_t *rkbuf,
                             int16_t version, int features) {
        rkbuf->rkbuf_reqhdr.ApiVersion = version;
        rkbuf->rkbuf_features = features;
}

void rd_kafka_buf_grow (rd_kafka_buf_t *rkbuf, size_t needed_len);


/**
 * Set buffer write position.
 */
static RD_INLINE RD_UNUSED void rd_kafka_buf_write_seek (rd_kafka_buf_t *rkbuf,
							int of) {
	rd_kafka_assert(NULL, of >= 0 && of < (int)rkbuf->rkbuf_size);
	rkbuf->rkbuf_wof = of;
}

/**
 * @returns the number of bytes remaining in the write buffer.
 */
static RD_INLINE size_t rd_kafka_buf_write_remain (rd_kafka_buf_t *rkbuf) {
	return rkbuf->rkbuf_size - rkbuf->rkbuf_wof;
}


/**
 * Write (copy) data to buffer at current write-buffer position.
 * There must be enough space allocated in the rkbuf.
 * Returns offset to written destination buffer.
 */
static RD_INLINE size_t rd_kafka_buf_write (rd_kafka_buf_t *rkbuf,
                                        const void *data, size_t len) {
        ssize_t remain = rkbuf->rkbuf_size - (rkbuf->rkbuf_wof + len);

        /* Make sure there's enough room, else increase buffer. */
        if (remain < 0)
                rd_kafka_buf_grow(rkbuf, rkbuf->rkbuf_wof + len);

        rd_kafka_assert(NULL, rkbuf->rkbuf_wof + len <= rkbuf->rkbuf_size);
        memcpy(rkbuf->rkbuf_wbuf + rkbuf->rkbuf_wof, data, len);
        rkbuf->rkbuf_wof += len;

	if (rkbuf->rkbuf_flags & RD_KAFKA_OP_F_CRC)
		rkbuf->rkbuf_crc = rd_crc32_update(rkbuf->rkbuf_crc, data, len);

        return rkbuf->rkbuf_wof - len;
}

/**
 * Returns pointer to buffer at 'offset' and makes sure at least 'len'
 * following bytes are available, else returns NULL.
 */
static RD_INLINE RD_UNUSED void *rd_kafka_buf_at (rd_kafka_buf_t *rkbuf,
						 int of, int len) {
	ssize_t remain = rkbuf->rkbuf_size - (of + len);

	if (remain < 0)
		return NULL;

	return rkbuf->rkbuf_wbuf + of;
}


/**
 * Write (copy) 'data' to buffer at 'ptr'.
 * There must be enough space to fit 'len'.
 * This will overwrite the buffer at given location and length.
 *
 * NOTE: rd_kafka_buf_update() MUST NOT be called when a CRC calculation
 *       is in progress (between rd_kafka_buf_crc_init() & .._crc_finalize())
 */
static RD_INLINE void rd_kafka_buf_update (rd_kafka_buf_t *rkbuf, size_t of,
                                          const void *data, size_t len) {
        ssize_t remain = rkbuf->rkbuf_size - (of + len);
        rd_kafka_assert(NULL, remain >= 0);
        rd_kafka_assert(NULL, of >= 0 && of < rkbuf->rkbuf_size);
	rd_kafka_assert(NULL, !(rkbuf->rkbuf_flags & RD_KAFKA_OP_F_CRC));

        memcpy(rkbuf->rkbuf_wbuf+of, data, len);
}

/**
 * Write int8_t to buffer.
 */
static RD_INLINE size_t rd_kafka_buf_write_i8 (rd_kafka_buf_t *rkbuf,
					      int8_t v) {
        return rd_kafka_buf_write(rkbuf, &v, sizeof(v));
}

/**
 * Update int8_t in buffer at offset 'of'.
 * 'of' should have been previously returned by `.._buf_write_i8()`.
 */
static RD_INLINE void rd_kafka_buf_update_i8 (rd_kafka_buf_t *rkbuf,
					     size_t of, int8_t v) {
        rd_kafka_buf_update(rkbuf, of, &v, sizeof(v));
}

/**
 * Write int16_t to buffer.
 * The value will be endian-swapped before write.
 */
static RD_INLINE size_t rd_kafka_buf_write_i16 (rd_kafka_buf_t *rkbuf,
					       int16_t v) {
        v = htobe16(v);
        return rd_kafka_buf_write(rkbuf, &v, sizeof(v));
}

/**
 * Update int16_t in buffer at offset 'of'.
 * 'of' should have been previously returned by `.._buf_write_i16()`.
 */
static RD_INLINE void rd_kafka_buf_update_i16 (rd_kafka_buf_t *rkbuf,
                                              size_t of, int16_t v) {
        v = htobe16(v);
        rd_kafka_buf_update(rkbuf, of, &v, sizeof(v));
}

/**
 * Write int32_t to buffer.
 * The value will be endian-swapped before write.
 */
static RD_INLINE size_t rd_kafka_buf_write_i32 (rd_kafka_buf_t *rkbuf,
                                               int32_t v) {
        v = htobe32(v);
        return rd_kafka_buf_write(rkbuf, &v, sizeof(v));
}

/**
 * Update int32_t in buffer at offset 'of'.
 * 'of' should have been previously returned by `.._buf_write_i32()`.
 */
static RD_INLINE void rd_kafka_buf_update_i32 (rd_kafka_buf_t *rkbuf,
                                              size_t of, int32_t v) {
        v = htobe32(v);
        rd_kafka_buf_update(rkbuf, of, &v, sizeof(v));
}

/**
 * Update int32_t in buffer at offset 'of'.
 * 'of' should have been previously returned by `.._buf_write_i32()`.
 */
static RD_INLINE void rd_kafka_buf_update_u32 (rd_kafka_buf_t *rkbuf,
                                              size_t of, uint32_t v) {
        v = htobe32(v);
        rd_kafka_buf_update(rkbuf, of, &v, sizeof(v));
}


/**
 * Write int64_t to buffer.
 * The value will be endian-swapped before write.
 */
static RD_INLINE size_t rd_kafka_buf_write_i64 (rd_kafka_buf_t *rkbuf, int64_t v) {
        v = htobe64(v);
        return rd_kafka_buf_write(rkbuf, &v, sizeof(v));
}

/**
 * Update int64_t in buffer at address 'ptr'.
 * 'of' should have been previously returned by `.._buf_write_i64()`.
 */
static RD_INLINE void rd_kafka_buf_update_i64 (rd_kafka_buf_t *rkbuf,
                                              size_t of, int64_t v) {
        v = htobe64(v);
        rd_kafka_buf_update(rkbuf, of, &v, sizeof(v));
}


/**
 * Write (copy) Kafka string to buffer.
 */
static RD_INLINE size_t rd_kafka_buf_write_kstr (rd_kafka_buf_t *rkbuf,
                                                const rd_kafkap_str_t *kstr) {
        return rd_kafka_buf_write(rkbuf, RD_KAFKAP_STR_SER(kstr),
				  RD_KAFKAP_STR_SIZE(kstr));
}

/**
 * Write (copy) char * string to buffer.
 */
static RD_INLINE size_t rd_kafka_buf_write_str (rd_kafka_buf_t *rkbuf,
                                               const char *str, size_t len) {
        size_t r;
        if (!str)
                len = RD_KAFKAP_STR_LEN_NULL;
        else if (len == (size_t)-1)
                len = strlen(str);
        r = rd_kafka_buf_write_i16(rkbuf, (int16_t) len);
        if (str)
                r = rd_kafka_buf_write(rkbuf, str, len);
        return r;
}


/**
 * Push (i.e., no copy) Kafka string to buffer iovec
 */
static RD_INLINE void rd_kafka_buf_push_kstr (rd_kafka_buf_t *rkbuf,
                                             const rd_kafkap_str_t *kstr) {
	rd_kafka_buf_push(rkbuf, RD_KAFKAP_STR_SER(kstr),
			  RD_KAFKAP_STR_SIZE(kstr));
}



/**
 * Write (copy) Kafka bytes to buffer.
 */
static RD_INLINE size_t rd_kafka_buf_write_kbytes (rd_kafka_buf_t *rkbuf,
					          const rd_kafkap_bytes_t *kbytes){
        return rd_kafka_buf_write(rkbuf, RD_KAFKAP_BYTES_SER(kbytes),
                                  RD_KAFKAP_BYTES_SIZE(kbytes));
}

/**
 * Push (i.e., no copy) Kafka bytes to buffer iovec
 */
static RD_INLINE void rd_kafka_buf_push_kbytes (rd_kafka_buf_t *rkbuf,
					       const rd_kafkap_bytes_t *kbytes){
	rd_kafka_buf_push(rkbuf, RD_KAFKAP_BYTES_SER(kbytes),
			  RD_KAFKAP_BYTES_SIZE(kbytes));
}

/**
 * Write (copy) binary bytes to buffer as Kafka bytes encapsulate data.
 */
static RD_INLINE size_t rd_kafka_buf_write_bytes (rd_kafka_buf_t *rkbuf,
                                                 const void *payload, size_t size) {
        size_t r;
        if (!payload)
                size = RD_KAFKAP_BYTES_LEN_NULL;
        r = rd_kafka_buf_write_i32(rkbuf, (int32_t) size);
        if (payload)
                r = rd_kafka_buf_write(rkbuf, payload, size);
        return r;
}




/**
 * Write Kafka Message to buffer
 * The number of bytes written is returned in '*outlenp'.
 *
 * Returns the buffer offset of the first byte.
 */
size_t rd_kafka_buf_write_Message (rd_kafka_broker_t *rkb,
				   rd_kafka_buf_t *rkbuf,
				   int64_t Offset, int8_t MagicByte,
				   int8_t Attributes, int64_t Timestamp,
				   const void *key, int32_t key_len,
				   const void *payload, int32_t len,
				   int *outlenp);

/**
 * Start calculating CRC from now and track it in '*crcp'.
 */
static RD_INLINE RD_UNUSED void rd_kafka_buf_crc_init (rd_kafka_buf_t *rkbuf) {
	rd_kafka_assert(NULL, !(rkbuf->rkbuf_flags & RD_KAFKA_OP_F_CRC));
	rkbuf->rkbuf_flags |= RD_KAFKA_OP_F_CRC;
	rkbuf->rkbuf_crc = rd_crc32_init();
}

/**
 * Finalizes CRC calculation and returns the calculated checksum.
 */
static RD_INLINE RD_UNUSED
rd_crc32_t rd_kafka_buf_crc_finalize (rd_kafka_buf_t *rkbuf) {
	rkbuf->rkbuf_flags &= ~RD_KAFKA_OP_F_CRC;
	return rd_crc32_finalize(rkbuf->rkbuf_crc);
}


/**
 * Returns the number of remaining unused iovecs in buffer.
 */
static RD_INLINE RD_UNUSED
int rd_kafka_buf_iov_remain (const rd_kafka_buf_t *rkbuf) {
	return rkbuf->rkbuf_iovcnt - rkbuf->rkbuf_msg.msg_iovlen;
}


rd_kafkap_bytes_t *rd_kafkap_bytes_from_buf (const rd_kafka_buf_t *rkbuf);

void rd_kafka_buf_hexdump (const char *what, const rd_kafka_buf_t *rkbuf,
			   int read_buffer);


/**
 * @brief Check if buffer's replyq.version is outdated.
 * @param rkbuf: may be NULL, for convenience.
 *
 * @returns 1 if this is an outdated buffer, else 0.
 */
static RD_UNUSED RD_INLINE int
rd_kafka_buf_version_outdated (const rd_kafka_buf_t *rkbuf, int version) {
        return rkbuf && rkbuf->rkbuf_replyq.version &&
                rkbuf->rkbuf_replyq.version < version;
}
