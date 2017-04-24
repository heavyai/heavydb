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

#define __need_IOV_MAX

#ifndef _MSC_VER
#define _GNU_SOURCE
#ifndef _AIX    /* AIX defines this and the value needs to be set correctly */
#define _XOPEN_SOURCE
#endif
#include <signal.h>
#endif

#include <stdio.h>
#include <stdarg.h>
#include <string.h>
#include <ctype.h>

#include "rd.h"
#include "rdkafka_int.h"
#include "rdkafka_msg.h"
#include "rdkafka_topic.h"
#include "rdkafka_partition.h"
#include "rdkafka_broker.h"
#include "rdkafka_offset.h"
#include "rdkafka_transport.h"
#include "rdkafka_proto.h"
#include "rdkafka_buf.h"
#include "rdkafka_request.h"
#include "rdkafka_sasl.h"
#include "rdtime.h"
#include "rdcrc32.h"
#include "rdrand.h"
#if WITH_ZLIB
#include "rdgz.h"
#endif
#if WITH_SNAPPY
#include "snappy.h"
#endif
#if WITH_LZ4_EXT
#include <lz4frame.h>
#else
#include "lz4frame.h"
#endif
#include "xxhash.h"
#if WITH_SSL
#include <openssl/err.h>
#endif
#include "rdendian.h"


const char *rd_kafka_broker_state_names[] = {
	"INIT",
	"DOWN",
	"CONNECT",
	"AUTH",
	"UP",
        "UPDATE",
	"APIVERSION_QUERY",
	"AUTH_HANDSHAKE"
};

const char *rd_kafka_secproto_names[] = {
	[RD_KAFKA_PROTO_PLAINTEXT] = "plaintext",
	[RD_KAFKA_PROTO_SSL] = "ssl",
	[RD_KAFKA_PROTO_SASL_PLAINTEXT] = "sasl_plaintext",
	[RD_KAFKA_PROTO_SASL_SSL] = "sasl_ssl",
	NULL
};





static void iov_print (rd_kafka_t *rk,
		       const char *what, int iov_idx, const struct iovec *iov,
		       int hexdump) {
	printf("%s:  iov #%i: %"PRIusz"\n", what, iov_idx,
	       (size_t)iov->iov_len);
	if (hexdump)
		rd_hexdump(stdout, what, iov->iov_base, iov->iov_len);
}


void msghdr_print (rd_kafka_t *rk,
		   const char *what, const struct msghdr *msg,
		   int hexdump) {
	int i;
	size_t len = 0;

	printf("%s: iovlen %"PRIusz"\n", what, (size_t)msg->msg_iovlen);

	for (i = 0 ; i < (int)msg->msg_iovlen ; i++) {
		iov_print(rk, what, i, &msg->msg_iov[i], hexdump);
		len += msg->msg_iov[i].iov_len;
	}
	printf("%s: ^ message was %"PRIusz" bytes in total\n", what, len);
}



#define rd_kafka_broker_terminating(rkb) \
        (rd_refcnt_get(&(rkb)->rkb_refcnt) <= 1)

static RD_UNUSED size_t rd_kafka_msghdr_size (const struct msghdr *msg) {
	int i;
	size_t tot = 0;

	for (i = 0 ; i < (int)msg->msg_iovlen ; i++)
		tot += msg->msg_iov[i].iov_len;

	return tot;
}


/**
 * Construct broker nodename.
 */
static void rd_kafka_mk_nodename (char *dest, size_t dsize,
                                  const char *name, uint16_t port) {
        rd_snprintf(dest, dsize, "%s:%hu", name, port);
}

/**
 * Construct descriptive broker name
 */
static void rd_kafka_mk_brokername (char *dest, size_t dsize,
				    rd_kafka_secproto_t proto,
				    const char *nodename, int32_t nodeid,
				    rd_kafka_confsource_t source) {

	/* Prepend protocol name to brokername, unless it is a
	 * standard plaintext broker in which case we omit the protocol part. */
	if (proto != RD_KAFKA_PROTO_PLAINTEXT) {
		int r = rd_snprintf(dest, dsize, "%s://",
				    rd_kafka_secproto_names[proto]);
		if (r >= (int)dsize) /* Skip proto name if it wont fit.. */
			r = 0;

		dest += r;
		dsize -= r;
	}

	if (nodeid == RD_KAFKA_NODEID_UA)
		rd_snprintf(dest, dsize, "%s/%s",
			    nodename,
			    source == RD_KAFKA_INTERNAL ?
			    "internal":"bootstrap");
	else
		rd_snprintf(dest, dsize, "%s/%"PRId32, nodename, nodeid);
}


/**
 * @brief Enable protocol feature(s) for the current broker.
 *
 * Locality: broker thread
 */
static void rd_kafka_broker_feature_enable (rd_kafka_broker_t *rkb,
					    int features) {
	if (features & rkb->rkb_features)
		return;

	rkb->rkb_features |= features;
	rd_rkb_dbg(rkb, BROKER | RD_KAFKA_DBG_PROTOCOL | RD_KAFKA_DBG_FEATURE,
		   "FEATURE",
		   "Updated enabled protocol features +%s to %s",
		   rd_kafka_features2str(features),
		   rd_kafka_features2str(rkb->rkb_features));
}


/**
 * @brief Disable protocol feature(s) for the current broker.
 *
 * Locality: broker thread
 */
static void rd_kafka_broker_feature_disable (rd_kafka_broker_t *rkb,
						       int features) {
	if (!(features & rkb->rkb_features))
		return;

	rkb->rkb_features &= ~features;
	rd_rkb_dbg(rkb, BROKER | RD_KAFKA_DBG_PROTOCOL | RD_KAFKA_DBG_FEATURE,
		   "FEATURE",
		   "Updated enabled protocol features -%s to %s",
		   rd_kafka_features2str(features),
		   rd_kafka_features2str(rkb->rkb_features));
}


/**
 * @brief Set protocol feature(s) for the current broker.
 *
 * @remark This replaces the previous feature set.
 *
 * @locality broker thread
 * @locks rd_kafka_broker_lock()
 */
static void rd_kafka_broker_features_set (rd_kafka_broker_t *rkb, int features) {
	if (rkb->rkb_features == features)
		return;

	rkb->rkb_features = features;
	rd_rkb_dbg(rkb, BROKER, "FEATURE",
		   "Updated enabled protocol features to %s",
		   rd_kafka_features2str(rkb->rkb_features));
}


/**
 * @brief Check and return supported ApiVersion for \p ApiKey.
 *
 * @returns the highest supported ApiVersion in the specified range (inclusive)
 *          or -1 if the ApiKey is not supported or no matching ApiVersion.
 *          The current feature set is also returned in \p featuresp
 * @locks none
 * @locality any
 */
int16_t rd_kafka_broker_ApiVersion_supported (rd_kafka_broker_t *rkb,
                                              int16_t ApiKey,
                                              int16_t minver, int16_t maxver,
                                              int *featuresp) {
        struct rd_kafka_ApiVersion skel = { .ApiKey = ApiKey };
        struct rd_kafka_ApiVersion ret, *retp;

        rd_kafka_broker_lock(rkb);
        retp = bsearch(&skel, rkb->rkb_ApiVersions, rkb->rkb_ApiVersions_cnt,
                       sizeof(*rkb->rkb_ApiVersions),
                       rd_kafka_ApiVersion_key_cmp);
        if (retp)
                ret = *retp;
        if (featuresp)
                *featuresp = rkb->rkb_features;
        rd_kafka_broker_unlock(rkb);

        if (!retp)
                return -1;

        if (ret.MaxVer < maxver) {
                if (ret.MaxVer < minver)
                        return -1;
                else
                        return ret.MaxVer;
        } else if (ret.MinVer > maxver)
                return -1;
        else
                return maxver;
}


/**
 * Locks: rd_kafka_broker_lock() MUST be held.
 * Locality: broker thread
 */
void rd_kafka_broker_set_state (rd_kafka_broker_t *rkb, int state) {
	if ((int)rkb->rkb_state == state)
		return;

	rd_kafka_dbg(rkb->rkb_rk, BROKER, "STATE",
		     "%s: Broker changed state %s -> %s",
		     rkb->rkb_name,
		     rd_kafka_broker_state_names[rkb->rkb_state],
		     rd_kafka_broker_state_names[state]);

	if (rkb->rkb_source == RD_KAFKA_INTERNAL) {
		/* no-op */
	} else if (state == RD_KAFKA_BROKER_STATE_DOWN &&
		   !rkb->rkb_down_reported &&
		   rkb->rkb_state != RD_KAFKA_BROKER_STATE_APIVERSION_QUERY) {
		/* Propagate ALL_BROKERS_DOWN event if all brokers are
		 * now down, unless we're terminating.
		 * Dont do this if we're querying for ApiVersion since it
		 * is bound to fail once on older brokers. */
		if (rd_atomic32_add(&rkb->rkb_rk->rk_broker_down_cnt, 1) ==
		    rd_atomic32_get(&rkb->rkb_rk->rk_broker_cnt) &&
		    !rd_atomic32_get(&rkb->rkb_rk->rk_terminate))
			rd_kafka_op_err(rkb->rkb_rk,
					RD_KAFKA_RESP_ERR__ALL_BROKERS_DOWN,
					"%i/%i brokers are down",
					rd_atomic32_get(&rkb->rkb_rk->
                                                        rk_broker_down_cnt),
					rd_atomic32_get(&rkb->rkb_rk->
                                                        rk_broker_cnt));
		rkb->rkb_down_reported = 1;

	} else if (rkb->rkb_state >= RD_KAFKA_BROKER_STATE_UP &&
		   rkb->rkb_down_reported) {
		rd_atomic32_sub(&rkb->rkb_rk->rk_broker_down_cnt, 1);
		rkb->rkb_down_reported = 0;
	}

	rkb->rkb_state = state;
        rkb->rkb_ts_state = rd_clock();

	rd_kafka_brokers_broadcast_state_change(rkb->rkb_rk);
}


/**
 * @brief Locks broker, acquires the states, unlocks, and returns
 *        the state.
 * @locks !broker_lock
 * @locality any
 */
int rd_kafka_broker_get_state (rd_kafka_broker_t *rkb) {
        int state;
        rd_kafka_broker_lock(rkb);
        state = rkb->rkb_state;
        rd_kafka_broker_unlock(rkb);
        return state;
}


/**
 * Failure propagation to application.
 * Will tear down connection to broker and trigger a reconnect.
 *
 * If 'fmt' is NULL nothing will be logged or propagated to the application.
 *
 * \p level is the log level, <=LOG_INFO will be logged while =LOG_DEBUG will
 * be debug-logged.
 * 
 * Locality: Broker thread
 */
void rd_kafka_broker_fail (rd_kafka_broker_t *rkb,
                           int level, rd_kafka_resp_err_t err,
			   const char *fmt, ...) {
	va_list ap;
	int errno_save = errno;
	rd_kafka_bufq_t tmpq_waitresp, tmpq;
        int old_state;

	rd_kafka_assert(rkb->rkb_rk, thrd_is_current(rkb->rkb_thread));

	rd_kafka_dbg(rkb->rkb_rk, BROKER | RD_KAFKA_DBG_PROTOCOL, "BROKERFAIL",
		     "%s: failed: err: %s: (errno: %s)",
		     rkb->rkb_name, rd_kafka_err2str(err),
		     rd_strerror(errno_save));

	rkb->rkb_err.err = errno_save;

	if (rkb->rkb_transport) {
		rd_kafka_transport_close(rkb->rkb_transport);
		rkb->rkb_transport = NULL;
	}

	rkb->rkb_req_timeouts = 0;

	if (rkb->rkb_recv_buf) {
		rd_kafka_buf_destroy(rkb->rkb_recv_buf);
		rkb->rkb_recv_buf = NULL;
	}

	rd_kafka_broker_lock(rkb);

	/* The caller may omit the format if it thinks this is a recurring
	 * failure, in which case the following things are omitted:
	 *  - log message
	 *  - application OP_ERR
	 *  - metadata request
	 *
	 * Dont log anything if this was the termination signal, or if the
	 * socket disconnected while trying ApiVersionRequest.
	 */
	if (fmt &&
	    !(errno_save == EINTR &&
	      rd_atomic32_get(&rkb->rkb_rk->rk_terminate)) &&
	    !(err == RD_KAFKA_RESP_ERR__TRANSPORT &&
	      rkb->rkb_state == RD_KAFKA_BROKER_STATE_APIVERSION_QUERY)) {
		int of;

		/* Insert broker name in log message if it fits. */
		of = rd_snprintf(rkb->rkb_err.msg, sizeof(rkb->rkb_err.msg),
			      "%s: ", rkb->rkb_name);
		if (of >= (int)sizeof(rkb->rkb_err.msg))
			of = 0;
		va_start(ap, fmt);
		rd_vsnprintf(rkb->rkb_err.msg+of,
			  sizeof(rkb->rkb_err.msg)-of, fmt, ap);
		va_end(ap);

                if (level >= LOG_DEBUG)
                        rd_kafka_dbg(rkb->rkb_rk, BROKER, "FAIL",
                                     "%s", rkb->rkb_err.msg);
                else {
                        /* Don't log if an error callback is registered */
                        if (!rkb->rkb_rk->rk_conf.error_cb)
                                rd_kafka_log(rkb->rkb_rk, level, "FAIL",
                                             "%s", rkb->rkb_err.msg);
                        /* Send ERR op back to application for processing. */
                        rd_kafka_op_err(rkb->rkb_rk, err,
                                        "%s", rkb->rkb_err.msg);
                }
	}

	/* If we're currently asking for ApiVersion and the connection
	 * went down it probably means the broker does not support that request
	 * and tore down the connection. In this case we disable that feature flag. */
	if (rkb->rkb_state == RD_KAFKA_BROKER_STATE_APIVERSION_QUERY)
		rd_kafka_broker_feature_disable(rkb, RD_KAFKA_FEATURE_APIVERSION);

	/* Set broker state */
        old_state = rkb->rkb_state;
	rd_kafka_broker_set_state(rkb, RD_KAFKA_BROKER_STATE_DOWN);

	/* Unlock broker since a requeue will try to lock it. */
	rd_kafka_broker_unlock(rkb);

	/*
	 * Purge all buffers
	 * (put bufs on a temporary queue since bufs may be requeued,
	 *  make sure outstanding requests are re-enqueued before
	 *  bufs on outbufs queue.)
	 */
	rd_kafka_bufq_init(&tmpq_waitresp);
	rd_kafka_bufq_init(&tmpq);
	rd_kafka_bufq_concat(&tmpq_waitresp, &rkb->rkb_waitresps);
	rd_kafka_bufq_concat(&tmpq, &rkb->rkb_outbufs);
        rd_atomic32_init(&rkb->rkb_blocking_request_cnt, 0);

	/* Purge the buffers (might get re-enqueued in case of retries) */
	rd_kafka_bufq_purge(rkb, &tmpq_waitresp, err);

	/* Put the outbufs back on queue */
	rd_kafka_bufq_concat(&rkb->rkb_outbufs, &tmpq);

	/* Update bufq for connection reset:
	 *  - Purge connection-setup requests from outbufs since they will be
	 *    reissued on the next connect.
	 *  - Reset any partially sent buffer's offset.
	 */
	rd_kafka_bufq_connection_reset(rkb, &rkb->rkb_outbufs);

	/* Extra debugging for tracking termination-hang issues:
	 * show what is keeping this broker from decommissioning. */
	if (rd_kafka_terminating(rkb->rkb_rk) &&
	    !rd_kafka_broker_terminating(rkb)) {
		rd_rkb_dbg(rkb, BROKER | RD_KAFKA_DBG_PROTOCOL, "BRKTERM",
			   "terminating: broker still has %d refcnt(s), "
			   "%"PRId32" buffer(s), %d partition(s)",
			   rd_refcnt_get(&rkb->rkb_refcnt),
			   rd_kafka_bufq_cnt(&rkb->rkb_outbufs),
			   rkb->rkb_toppar_cnt);
		rd_kafka_bufq_dump(rkb, "BRKOUTBUFS", &rkb->rkb_outbufs);
#if ENABLE_SHAREDPTR_DEBUG
		if (rd_refcnt_get(&rkb->rkb_refcnt) > 1) {
			rd_rkb_dbg(rkb, BROKER, "BRKTERM",
				   "Dumping shared pointers: "
				   "this broker is %p", rkb);
			rd_shared_ptrs_dump();
		}
#endif
	}


        /* Query for topic leaders to quickly pick up on failover. */
        if (fmt && err != RD_KAFKA_RESP_ERR__DESTROY &&
            old_state >= RD_KAFKA_BROKER_STATE_UP)
                rd_kafka_metadata_refresh_known_topics(rkb->rkb_rk, NULL,
                                                       1/*force*/,
                                                       "broker down");
}





/**
 * Scan bufq for buffer timeouts, trigger buffer callback on timeout.
 *
 * If \p partial_cntp is non-NULL any partially sent buffers will increase
 * the provided counter by 1.
 *
 * @returns the number of timed out buffers.
 *
 * @locality broker thread
 */
static int rd_kafka_broker_bufq_timeout_scan (rd_kafka_broker_t *rkb,
					      int is_waitresp_q,
					      rd_kafka_bufq_t *rkbq,
					      int *partial_cntp,
					      rd_kafka_resp_err_t err,
					      rd_ts_t now) {
	rd_kafka_buf_t *rkbuf, *tmp;
	int cnt = 0;

	TAILQ_FOREACH_SAFE(rkbuf, &rkbq->rkbq_bufs, rkbuf_link, tmp) {

		if (likely(now && rkbuf->rkbuf_ts_timeout > now))
			continue;

		if (partial_cntp && rkbuf->rkbuf_of)
			(*partial_cntp)++;

		/* Convert rkbuf_ts_sent to elapsed time since request */
		if (rkbuf->rkbuf_ts_sent)
			rkbuf->rkbuf_ts_sent = now - rkbuf->rkbuf_ts_sent;
		else
			rkbuf->rkbuf_ts_sent = now - rkbuf->rkbuf_ts_enq;

		rd_kafka_bufq_deq(rkbq, rkbuf);

		if (is_waitresp_q && rkbuf->rkbuf_flags & RD_KAFKA_OP_F_BLOCKING
		    && rd_atomic32_sub(&rkb->rkb_blocking_request_cnt, 1) == 0)
			rd_kafka_brokers_broadcast_state_change(rkb->rkb_rk);

                rd_kafka_buf_callback(rkb->rkb_rk, rkb, err, NULL, rkbuf);
		cnt++;
	}

	return cnt;
}


/**
 * Scan the wait-response and outbuf queues for message timeouts.
 *
 * Locality: Broker thread
 */
static void rd_kafka_broker_timeout_scan (rd_kafka_broker_t *rkb, rd_ts_t now) {
	int req_cnt, retry_cnt, q_cnt;

	rd_kafka_assert(rkb->rkb_rk, thrd_is_current(rkb->rkb_thread));

	/* Outstanding requests waiting for response */
	req_cnt = rd_kafka_broker_bufq_timeout_scan(
		rkb, 1, &rkb->rkb_waitresps, NULL,
		RD_KAFKA_RESP_ERR__TIMED_OUT, now);
	/* Requests in retry queue */
	retry_cnt = rd_kafka_broker_bufq_timeout_scan(
		rkb, 0, &rkb->rkb_retrybufs, NULL,
		RD_KAFKA_RESP_ERR__TIMED_OUT, now);
	/* Requests in local queue not sent yet. */
	q_cnt = rd_kafka_broker_bufq_timeout_scan(
		rkb, 0, &rkb->rkb_outbufs, &req_cnt,
		RD_KAFKA_RESP_ERR__TIMED_OUT, now);

	if (req_cnt + retry_cnt + q_cnt > 0) {
		rd_rkb_dbg(rkb, MSG|RD_KAFKA_DBG_BROKER,
			   "REQTMOUT", "Timed out %i+%i+%i requests",
			   req_cnt, retry_cnt, q_cnt);

                /* Fail the broker if socket.max.fails is configured and
                 * now exceeded. */
                rkb->rkb_req_timeouts   += req_cnt + q_cnt;
                rd_atomic64_add(&rkb->rkb_c.req_timeouts, req_cnt + q_cnt);

		/* If this was an in-flight request that timed out, or
		 * the other queues has reached the socket.max.fails threshold,
		 * we need to take down the connection. */
                if ((req_cnt > 0 ||
		     (rkb->rkb_rk->rk_conf.socket_max_fails &&
		      rkb->rkb_req_timeouts >=
		      rkb->rkb_rk->rk_conf.socket_max_fails)) &&
                    rkb->rkb_state >= RD_KAFKA_BROKER_STATE_UP) {
                        char rttinfo[32];
                        /* Print average RTT (if avail) to help diagnose. */
                        rd_avg_calc(&rkb->rkb_avg_rtt, now);
                        if (rkb->rkb_avg_rtt.ra_v.avg)
                                rd_snprintf(rttinfo, sizeof(rttinfo),
                                            " (average rtt %.3fms)",
                                            (float)(rkb->rkb_avg_rtt.ra_v.avg/
                                                    1000.0f));
                        else
                                rttinfo[0] = 0;
                        errno = ETIMEDOUT;
                        rd_kafka_broker_fail(rkb, LOG_ERR,
                                             RD_KAFKA_RESP_ERR__MSG_TIMED_OUT,
                                             "%i request(s) timed out: "
                                             "disconnect%s",
                                             rkb->rkb_req_timeouts, rttinfo);
                }
        }
}



static ssize_t rd_kafka_broker_send (rd_kafka_broker_t *rkb,
				     const struct msghdr *msg) {
	ssize_t r;
	char errstr[128];

	rd_kafka_assert(rkb->rkb_rk, rkb->rkb_state >= RD_KAFKA_BROKER_STATE_UP);
	rd_kafka_assert(rkb->rkb_rk, rkb->rkb_transport);

	r = rd_kafka_transport_sendmsg(rkb->rkb_transport, msg, errstr, sizeof(errstr));

	if (r == -1) {
		rd_kafka_broker_fail(rkb, LOG_ERR, RD_KAFKA_RESP_ERR__TRANSPORT,
                                     "Send failed: %s", errstr);
		rd_atomic64_add(&rkb->rkb_c.tx_err, 1);
		return -1;
	}

	rd_atomic64_add(&rkb->rkb_c.tx_bytes, r);
	rd_atomic64_add(&rkb->rkb_c.tx, 1);
	return r;
}




static int rd_kafka_broker_resolve (rd_kafka_broker_t *rkb) {
	const char *errstr;

	if (rkb->rkb_rsal &&
	    rkb->rkb_t_rsal_last + rkb->rkb_rk->rk_conf.broker_addr_ttl <
	    time(NULL)) { // FIXME: rd_clock()
		/* Address list has expired. */
		rd_sockaddr_list_destroy(rkb->rkb_rsal);
		rkb->rkb_rsal = NULL;
	}

	if (!rkb->rkb_rsal) {
		/* Resolve */

		rkb->rkb_rsal = rd_getaddrinfo(rkb->rkb_nodename,
					       RD_KAFKA_PORT_STR,
					       AI_ADDRCONFIG,
					       rkb->rkb_rk->rk_conf.
                                               broker_addr_family,
                                               SOCK_STREAM,
					       IPPROTO_TCP, &errstr);

		if (!rkb->rkb_rsal) {
                        rd_kafka_broker_fail(rkb, LOG_ERR,
                                             RD_KAFKA_RESP_ERR__RESOLVE,
                                             /* Avoid duplicate log messages */
                                             rkb->rkb_err.err == errno ?
                                             NULL :
                                             "Failed to resolve '%s': %s",
                                             rkb->rkb_nodename, errstr);
			return -1;
		}
	}

	return 0;
}


static void rd_kafka_broker_buf_enq0 (rd_kafka_broker_t *rkb,
				      rd_kafka_buf_t *rkbuf, int at_head) {
	rd_kafka_assert(rkb->rkb_rk, thrd_is_current(rkb->rkb_thread));

        rkbuf->rkbuf_ts_enq = rd_clock();

        /* Set timeout if not already set */
        if (!rkbuf->rkbuf_ts_timeout)
        	rkbuf->rkbuf_ts_timeout = rkbuf->rkbuf_ts_enq +
                        rkb->rkb_rk->rk_conf.socket_timeout_ms * 1000;

	if (unlikely(at_head)) {
		/* Insert message at head of queue */
		rd_kafka_buf_t *prev, *after = NULL;

		/* Put us behind any flash messages and partially sent buffers.
		 * We need to check if buf corrid is set rather than
		 * rkbuf_of since SSL_write may return 0 and expect the
		 * exact same arguments the next call. */
		TAILQ_FOREACH(prev, &rkb->rkb_outbufs.rkbq_bufs, rkbuf_link) {
			if (!(prev->rkbuf_flags & RD_KAFKA_OP_F_FLASH) &&
			    prev->rkbuf_corrid == 0)
				break;
			after = prev;
		}

		if (after)
			TAILQ_INSERT_AFTER(&rkb->rkb_outbufs.rkbq_bufs,
					   after, rkbuf, rkbuf_link);
		else
			TAILQ_INSERT_HEAD(&rkb->rkb_outbufs.rkbq_bufs,
					  rkbuf, rkbuf_link);
	} else {
		/* Insert message at tail of queue */
		TAILQ_INSERT_TAIL(&rkb->rkb_outbufs.rkbq_bufs,
				  rkbuf, rkbuf_link);
	}

	(void)rd_atomic32_add(&rkb->rkb_outbufs.rkbq_cnt, 1);
	(void)rd_atomic32_add(&rkb->rkb_outbufs.rkbq_msg_cnt,
                            rd_atomic32_get(&rkbuf->rkbuf_msgq.rkmq_msg_cnt));
}


/**
 * Finalize a stuffed rkbuf for sending to broker.
 */
static void rd_kafka_buf_finalize (rd_kafka_t *rk, rd_kafka_buf_t *rkbuf) {
	size_t of_Size;

	/* Autopush final buffer work space if not already done. */
	rd_kafka_buf_autopush(rkbuf);

	/* Write header */
	rd_kafka_buf_write_seek(rkbuf, 0);
	of_Size = rd_kafka_buf_write_i32(rkbuf, 0); /* Size: Updated below */
	rd_kafka_buf_write_i16(rkbuf, rkbuf->rkbuf_reqhdr.ApiKey);
	rd_kafka_buf_write_i16(rkbuf, rkbuf->rkbuf_reqhdr.ApiVersion);
	rd_kafka_buf_write_i32(rkbuf, 0); /* CorrId: Updated in enq0() */

	/* Write clientId */
        rd_kafka_buf_write_kstr(rkbuf, rk->rk_conf.client_id);

	/* Calculate total message buffer length. */
	rkbuf->rkbuf_of          = 0;  /* Indicates send position */

	rd_dassert(rkbuf->rkbuf_len == rd_kafka_msghdr_size(&rkbuf->rkbuf_msg));

	rd_kafka_assert(NULL,
			rkbuf->rkbuf_len-4 < (size_t)rk->rk_conf.max_msg_size);
	rd_kafka_buf_update_i32(rkbuf, of_Size, (int32_t) rkbuf->rkbuf_len-4);
}


void rd_kafka_broker_buf_enq1 (rd_kafka_broker_t *rkb,
                               rd_kafka_buf_t *rkbuf,
                               rd_kafka_resp_cb_t *resp_cb,
                               void *opaque) {


        rkbuf->rkbuf_cb     = resp_cb;
	rkbuf->rkbuf_opaque = opaque;

        rd_kafka_buf_finalize(rkb->rkb_rk, rkbuf);

	rd_kafka_broker_buf_enq0(rkb, rkbuf,
				 (rkbuf->rkbuf_flags & RD_KAFKA_OP_F_FLASH)?
				 1/*head*/: 0/*tail*/);
}


/**
 * Enqueue buffer on broker's xmit queue, but fail buffer immediately
 * if broker is not up.
 *
 * Locality: broker thread
 */
static int rd_kafka_broker_buf_enq2 (rd_kafka_broker_t *rkb,
				      rd_kafka_buf_t *rkbuf) {
        if (unlikely(rkb->rkb_source == RD_KAFKA_INTERNAL)) {
                /* Fail request immediately if this is the internal broker. */
                rd_kafka_buf_callback(rkb->rkb_rk, rkb,
				      RD_KAFKA_RESP_ERR__TRANSPORT,
                                      NULL, rkbuf);
                return -1;
        }

	rd_kafka_broker_buf_enq0(rkb, rkbuf,
				 (rkbuf->rkbuf_flags & RD_KAFKA_OP_F_FLASH)?
				 1/*head*/: 0/*tail*/);

	return 0;
}



/**
 * Enqueue buffer for tranmission.
 * Responses are enqueued on 'replyq' (RD_KAFKA_OP_RECV_BUF)
 *
 * Locality: any thread
 */
void rd_kafka_broker_buf_enq_replyq (rd_kafka_broker_t *rkb,
                                     rd_kafka_buf_t *rkbuf,
                                     rd_kafka_replyq_t replyq,
                                     rd_kafka_resp_cb_t *resp_cb,
                                     void *opaque) {

        assert(rkbuf->rkbuf_rkb == NULL);
        rkbuf->rkbuf_rkb = rkb;
        rd_kafka_broker_keep(rkb);
        if (resp_cb) {
                rkbuf->rkbuf_replyq = replyq;
                rkbuf->rkbuf_cb     = resp_cb;
                rkbuf->rkbuf_opaque = opaque;
        } else {
		rd_dassert(!replyq.q);
	}

        rd_kafka_buf_finalize(rkb->rkb_rk, rkbuf);


	if (thrd_is_current(rkb->rkb_thread)) {
		rd_kafka_broker_buf_enq2(rkb, rkbuf);

	} else {
		rd_kafka_op_t *rko = rd_kafka_op_new(RD_KAFKA_OP_XMIT_BUF);
		rko->rko_u.xbuf.rkbuf = rkbuf;
		rd_kafka_q_enq(rkb->rkb_ops, rko);
	}
}




/**
 * @returns the current broker state change version.
 *          Pass this value to fugure rd_kafka_brokers_wait_state_change() calls
 *          to avoid the race condition where a state-change happens between
 *          an initial call to some API that fails and the sub-sequent
 *          .._wait_state_change() call.
 */
int rd_kafka_brokers_get_state_version (rd_kafka_t *rk) {
	int version;
	mtx_lock(&rk->rk_broker_state_change_lock);
	version = rk->rk_broker_state_change_version;
	mtx_unlock(&rk->rk_broker_state_change_lock);
	return version;
}

/**
 * @brief Wait at most \p timeout_ms for any state change for any broker.
 *        \p stored_version is the value previously returned by
 *        rd_kafka_brokers_get_state_version() prior to another API call
 *        that failed due to invalid state.
 *
 * Triggers:
 *   - broker state changes
 *   - broker transitioning from blocking to non-blocking
 *   - partition leader changes
 *   - group state changes
 *
 * @remark There is no guarantee that a state change actually took place.
 *
 * @returns 1 if a state change was signaled (maybe), else 0 (timeout)
 *
 * @locality any thread
 */
int rd_kafka_brokers_wait_state_change (rd_kafka_t *rk, int stored_version,
					int timeout_ms) {
	int r;
	mtx_lock(&rk->rk_broker_state_change_lock);
	if (stored_version != rk->rk_broker_state_change_version)
		r = 1;
	else
		r = cnd_timedwait_ms(&rk->rk_broker_state_change_cnd,
				     &rk->rk_broker_state_change_lock,
				     timeout_ms) == thrd_success;
	mtx_unlock(&rk->rk_broker_state_change_lock);
	return r;
}


/**
 * @brief Broadcast broker state change to listeners, if any.
 *
 * @locality any thread
 */
void rd_kafka_brokers_broadcast_state_change (rd_kafka_t *rk) {
	rd_kafka_dbg(rk, GENERIC, "BROADCAST",
		     "Broadcasting state change");
	mtx_lock(&rk->rk_broker_state_change_lock);
	rk->rk_broker_state_change_version++;
	cnd_broadcast(&rk->rk_broker_state_change_cnd);
	mtx_unlock(&rk->rk_broker_state_change_lock);
}


/**
 * Returns a random broker (with refcnt increased) in state 'state'.
 * Uses Reservoir sampling.
 *
 * 'filter' is an optional callback used to filter out undesired brokers.
 * The filter function should return 1 to filter out a broker, or 0 to keep it
 * in the list of eligible brokers to return.
 * rd_kafka_broker_lock() is held during the filter callback.
 *
 * Locks: rd_kafka_rdlock(rk) MUST be held.
 * Locality: any thread
 */
rd_kafka_broker_t *rd_kafka_broker_any (rd_kafka_t *rk, int state,
                                        int (*filter) (rd_kafka_broker_t *rkb,
                                                       void *opaque),
                                        void *opaque) {
	rd_kafka_broker_t *rkb, *good = NULL;
        int cnt = 0;

	TAILQ_FOREACH(rkb, &rk->rk_brokers, rkb_link) {
		rd_kafka_broker_lock(rkb);
		if ((int)rkb->rkb_state == state &&
                    (!filter || !filter(rkb, opaque))) {
                        if (cnt < 1 || rd_jitter(0, cnt) < 1) {
                                if (good)
                                        rd_kafka_broker_destroy(good);
                                rd_kafka_broker_keep(rkb);
                                good = rkb;
                        }
                        cnt += 1;
                }
		rd_kafka_broker_unlock(rkb);
	}

        return good;
}


/**
 * @brief Spend at most \p timeout_ms to acquire a usable (Up && non-blocking)
 *        broker.
 *
 * @returns A probably usable broker with increased refcount, or NULL on timeout
 * @locks rd_kafka_*lock() if !do_lock
 * @locality any
 */
rd_kafka_broker_t *rd_kafka_broker_any_usable (rd_kafka_t *rk,
                                                int timeout_ms,
                                                int do_lock) {
	const rd_ts_t ts_end = rd_timeout_init(timeout_ms);

	while (1) {
		rd_kafka_broker_t *rkb;
		int remains;
		int version = rd_kafka_brokers_get_state_version(rk);

                /* Try non-blocking (e.g., non-fetching) brokers first. */
                if (do_lock)
                        rd_kafka_rdlock(rk);
                rkb = rd_kafka_broker_any(rk, RD_KAFKA_BROKER_STATE_UP,
                                          rd_kafka_broker_filter_non_blocking,
                                          NULL);
                if (!rkb)
                        rkb = rd_kafka_broker_any(rk, RD_KAFKA_BROKER_STATE_UP,
                                                  NULL, NULL);
                if (do_lock)
                        rd_kafka_rdunlock(rk);

                if (rkb)
                        return rkb;

		remains = rd_timeout_remains(ts_end);
		if (rd_timeout_expired(remains))
			return NULL;

		rd_kafka_brokers_wait_state_change(rk, version, remains);
	}

	return NULL;
}



/**
 * Returns a broker in state `state`, preferring the one with
 * matching `broker_id`.
 * Uses Reservoir sampling.
 *
 * Locks: rd_kafka_rdlock(rk) MUST be held.
 * Locality: any thread
 */
rd_kafka_broker_t *rd_kafka_broker_prefer (rd_kafka_t *rk, int32_t broker_id,
					   int state) {
	rd_kafka_broker_t *rkb, *good = NULL;
        int cnt = 0;

	TAILQ_FOREACH(rkb, &rk->rk_brokers, rkb_link) {
		rd_kafka_broker_lock(rkb);
		if ((int)rkb->rkb_state == state) {
                        if (broker_id != -1 && rkb->rkb_nodeid == broker_id) {
                                if (good)
                                        rd_kafka_broker_destroy(good);
                                rd_kafka_broker_keep(rkb);
                                good = rkb;
                                rd_kafka_broker_unlock(rkb);
                                break;
                        }
                        if (cnt < 1 || rd_jitter(0, cnt) < 1) {
                                if (good)
                                        rd_kafka_broker_destroy(good);
                                rd_kafka_broker_keep(rkb);
                                good = rkb;
                        }
                        cnt += 1;
                }
		rd_kafka_broker_unlock(rkb);
	}

        return good;
}






/**
 * Find a waitresp (rkbuf awaiting response) by the correlation id.
 */
static rd_kafka_buf_t *rd_kafka_waitresp_find (rd_kafka_broker_t *rkb,
					       int32_t corrid) {
	rd_kafka_buf_t *rkbuf;
	rd_ts_t now = rd_clock();

	rd_kafka_assert(rkb->rkb_rk, thrd_is_current(rkb->rkb_thread));

	TAILQ_FOREACH(rkbuf, &rkb->rkb_waitresps.rkbq_bufs, rkbuf_link)
		if (rkbuf->rkbuf_corrid == corrid) {
			/* Convert ts_sent to RTT */
			rkbuf->rkbuf_ts_sent = now - rkbuf->rkbuf_ts_sent;
			rd_avg_add(&rkb->rkb_avg_rtt, rkbuf->rkbuf_ts_sent);

                        if (rkbuf->rkbuf_flags & RD_KAFKA_OP_F_BLOCKING &&
			    rd_atomic32_sub(&rkb->rkb_blocking_request_cnt,
					    1) == 1)
				rd_kafka_brokers_broadcast_state_change(
					rkb->rkb_rk);

			rd_kafka_bufq_deq(&rkb->rkb_waitresps, rkbuf);
			return rkbuf;
		}
	return NULL;
}




/**
 * Map a response message to a request.
 */
static int rd_kafka_req_response (rd_kafka_broker_t *rkb,
				  rd_kafka_buf_t *rkbuf) {
	rd_kafka_buf_t *req;

	rd_kafka_assert(rkb->rkb_rk, thrd_is_current(rkb->rkb_thread));


	/* Find corresponding request message by correlation id */
	if (unlikely(!(req =
		       rd_kafka_waitresp_find(rkb,
					      rkbuf->rkbuf_reshdr.CorrId)))) {
		/* unknown response. probably due to request timeout */
                rd_atomic64_add(&rkb->rkb_c.rx_corrid_err, 1);
		rd_rkb_dbg(rkb, BROKER, "RESPONSE",
			   "Response for unknown CorrId %"PRId32" (timed out?)",
			   rkbuf->rkbuf_reshdr.CorrId);
                rd_kafka_buf_destroy(rkbuf);
                return -1;
	}

	rd_rkb_dbg(rkb, PROTOCOL, "RECV",
		   "Received %sResponse (v%hd, %"PRIusz" bytes, CorrId %"PRId32
		   ", rtt %.2fms)",
		   rd_kafka_ApiKey2str(req->rkbuf_reqhdr.ApiKey),
                   req->rkbuf_reqhdr.ApiVersion,
		   rkbuf->rkbuf_len, rkbuf->rkbuf_reshdr.CorrId,
		   (float)req->rkbuf_ts_sent / 1000.0f);

	/* Call callback. */
        rd_kafka_buf_callback(rkb->rkb_rk, rkb, 0, rkbuf, req);

	return 0;
}


/**
 * Rebuilds 'src' into 'dst' starting at byte offset 'of'.
 */
static void rd_kafka_msghdr_rebuild (struct msghdr *dst, size_t dst_len,
				     size_t max_bytes,
				     struct msghdr *src, ssize_t of) {
	struct iovec *siov = src->msg_iov, *s_end = &src->msg_iov[src->msg_iovlen];
	struct iovec *diov = dst->msg_iov, *d_end = &dst->msg_iov[dst_len];

	*dst = *src;
	dst->msg_iov = diov;
	dst->msg_iovlen = 0;

	for ( ; max_bytes > 0 && siov < s_end && diov < d_end ; siov++) {
		ssize_t rlen;

		if (of >= (ssize_t)siov->iov_len) {
			of -= siov->iov_len;
			continue;
		}

		rlen = RD_MIN(siov->iov_len - of, max_bytes);

		diov->iov_base = ((char *)(siov->iov_base)) + of;
		diov->iov_len  = rlen;
		rd_dassert(rlen > 0);
		max_bytes     -= rlen;
		of             = 0;
		dst->msg_iovlen++;
		diov++;
	}
}



int rd_kafka_recv (rd_kafka_broker_t *rkb) {
	rd_kafka_buf_t *rkbuf;
	ssize_t r;
	struct msghdr msg;
	struct iovec iov;
	char errstr[512];
	rd_kafka_resp_err_t err_code = RD_KAFKA_RESP_ERR__BAD_MSG;
	const int log_decode_errors = 1;

	/**
	 * The receive buffers are split up in two parts:
	 *   - the first part is mainly for reading the first 4 bytes
	 *     where the remaining length is coded.
	 *     But for short packets we want to avoid a second recv() call
	 *     so the first buffer should be large enough for common short
	 *     packets.
	 *     This is iov[0] and iov[1].
	 *
	 *   - the second part is mainly for data response, this buffer
	 *     must be contigious and will be provided to the application
	 *     as is (Fetch response).
	 *     This is iov[2].
	 *
	 *   It is impossible to estimate the correct size of the first
	 *   buffer, so we make it big enough to probably fit all kinds of
	 *   non-data responses so we dont have to allocate a second buffer
	 *   for such responses. And we make it small enough that a copy
	 *   to the second buffer isn't too costly in case we receive a
	 *   real data packet.
	 *
	 * Minimum packet sizes per response type:
	 *   Metadata: 4+4+2+host+4+4+2+2+topic+2+4+4+4+4+4+4.. =~ 48
	 *   Produce:  4+2+topic+4+4+2+8.. =~ 24
	 *   Fetch:    4+2+topic+4+4+2+8+8+4.. =~ 36
	 *   Offset:   4+2+topic+4+4+2+4+8.. =~ 28
	 *   ...
	 *
	 * Plus 4 + 4 for Size and CorrId.
	 *
	 * First buffer size should thus be: 96 bytes
	 */
	/* FIXME: skip the above, just go for the header. */
	if (!(rkbuf = rkb->rkb_recv_buf)) {
		/* No receive in progress: new message. */

		rkbuf = rd_kafka_buf_new(rkb->rkb_rk, RD_KAFKAP_None, 0, 0);

		/* The iov[0] buffer is already allocated by buf_new(),
		 * shrink it to only allow for the response header. */
		rkbuf->rkbuf_iov[0].iov_len = RD_KAFKAP_RESHDR_SIZE;
		rkbuf->rkbuf_wof = 0;

		rkbuf->rkbuf_msg.msg_iov = rkbuf->rkbuf_iov;
		rkbuf->rkbuf_msg.msg_iovlen = 1;
		rkbuf->rkbuf_len = 0; /* read bytes is zero */

		msg = rkbuf->rkbuf_msg;

		/* Point read buffer to main buffer. */
		rkbuf->rkbuf_rbuf = rkbuf->rkbuf_buf;

		rkb->rkb_recv_buf = rkbuf;

	} else {
		/* Receive in progress: adjust the msg to allow more data. */
		msg.msg_iov = &iov;
		rd_kafka_msghdr_rebuild(&msg, rkbuf->rkbuf_msg.msg_iovlen,
					rkb->rkb_rk->rk_conf.recv_max_msg_size,
					&rkbuf->rkbuf_msg,
					rkbuf->rkbuf_wof);
	}

	rd_dassert(rd_kafka_msghdr_size(&msg) > 0);

	r = rd_kafka_transport_recvmsg(rkb->rkb_transport, &msg,
				       errstr, sizeof(errstr));
	if (r == 0)
		return 0; /* EAGAIN */
	else if (r == -1) {
		err_code = RD_KAFKA_RESP_ERR__TRANSPORT;
		rd_atomic64_add(&rkb->rkb_c.rx_err, 1);
		goto err;
	}

	rkbuf->rkbuf_wof += r;

	if (rkbuf->rkbuf_len == 0) {
		/* Packet length not known yet. */

		if (unlikely(rkbuf->rkbuf_wof < RD_KAFKAP_RESHDR_SIZE)) {
			/* Need response header for packet length and corrid.
			 * Wait for more data. */ 
			return 0;
		}

		/* Read protocol header */
		rd_kafka_buf_read_i32(rkbuf, &rkbuf->rkbuf_reshdr.Size);
		rd_kafka_buf_read_i32(rkbuf, &rkbuf->rkbuf_reshdr.CorrId);
		rkbuf->rkbuf_len = rkbuf->rkbuf_reshdr.Size;

		/* Make sure message size is within tolerable limits. */
		if (rkbuf->rkbuf_len < 4/*CorrId*/ ||
		    rkbuf->rkbuf_len >
		    (size_t)rkb->rkb_rk->rk_conf.recv_max_msg_size) {
			rd_snprintf(errstr, sizeof(errstr),
				 "Invalid message size %"PRIusz" (0..%i): "
				 "increase receive.message.max.bytes",
				 rkbuf->rkbuf_len-4,
				 rkb->rkb_rk->rk_conf.recv_max_msg_size);
			rd_atomic64_add(&rkb->rkb_c.rx_err, 1);
			err_code = RD_KAFKA_RESP_ERR__BAD_MSG;

			goto err;
		}

		rkbuf->rkbuf_len -= 4; /*CorrId*/

		if (rkbuf->rkbuf_len > 0) {
			/* Allocate another buffer that fits all data (short of
			 * the common response header). We want all
			 * data to be in contigious memory. */

			rd_kafka_buf_alloc_recvbuf(rkbuf, rkbuf->rkbuf_len);
		}
	}

	if (rkbuf->rkbuf_wof == rkbuf->rkbuf_len) {
		/* Message is complete, pass it on to the original requester. */
		rkb->rkb_recv_buf = NULL;
		(void)rd_atomic64_add(&rkb->rkb_c.rx, 1);
		(void)rd_atomic64_add(&rkb->rkb_c.rx_bytes, rkbuf->rkbuf_wof);
                rkbuf->rkbuf_rkb = rkb;
		rd_kafka_broker_keep(rkb);
		rd_kafka_req_response(rkb, rkbuf);
	}

	return 1;

err:
	rd_kafka_broker_fail(rkb,
                             !rkb->rkb_rk->rk_conf.log_connection_close &&
                             !strcmp(errstr, "Disconnected") ?
                             LOG_DEBUG : LOG_ERR, err_code,
                             "Receive failed: %s", errstr);
	return -1;
}


/**
 * Linux version of socket_cb providing racefree CLOEXEC.
 */
int rd_kafka_socket_cb_linux (int domain, int type, int protocol,
                              void *opaque) {
#ifdef SOCK_CLOEXEC
        return socket(domain, type | SOCK_CLOEXEC, protocol);
#else
        return rd_kafka_socket_cb_generic(domain, type, protocol, opaque);
#endif
}

/**
 * Fallback version of socket_cb NOT providing racefree CLOEXEC,
 * but setting CLOEXEC after socket creation (if FD_CLOEXEC is defined).
 */
int rd_kafka_socket_cb_generic (int domain, int type, int protocol,
                                void *opaque) {
        int s;
        int on = 1;
        s = (int)socket(domain, type, protocol);
        if (s == -1)
                return -1;
#ifdef FD_CLOEXEC
        fcntl(s, F_SETFD, FD_CLOEXEC, &on);
#endif
        return s;
}


/**
 * Initiate asynchronous connection attempt to the next address
 * in the broker's address list.
 * While the connect is asynchronous and its IO served in the CONNECT state,
 * the initial name resolve is blocking.
 *
 * Returns -1 on error, else 0.
 */
static int rd_kafka_broker_connect (rd_kafka_broker_t *rkb) {
	const rd_sockaddr_inx_t *sinx;
	char errstr[512];

	rd_rkb_dbg(rkb, BROKER, "CONNECT",
		"broker in state %s connecting",
		rd_kafka_broker_state_names[rkb->rkb_state]);

	if (rd_kafka_broker_resolve(rkb) == -1)
		return -1;

	sinx = rd_sockaddr_list_next(rkb->rkb_rsal);

	rd_kafka_assert(rkb->rkb_rk, !rkb->rkb_transport);

	if (!(rkb->rkb_transport = rd_kafka_transport_connect(rkb, sinx,
		errstr, sizeof(errstr)))) {
		/* Avoid duplicate log messages */
		if (rkb->rkb_err.err == errno)
			rd_kafka_broker_fail(rkb, LOG_DEBUG,
                                             RD_KAFKA_RESP_ERR__FAIL, NULL);
		else
			rd_kafka_broker_fail(rkb, LOG_ERR,
                                             RD_KAFKA_RESP_ERR__TRANSPORT,
					     "%s", errstr);
		return -1;
	}

	rd_kafka_broker_lock(rkb);
	rd_kafka_broker_set_state(rkb, RD_KAFKA_BROKER_STATE_CONNECT);
	rd_kafka_broker_unlock(rkb);

	return 0;
}


/**
 * @brief Call when connection is ready to transition to fully functional
 *        UP state.
 *
 * @locality Broker thread
 */
void rd_kafka_broker_connect_up (rd_kafka_broker_t *rkb) {

	rkb->rkb_max_inflight = rkb->rkb_rk->rk_conf.max_inflight;
        rkb->rkb_err.err = 0;

	rd_kafka_broker_lock(rkb);
	rd_kafka_broker_set_state(rkb, RD_KAFKA_BROKER_STATE_UP);
	rd_kafka_broker_unlock(rkb);

        /* Request metadata (async):
         * try locally known topics first and if there are none try
         * getting just the broker list. */
        if (rd_kafka_metadata_refresh_known_topics(NULL, rkb, 0/*dont force*/,
                                                   "connected") ==
            RD_KAFKA_RESP_ERR__UNKNOWN_TOPIC)
                rd_kafka_metadata_refresh_brokers(NULL, rkb, "connected");
}



static void rd_kafka_broker_connect_auth (rd_kafka_broker_t *rkb);


#if WITH_SASL
/**
 * @brief Parses and handles SaslMechanism response, transitions
 *        the broker state.
 *
 */
static void
rd_kafka_broker_handle_SaslHandshake (rd_kafka_t *rk,
				      rd_kafka_broker_t *rkb,
				      rd_kafka_resp_err_t err,
				      rd_kafka_buf_t *rkbuf,
				      rd_kafka_buf_t *request,
				      void *opaque) {
        const int log_decode_errors = 1;
	int32_t MechCnt;
	int16_t ErrorCode;
	int i = 0;
	char *mechs = "(n/a)";
	size_t msz, mof = 0;

	if (err == RD_KAFKA_RESP_ERR__DESTROY)
		return;

        if (err)
                goto err;

	rd_kafka_buf_read_i16(rkbuf, &ErrorCode);
        rd_kafka_buf_read_i32(rkbuf, &MechCnt);

	/* Build a CSV string of supported mechanisms. */
	msz = RD_MIN(511, MechCnt * 32);
	mechs = rd_alloca(msz);
	*mechs = '\0';

	for (i = 0 ; i < MechCnt ; i++) {
		rd_kafkap_str_t mech;
		rd_kafka_buf_read_str(rkbuf, &mech);

		mof += rd_snprintf(mechs+mof, msz-mof, "%s%.*s",
				   i ? ",":"", RD_KAFKAP_STR_PR(&mech));

		if (mof >= msz)
			break;
        }

	rd_rkb_dbg(rkb,
		   PROTOCOL | RD_KAFKA_DBG_SECURITY | RD_KAFKA_DBG_BROKER,
		   "SASLMECHS", "Broker supported SASL mechanisms: %s",
		   mechs);

	if (ErrorCode) {
		err = ErrorCode;
		goto err;
	}

	/* Circle back to connect_auth() to start proper AUTH state. */
	rd_kafka_broker_connect_auth(rkb);
	return;

 err:
	rd_kafka_broker_fail(rkb, LOG_ERR,
			     RD_KAFKA_RESP_ERR__AUTHENTICATION,
			     "SASL mechanism handshake failed: %s: "
			     "broker's supported mechanisms: %s",
			     rd_kafka_err2str(err), mechs);
}
#endif


/**
 * @brief Transition state to:
 *        - AUTH_HANDSHAKE (if SASL is configured and handshakes supported)
 *        - AUTH (if SASL is configured but no handshake is required or
 *                not supported, or has already taken place.)
 *        - UP (if SASL is not configured)
 */
static void rd_kafka_broker_connect_auth (rd_kafka_broker_t *rkb) {

#if WITH_SASL
	if ((rkb->rkb_proto == RD_KAFKA_PROTO_SASL_PLAINTEXT ||
	     rkb->rkb_proto == RD_KAFKA_PROTO_SASL_SSL)) {

		rd_rkb_dbg(rkb, SECURITY | RD_KAFKA_DBG_BROKER, "AUTH",
			   "Auth in state %s (handshake %ssupported)",
			   rd_kafka_broker_state_names[rkb->rkb_state],
			   (rkb->rkb_features&RD_KAFKA_FEATURE_SASL_HANDSHAKE)
			   ? "" : "not ");

		/* Broker >= 0.10.0: send request to select mechanism */
		if (rkb->rkb_state != RD_KAFKA_BROKER_STATE_AUTH_HANDSHAKE &&
		    (rkb->rkb_features & RD_KAFKA_FEATURE_SASL_HANDSHAKE)) {

			rd_kafka_broker_lock(rkb);
			rd_kafka_broker_set_state(
				rkb, RD_KAFKA_BROKER_STATE_AUTH_HANDSHAKE);
			rd_kafka_broker_unlock(rkb);

			rd_kafka_SaslHandshakeRequest(
				rkb, rkb->rkb_rk->rk_conf.sasl.mechanisms,
				RD_KAFKA_NO_REPLYQ,
				rd_kafka_broker_handle_SaslHandshake,
				NULL, 1 /* flash */);

		} else {
			/* Either Handshake succeeded (protocol selected)
			 * or Handshakes were not supported.
			 * In both cases continue with authentication. */
			char sasl_errstr[512];

			rd_kafka_broker_lock(rkb);
			rd_kafka_broker_set_state(rkb,
						  RD_KAFKA_BROKER_STATE_AUTH);
			rd_kafka_broker_unlock(rkb);

			if (rd_kafka_sasl_client_new(
				    rkb->rkb_transport, sasl_errstr,
				    sizeof(sasl_errstr)) == -1) {
				errno = EINVAL;
				rd_kafka_broker_fail(
					rkb, LOG_ERR,
					RD_KAFKA_RESP_ERR__AUTHENTICATION,
					"Failed to initialize "
					"SASL authentication: %s",
					sasl_errstr);
				return;
			}

			/* Enter non-Kafka-protocol-framed SASL communication
			 * state handled in rdkafka_sasl.c */
			rd_kafka_broker_lock(rkb);
			rd_kafka_broker_set_state(rkb,
						  RD_KAFKA_BROKER_STATE_AUTH);
			rd_kafka_broker_unlock(rkb);
		}

		return;
	}
#endif

	/* No authentication required. */
	rd_kafka_broker_connect_up(rkb);
}


/**
 * @brief Specify API versions to use for this connection.
 *
 * @param apis is an allocated list of supported partitions.
 *        If NULL the default set will be used based on the
 *        \p broker.version.fallback property.
 * @param api_cnt number of elements in \p apis
 *
 * @remark \p rkb takes ownership of \p apis.
 *
 * @locality Broker thread
 * @locks none
 */
static void rd_kafka_broker_set_api_versions (rd_kafka_broker_t *rkb,
					      struct rd_kafka_ApiVersion *apis,
					      size_t api_cnt) {

        rd_kafka_broker_lock(rkb);

	if (rkb->rkb_ApiVersions)
		rd_free(rkb->rkb_ApiVersions);


	if (!apis) {
		rd_rkb_dbg(rkb, PROTOCOL | RD_KAFKA_DBG_BROKER, "APIVERSION",
			   "Using (configuration fallback) %s protocol features",
			   rkb->rkb_rk->rk_conf.broker_version_fallback);


		rd_kafka_get_legacy_ApiVersions(rkb->rkb_rk->rk_conf.
						broker_version_fallback,
						&apis, &api_cnt,
						rkb->rkb_rk->rk_conf.
						broker_version_fallback);

		/* Make a copy to store on broker. */
		rd_kafka_ApiVersions_copy(apis, api_cnt, &apis, &api_cnt);
	}

	rkb->rkb_ApiVersions = apis;
	rkb->rkb_ApiVersions_cnt = api_cnt;

	/* Update feature set based on supported broker APIs. */
	rd_kafka_broker_features_set(rkb,
				     rd_kafka_features_check(rkb, apis, api_cnt));

        rd_kafka_broker_unlock(rkb);
}


/**
 * Handler for ApiVersion response.
 */
static void
rd_kafka_broker_handle_ApiVersion (rd_kafka_t *rk,
				   rd_kafka_broker_t *rkb,
				   rd_kafka_resp_err_t err,
				   rd_kafka_buf_t *rkbuf,
				   rd_kafka_buf_t *request, void *opaque) {
	struct rd_kafka_ApiVersion *apis;
	size_t api_cnt;

	if (err == RD_KAFKA_RESP_ERR__DESTROY)
		return;

	err = rd_kafka_handle_ApiVersion(rk, rkb, err, rkbuf, request,
					 &apis, &api_cnt);

	if (err) {
		rd_kafka_broker_fail(rkb, LOG_DEBUG,
				     RD_KAFKA_RESP_ERR__NOT_IMPLEMENTED,
				     "ApiVersionRequest failed: %s: "
				     "probably due to old broker version",
				     rd_kafka_err2str(err));
		return;
	}

	rd_kafka_broker_set_api_versions(rkb, apis, api_cnt);

	rd_kafka_broker_connect_auth(rkb);
}


/**
 * Call when asynchronous connection attempt completes, either succesfully
 * (if errstr is NULL) or fails.
 *
 * Locality: broker thread
 */
void rd_kafka_broker_connect_done (rd_kafka_broker_t *rkb, const char *errstr) {

	if (errstr) {
		/* Connect failed */
                rd_kafka_broker_fail(rkb,
                                     errno != 0 && rkb->rkb_err.err == errno ?
                                     LOG_DEBUG : LOG_ERR,
                                     RD_KAFKA_RESP_ERR__TRANSPORT,
                                     "%s", errstr);
		return;
	}

	/* Connect succeeded */
	rkb->rkb_connid++;
	rd_rkb_dbg(rkb, BROKER | RD_KAFKA_DBG_PROTOCOL,
		   "CONNECTED", "Connected (#%d)", rkb->rkb_connid);
	rkb->rkb_err.err = 0;
	rkb->rkb_max_inflight = 1; /* Hold back other requests until
				    * ApiVersion, SaslHandshake, etc
				    * are done. */

	rd_kafka_transport_poll_set(rkb->rkb_transport, POLLIN);

	if (rkb->rkb_rk->rk_conf.api_version_request &&
	    rd_interval_immediate(&rkb->rkb_ApiVersion_fail_intvl, 0, 0) > 0) {
		/* Use ApiVersion to query broker for supported API versions. */
		rd_kafka_broker_feature_enable(rkb, RD_KAFKA_FEATURE_APIVERSION);
	}


	if (rkb->rkb_features & RD_KAFKA_FEATURE_APIVERSION) {
		/* Query broker for supported API versions.
		 * This may fail with a disconnect on non-supporting brokers
		 * so hold off any other requests until we get a response,
		 * and if the connection is torn down we disable this feature. */
		rd_kafka_broker_lock(rkb);
		rd_kafka_broker_set_state(rkb,RD_KAFKA_BROKER_STATE_APIVERSION_QUERY);
		rd_kafka_broker_unlock(rkb);

		rd_kafka_ApiVersionRequest(
			rkb, RD_KAFKA_NO_REPLYQ,
			rd_kafka_broker_handle_ApiVersion, NULL,
			1 /*Flash message: prepend to transmit queue*/);
	} else {

		/* Use configured broker.version.fallback to
		 * figure out API versions */
		rd_kafka_broker_set_api_versions(rkb, NULL, 0);

		/* Authenticate if necessary */
		rd_kafka_broker_connect_auth(rkb);
	}

}



/**
 * @brief Checks if the given API request+version is supported by the broker.
 * @returns 1 if supported, else 0.
 * @locality broker thread
 * @locks none
 */
static RD_INLINE int
rd_kafka_broker_request_supported (rd_kafka_broker_t *rkb,
                                   rd_kafka_buf_t *rkbuf) {
        struct rd_kafka_ApiVersion skel = {
                .ApiKey = rkbuf->rkbuf_reqhdr.ApiKey
        };
        struct rd_kafka_ApiVersion *ret;

        if (unlikely(rkbuf->rkbuf_reqhdr.ApiKey == RD_KAFKAP_ApiVersion))
                return 1; /* ApiVersion requests are used to detect
                           * the supported API versions, so should always
                           * be allowed through. */

        /* First try feature flags, if any, which may cover a larger
         * set of APIs. */
        if (rkbuf->rkbuf_features)
                return (rkb->rkb_features & rkbuf->rkbuf_features) ==
                        rkbuf->rkbuf_features;

        /* Then try the ApiVersion map. */
        ret = bsearch(&skel, rkb->rkb_ApiVersions, rkb->rkb_ApiVersions_cnt,
                      sizeof(*rkb->rkb_ApiVersions),
                      rd_kafka_ApiVersion_key_cmp);
        if (!ret)
                return 0;

        return ret->MinVer <= rkbuf->rkbuf_reqhdr.ApiVersion &&
                rkbuf->rkbuf_reqhdr.ApiVersion <= ret->MaxVer;
}


/**
 * Send queued messages to broker
 *
 * Locality: io thread
 */
int rd_kafka_send (rd_kafka_broker_t *rkb) {
	rd_kafka_buf_t *rkbuf;
	unsigned int cnt = 0;

	rd_kafka_assert(rkb->rkb_rk, thrd_is_current(rkb->rkb_thread));

	while (rkb->rkb_state >= RD_KAFKA_BROKER_STATE_UP &&
	       rd_kafka_bufq_cnt(&rkb->rkb_waitresps) < rkb->rkb_max_inflight &&
	       (rkbuf = TAILQ_FIRST(&rkb->rkb_outbufs.rkbq_bufs))) {
		ssize_t r;
		struct msghdr *msg;
		struct msghdr msg2;
		struct iovec iov[IOV_MAX];
		size_t of = rkbuf->rkbuf_of;

                /* Check for broker support */
                if (unlikely(!rd_kafka_broker_request_supported(rkb, rkbuf))) {
                        rd_kafka_bufq_deq(&rkb->rkb_outbufs, rkbuf);
                        rd_rkb_dbg(rkb, BROKER | RD_KAFKA_DBG_PROTOCOL,
                                   "UNSUPPORTED",
                                   "Failing %sResponse "
                                   "(v%hd, %"PRIusz" bytes, CorrId %"PRId32"): "
                                   "request not supported by broker "
                                   "(incorrect broker.version.fallback?)",
                                   rd_kafka_ApiKey2str(rkbuf->rkbuf_reqhdr.
                                                       ApiKey),
                                   rkbuf->rkbuf_reqhdr.ApiVersion,
                                   rkbuf->rkbuf_len,
                                   rkbuf->rkbuf_reshdr.CorrId);
                        rd_kafka_buf_callback(
                                rkb->rkb_rk, rkb,
                                RD_KAFKA_RESP_ERR__UNSUPPORTED_FEATURE,
                                NULL, rkbuf);
                        continue;
                }

		/* Set CorrId header field, unless this is the latter part
		 * of a partial send in which case the corrid has already
		 * been set.
		 * Due to how SSL_write() will accept a buffer but still
		 * return 0 in some cases we can't rely on the buffer offset
		 * but need to use corrid to check this. SSL_write() expects
		 * us to send the same buffer again when 0 is returned.
		 */
		if (rkbuf->rkbuf_corrid == 0 ||
		    rkbuf->rkbuf_connid != rkb->rkb_connid) {
			rd_kafka_assert(NULL, rkbuf->rkbuf_of == 0);
			rkbuf->rkbuf_corrid = ++rkb->rkb_corrid;
			rd_kafka_buf_update_i32(rkbuf, 4+2+2,
						rkbuf->rkbuf_corrid);
			rkbuf->rkbuf_connid = rkb->rkb_connid;
		} else if (rkbuf->rkbuf_of > 0) {
			rd_kafka_assert(NULL,
					rkbuf->rkbuf_connid == rkb->rkb_connid);
                }

		if (rkbuf->rkbuf_of > 0 ||
		    rkbuf->rkbuf_iovcnt > IOV_MAX) {

			/* If message has been partially sent or contains
			 * too many iovecs for sendmsg() we need to construct
			 * a new msg+iovec skipping the sent bytes or
			 * excessive iovecs. */

			msg2.msg_iov = iov;
			rd_kafka_msghdr_rebuild(&msg2, IOV_MAX,
						rkb->rkb_rk->rk_conf.
						max_msg_size,
						&rkbuf->rkbuf_msg,
						rkbuf->rkbuf_of);
			msg = &msg2;
		} else
			msg = &rkbuf->rkbuf_msg;


		if (0) {
			rd_rkb_dbg(rkb, PROTOCOL, "SEND",
				   "Send %s corrid %"PRId32" at "
				   "offset %"PRIusz"/%"PRIusz,
				   rd_kafka_ApiKey2str(rkbuf->rkbuf_reqhdr.
						       ApiKey),
				   rkbuf->rkbuf_corrid,
				   rkbuf->rkbuf_of, rkbuf->rkbuf_len);
			msghdr_print(rkb->rkb_rk, "SEND", msg, 1);
		}

		if ((r = rd_kafka_broker_send(rkb, msg)) == -1) {
			/* FIXME: */
			return -1;
		}

		rkbuf->rkbuf_of += r;

		/* Partial send? Continue next time. */
		if (rkbuf->rkbuf_of < rkbuf->rkbuf_len) {
			rd_rkb_dbg(rkb, PROTOCOL, "SEND",
				   "Sent partial %sRequest "
				   "(v%hd, "
				   "%"PRIusz"+%"PRIdsz"/%"PRIusz" bytes, "
				   "CorrId %"PRId32")",
				   rd_kafka_ApiKey2str(rkbuf->rkbuf_reqhdr.
						       ApiKey),
				   rkbuf->rkbuf_reqhdr.ApiVersion,
				   rkbuf->rkbuf_of-r, r, rkbuf->rkbuf_len,
				   rkbuf->rkbuf_corrid);
			return 0;
		}

		rd_rkb_dbg(rkb, PROTOCOL, "SEND",
			   "Sent %sRequest (v%hd, %"PRIusz" bytes @ %"PRIusz", "
			   "CorrId %"PRId32")",
			   rd_kafka_ApiKey2str(rkbuf->rkbuf_reqhdr.ApiKey),
                           rkbuf->rkbuf_reqhdr.ApiVersion,
			   rkbuf->rkbuf_len, of, rkbuf->rkbuf_corrid);

		/* Entire buffer sent, unlink from outbuf */
		rd_kafka_bufq_deq(&rkb->rkb_outbufs, rkbuf);

		/* Store time for RTT calculation */
		rkbuf->rkbuf_ts_sent = rd_clock();

                if (rkbuf->rkbuf_flags & RD_KAFKA_OP_F_BLOCKING &&
		    rd_atomic32_add(&rkb->rkb_blocking_request_cnt, 1) == 1)
			rd_kafka_brokers_broadcast_state_change(rkb->rkb_rk);

		/* Put buffer on response wait list unless we are not
		 * expecting a response (required_acks=0). */
		if (!(rkbuf->rkbuf_flags & RD_KAFKA_OP_F_NO_RESPONSE))
			rd_kafka_bufq_enq(&rkb->rkb_waitresps, rkbuf);
		else { /* Call buffer callback for delivery report. */
                        rd_kafka_buf_callback(rkb->rkb_rk, rkb, 0, NULL, rkbuf);
                }

		cnt++;
	}

	return cnt;
}


/**
 * Add 'rkbuf' to broker 'rkb's retry queue.
 */
void rd_kafka_broker_buf_retry (rd_kafka_broker_t *rkb, rd_kafka_buf_t *rkbuf) {

        /* Restore original replyq since replyq.q will have been NULLed
         * by buf_callback()/replyq_enq(). */
        if (!rkbuf->rkbuf_replyq.q && rkbuf->rkbuf_orig_replyq.q) {
                rkbuf->rkbuf_replyq = rkbuf->rkbuf_orig_replyq;
                rd_kafka_replyq_clear(&rkbuf->rkbuf_orig_replyq);
        }

        /* If called from another thread than rkb's broker thread
         * enqueue the buffer on the broker's op queue. */
        if (!thrd_is_current(rkb->rkb_thread)) {
                rd_kafka_op_t *rko = rd_kafka_op_new(RD_KAFKA_OP_XMIT_RETRY);
                rko->rko_u.xbuf.rkbuf = rkbuf;
                rd_kafka_q_enq(rkb->rkb_ops, rko);
                return;
        }

        rd_rkb_dbg(rkb, PROTOCOL, "RETRY",
                   "Retrying %sRequest (v%hd, %"PRIusz" bytes, retry %d/%d)",
                   rd_kafka_ApiKey2str(rkbuf->rkbuf_reqhdr.ApiKey),
                   rkbuf->rkbuf_reqhdr.ApiVersion, rkbuf->rkbuf_len,
                   rkbuf->rkbuf_retries, rkb->rkb_rk->rk_conf.max_retries);

	rd_atomic64_add(&rkb->rkb_c.tx_retries, 1);

	rkbuf->rkbuf_ts_retry = rd_clock() +
		(rkb->rkb_rk->rk_conf.retry_backoff_ms * 1000);
	/* Reset send offset */
	rkbuf->rkbuf_of = 0;
	rkbuf->rkbuf_corrid = 0;

	rd_kafka_bufq_enq(&rkb->rkb_retrybufs, rkbuf);
}


/**
 * Move buffers that have expired their retry backoff time from the 
 * retry queue to the outbuf.
 */
static void rd_kafka_broker_retry_bufs_move (rd_kafka_broker_t *rkb) {
	rd_ts_t now = rd_clock();
	rd_kafka_buf_t *rkbuf;

	while ((rkbuf = TAILQ_FIRST(&rkb->rkb_retrybufs.rkbq_bufs))) {
		if (rkbuf->rkbuf_ts_retry > now)
			break;

		rd_kafka_bufq_deq(&rkb->rkb_retrybufs, rkbuf);

		rd_kafka_broker_buf_enq0(rkb, rkbuf, 0/*tail*/);
	}
}


/**
 * Propagate delivery report for entire message queue.
 */
void rd_kafka_dr_msgq (rd_kafka_itopic_t *rkt,
		       rd_kafka_msgq_t *rkmq, rd_kafka_resp_err_t err) {
        rd_kafka_t *rk = rkt->rkt_rk;

	if (unlikely(rd_kafka_msgq_len(rkmq) == 0))
	    return;

        if ((rk->rk_conf.enabled_events & RD_KAFKA_EVENT_DR) &&
	    (!rk->rk_conf.dr_err_only || err)) {
		/* Pass all messages to application thread in one op. */
		rd_kafka_op_t *rko;

		rko = rd_kafka_op_new(RD_KAFKA_OP_DR);
		rko->rko_err = err;
		rko->rko_u.dr.s_rkt = rd_kafka_topic_keep(rkt);
		rd_kafka_msgq_init(&rko->rko_u.dr.msgq);

		/* Move all messages to op's msgq */
		rd_kafka_msgq_move(&rko->rko_u.dr.msgq, rkmq);

		rd_kafka_q_enq(rk->rk_rep, rko);

	} else {
		/* No delivery report callback, destroy the messages
		 * right away. */
		rd_kafka_msgq_purge(rk, rkmq);
	}
}



/**
 * Parses a Produce reply.
 * Returns 0 on success or an error code on failure.
 */
static rd_kafka_resp_err_t
rd_kafka_produce_reply_handle (rd_kafka_broker_t *rkb,
			       rd_kafka_buf_t *rkbuf,
			       rd_kafka_buf_t *request,
                               int64_t *offsetp) {
	int32_t TopicArrayCnt;
	int32_t PartitionArrayCnt;
	struct {
		int32_t Partition;
		int16_t ErrorCode;
		int64_t Offset;
	} hdr;
        const int log_decode_errors = 1;

	rd_kafka_buf_read_i32(rkbuf, &TopicArrayCnt);
	if (TopicArrayCnt != 1)
		goto err;

	/* Since we only produce to one single topic+partition in each
	 * request we assume that the reply only contains one topic+partition
	 * and that it is the same that we requested.
	 * If not the broker is buggy. */
	rd_kafka_buf_skip_str(rkbuf);
	rd_kafka_buf_read_i32(rkbuf, &PartitionArrayCnt);

	if (PartitionArrayCnt != 1)
		goto err;

	rd_kafka_buf_read_i32(rkbuf, &hdr.Partition);
	rd_kafka_buf_read_i16(rkbuf, &hdr.ErrorCode);
	rd_kafka_buf_read_i64(rkbuf, &hdr.Offset);

        *offsetp = hdr.Offset;

	if (request->rkbuf_reqhdr.ApiVersion == 1) {
		int32_t Throttle_Time;
		rd_kafka_buf_read_i32(rkbuf, &Throttle_Time);

		rd_kafka_op_throttle_time(rkb, rkb->rkb_rk->rk_rep,
					  Throttle_Time);
	}


	return hdr.ErrorCode;

err:
	return RD_KAFKA_RESP_ERR__BAD_MSG;
}


/**
 * Locality: io thread
 */
static void rd_kafka_produce_msgset_reply (rd_kafka_t *rk,
					   rd_kafka_broker_t *rkb,
					   rd_kafka_resp_err_t err,
					   rd_kafka_buf_t *reply,
					   rd_kafka_buf_t *request,
					   void *opaque) {
	shptr_rd_kafka_toppar_t *s_rktp = opaque;
        rd_kafka_toppar_t *rktp = rd_kafka_toppar_s2i(s_rktp);
        int64_t offset = RD_KAFKA_OFFSET_INVALID;

	/* Parse Produce reply (unless the request errored) */
	if (!err && reply)
		err = rd_kafka_produce_reply_handle(rkb, reply,
						    request, &offset);

	rd_rkb_dbg(rkb, MSG, "MSGSET",
		   "MessageSet with %i message(s) %sdelivered",
		   rd_atomic32_get(&request->rkbuf_msgq.rkmq_msg_cnt),
		   err ? "not ": "");



	if (err) {
		int actions;

		actions = rd_kafka_err_action(
			rkb, err, reply, request,

			RD_KAFKA_ERR_ACTION_REFRESH|RD_KAFKA_ERR_ACTION_RETRY,
			RD_KAFKA_RESP_ERR__TRANSPORT,

			RD_KAFKA_ERR_ACTION_REFRESH,
			RD_KAFKA_RESP_ERR_UNKNOWN_TOPIC_OR_PART,

			RD_KAFKA_ERR_ACTION_END);

		rd_rkb_dbg(rkb, MSG, "MSGSET", "MessageSet with %i message(s) "
			   "encountered error: %s (actions 0x%x)",
			   rd_atomic32_get(&request->rkbuf_msgq.rkmq_msg_cnt),
			   rd_kafka_err2str(err), actions);

                if (actions & RD_KAFKA_ERR_ACTION_REFRESH) {
                        /* Request metadata information update */
                        rd_kafka_toppar_leader_unavailable(rktp,
                                                           "produce", err);

			/* Move messages (in the rkbuf) back to the partition's
			 * queue head. They will be resent when a new leader
			 * is delegated. */
			rd_kafka_toppar_insert_msgq(rktp, &request->rkbuf_msgq);
		}

		if ((actions & RD_KAFKA_ERR_ACTION_RETRY) &&
		    rd_kafka_buf_retry(rkb, request))
                        return; /* Scheduled for retry */

                /* Refresh implies a later retry through other means */
                if (actions & RD_KAFKA_ERR_ACTION_REFRESH)
                        goto done;


		/* Fatal errors: no message transmission retries */
		/* FALLTHRU */
	}

        /* Propagate assigned offset back to app. */
        if (likely(offset != RD_KAFKA_OFFSET_INVALID)) {
                rd_kafka_msg_t *rkm;
                if (rktp->rktp_rkt->rkt_conf.produce_offset_report) {
                        /* produce.offset.report: each message */
                        TAILQ_FOREACH(rkm, &request->rkbuf_msgq.rkmq_msgs,
                                      rkm_link)
                                rkm->rkm_offset = offset++;
                } else {
                        /* Last message in each batch */
                        rkm = TAILQ_LAST(&request->rkbuf_msgq.rkmq_msgs,
                                         rd_kafka_msg_head_s);
                        rkm->rkm_offset = offset +
                                rd_atomic32_get(&request->rkbuf_msgq.
                                                rkmq_msg_cnt);
                }
        }

	/* Enqueue messages for delivery report */
        rd_kafka_dr_msgq(rktp->rktp_rkt, &request->rkbuf_msgq, err);

done:
	rd_kafka_toppar_destroy(s_rktp); /* from produce_toppar() */
}




/**
 * Fix-up bad LZ4 framing caused by buggy Kafka client / broker.
 * The LZ4F framing format is described in detail here:
 * https://github.com/Cyan4973/lz4/blob/master/lz4_Frame_format.md
 *
 * NOTE: This modifies 'inbuf'.
 *
 * Returns an error on failure to fix (nothing modified), else NO_ERROR.
 */
static rd_kafka_resp_err_t
rd_kafka_lz4_decompress_fixup_bad_framing (rd_kafka_broker_t *rkb,
					   char *inbuf, size_t inlen) {
        static const char magic[4] = { 0x04, 0x22, 0x4d, 0x18 };
        uint8_t FLG, HC, correct_HC;
        size_t of = 4;

        /* Format is:
         *    int32_t magic;
         *    int8_t_ FLG;
         *    int8_t  BD;
         *  [ int64_t contentSize; ]
         *    int8_t  HC;
         */
        if (inlen < 4+3 || memcmp(inbuf, magic, 4)) {
                rd_rkb_dbg(rkb, BROKER,  "LZ4FIXUP",
                           "Unable to fix-up legacy LZ4 framing "
			   "(%"PRIusz" bytes): invalid length or magic value",
                           inlen);
                return RD_KAFKA_RESP_ERR__BAD_COMPRESSION;
        }

        of = 4; /* past magic */
        FLG = inbuf[of++];
        of++; /* BD */

        if ((FLG >> 3) & 1) /* contentSize */
                of += 8;

        if (of >= inlen) {
                rd_rkb_dbg(rkb, BROKER,  "LZ4FIXUP",
                           "Unable to fix-up legacy LZ4 framing (%"PRIusz" bytes): "
                           "requires %"PRIusz" bytes",
                           inlen, of);
                return RD_KAFKA_RESP_ERR__BAD_COMPRESSION;
        }

        /* Header hash code */
        HC = inbuf[of];

        /* Calculate correct header hash code */
        correct_HC = (XXH32(inbuf+4, of-4, 0) >> 8) & 0xff;

        if (HC != correct_HC)
                inbuf[of] = correct_HC;

        return RD_KAFKA_RESP_ERR_NO_ERROR;
}


/**
 * Reverse of fix-up: break LZ4 framing caused to be compatbile with with
 * buggy Kafka client / broker.
 *
 * NOTE: This modifies 'outbuf'.
 *
 * Returns an error on failure to recognize format (nothing modified),
 * else NO_ERROR.
 */
static rd_kafka_resp_err_t
rd_kafka_lz4_compress_break_framing (rd_kafka_broker_t *rkb,
				     char *outbuf, size_t outlen) {
        static const char magic[4] = { 0x04, 0x22, 0x4d, 0x18 };
        uint8_t FLG, HC, bad_HC;
        size_t of = 4;

        /* Format is:
         *    int32_t magic;
         *    int8_t_ FLG;
         *    int8_t  BD;
         *  [ int64_t contentSize; ]
         *    int8_t  HC;
         */
        if (outlen < 4+3 || memcmp(outbuf, magic, 4)) {
                rd_rkb_dbg(rkb, BROKER,  "LZ4FIXDOWN",
                           "Unable to break legacy LZ4 framing "
			   "(%"PRIusz" bytes): invalid length or magic value",
                           outlen);
                return RD_KAFKA_RESP_ERR__BAD_COMPRESSION;
        }

        of = 4; /* past magic */
        FLG = outbuf[of++];
        of++; /* BD */

        if ((FLG >> 3) & 1) /* contentSize */
                of += 8;

        if (of >= outlen) {
                rd_rkb_dbg(rkb, BROKER,  "LZ4FIXUP",
                           "Unable to break legacy LZ4 framing "
			   "(%"PRIusz" bytes): requires %"PRIusz" bytes",
                           outlen, of);
                return RD_KAFKA_RESP_ERR__BAD_COMPRESSION;
        }

        /* Header hash code */
        HC = outbuf[of];

        /* Calculate bad header hash code (include magic) */
        bad_HC = (XXH32(outbuf, of, 0) >> 8) & 0xff;

        if (HC != bad_HC)
                outbuf[of] = bad_HC;

        return RD_KAFKA_RESP_ERR_NO_ERROR;
}



/**
 * Decompress LZ4F (framed) data.
 * Kafka broker versions <0.10.0.0 breaks LZ4 framing checksum, if
 * \p proper_hc we assume the checksum is okay (broker version >=0.10.0) else
 * we fix it up.
 */
static rd_kafka_resp_err_t
rd_kafka_lz4_decompress (rd_kafka_broker_t *rkb, int proper_hc, int64_t Offset,
                         char *inbuf, size_t inlen,
			 void **outbuf, size_t *outlenp) {
        LZ4F_errorCode_t code;
        LZ4F_decompressionContext_t dctx;
        LZ4F_frameInfo_t fi;
        size_t in_sz, out_sz;
        size_t in_of, out_of;
        size_t r;
        size_t estimated_uncompressed_size;
        size_t outlen;
        rd_kafka_resp_err_t err = RD_KAFKA_RESP_ERR_NO_ERROR;
        char *out = NULL;

        *outbuf = NULL;

        code = LZ4F_createDecompressionContext(&dctx, LZ4F_VERSION);
        if (LZ4F_isError(code)) {
                rd_rkb_dbg(rkb, BROKER, "LZ4DECOMPR",
                           "Unable to create LZ4 decompression context: %s",
                           LZ4F_getErrorName(code));
                return RD_KAFKA_RESP_ERR__CRIT_SYS_RESOURCE;
        }

        if (!proper_hc) {
                /* The original/legacy LZ4 framing in Kafka was buggy and
                 * calculated the LZ4 framing header hash code (HC) incorrectly.
                 * We do a fix-up of it here. */
                if ((err = rd_kafka_lz4_decompress_fixup_bad_framing(rkb,
								     inbuf,
								     inlen)))
                        goto done;
        }

        in_sz = inlen;
        r = LZ4F_getFrameInfo(dctx, &fi, (const void *)inbuf, &in_sz);
        if (LZ4F_isError(r)) {
                rd_rkb_dbg(rkb, BROKER, "LZ4DECOMPR",
                           "Failed to gather LZ4 frame info: %s",
                           LZ4F_getErrorName(r));
                err = RD_KAFKA_RESP_ERR__BAD_COMPRESSION;
                goto done;
        }

        /* If uncompressed size is unknown or out of bounds make up a
	 * worst-case uncompressed size
         * More info on max size: http://stackoverflow.com/a/25751871/1821055 */
        if (fi.contentSize == 0 || fi.contentSize > inlen * 255)
                estimated_uncompressed_size = inlen * 255;
        else
                estimated_uncompressed_size = (size_t)fi.contentSize;

        /* Allocate output buffer, we increase this later if needed,
	 * but hopefully not. */
        out = rd_malloc(estimated_uncompressed_size);
        if (!out) {
                rd_rkb_log(rkb, LOG_WARNING, "LZ4DEC",
                           "Unable to allocate decompression "
			   "buffer of %zd bytes: %s",
                           estimated_uncompressed_size, rd_strerror(errno));
                err = RD_KAFKA_RESP_ERR__CRIT_SYS_RESOURCE;
                goto done;
        }


        /* Decompress input buffer to output buffer until input is exhausted. */
        outlen = estimated_uncompressed_size;
        in_of = in_sz;
        out_of = 0;
        while (in_of < inlen) {
                out_sz = outlen - out_of;
                in_sz = inlen - in_of;
                r = LZ4F_decompress(dctx, out+out_of, &out_sz,
				    inbuf+in_of, &in_sz, NULL);
                if (unlikely(LZ4F_isError(r))) {
                        rd_rkb_dbg(rkb, MSG, "LZ4DEC",
                                   "Failed to LZ4 (%s HC) decompress message "
				   "(offset %"PRId64") at "
                                   "payload offset %"PRIusz"/%"PRIusz": %s",
				   proper_hc ? "proper":"legacy",
                                   Offset, in_of, inlen,  LZ4F_getErrorName(r));
                        err = RD_KAFKA_RESP_ERR__BAD_COMPRESSION;
                        goto done;
                }

                rd_kafka_assert(NULL, out_of + out_sz < outlen &&
				in_of + in_sz <= inlen);
                out_of += out_sz;
                in_of += in_sz;
                if (r == 0)
                        break;

                /* Need to grow output buffer, this shouldn't happen if
		 * contentSize was properly set. */
                if (unlikely(r > 0 && out_of == outlen)) {
                        char *tmp;
                        size_t extra = (r > 1024 ? r : 1024) * 2;

                        rd_atomic64_add(&rkb->rkb_c.zbuf_grow, 1);

                        if ((tmp = rd_realloc(outbuf, outlen + extra))) {
                                rd_rkb_log(rkb, LOG_WARNING, "LZ4DEC",
                                           "Unable to grow decompression "
					   "buffer to %zd+%zd bytes: %s",
                                           outlen, extra,rd_strerror(errno));
                                err = RD_KAFKA_RESP_ERR__CRIT_SYS_RESOURCE;
                                goto done;
                        }
                        outlen += extra;
                }
        }


        if (in_of < inlen) {
                rd_rkb_dbg(rkb, MSG, "LZ4DEC",
                           "Failed to LZ4 (%s HC) decompress message "
			   "(offset %"PRId64"): "
			   "%"PRIusz" (out of %"PRIusz") bytes remaining",
			   proper_hc ? "proper":"legacy",
                           Offset, inlen-in_of, inlen);
                err = RD_KAFKA_RESP_ERR__BAD_MSG;
                goto done;
        }

        *outbuf = out;
        *outlenp = out_of;

done:
        code = LZ4F_freeDecompressionContext(dctx);
        if (LZ4F_isError(code)) {
                rd_rkb_dbg(rkb, BROKER, "LZ4DECOMPR",
                           "Failed to close LZ4 compression context: %s",
                           LZ4F_getErrorName(code));
                err = RD_KAFKA_RESP_ERR__BAD_COMPRESSION;
        }

        if (err && out)
                rd_free(out);

        return err;
}


/**
 * Allocate space for \p *outbuf and compress all \p iovlen buffers in \p iov.
 * @param proper_hc generate a proper HC (checksum) (kafka >=0.10.0.0)
 * @param MessageSetSize indicates full uncompressed data size.
 *
 * @returns allocated buffer in \p *outbuf, length in \p *outlenp.
 */
static rd_kafka_resp_err_t
rd_kafka_lz4_compress (rd_kafka_broker_t *rkb, int proper_hc,
                       const struct iovec *iov, int iov_len,
                       int32_t MessageSetSize,
                       void **outbuf, size_t *outlenp) {
        LZ4F_compressionContext_t cctx;
        LZ4F_errorCode_t r;
        rd_kafka_resp_err_t err = RD_KAFKA_RESP_ERR_NO_ERROR;
        size_t out_sz;
        size_t out_of = 0;
        char *out;
        int i;
	/* Required by Kafka */
        const LZ4F_preferences_t prefs =
                { .frameInfo = { .blockMode = LZ4F_blockIndependent } };

        *outbuf = NULL;

        out_sz = LZ4F_compressBound(MessageSetSize, NULL) + 1000;
        if (LZ4F_isError(out_sz)) {
                rd_rkb_dbg(rkb, MSG, "LZ4COMPR",
                           "Unable to query LZ4 compressed size "
                           "(for %"PRId32" uncompressed bytes): %s",
                           MessageSetSize, LZ4F_getErrorName(out_sz));
                return RD_KAFKA_RESP_ERR__BAD_MSG;
        }

        out = rd_malloc(out_sz);
        if (!out) {
                rd_rkb_dbg(rkb, MSG, "LZ4COMPR",
                           "Unable to allocate output buffer (%"PRIusz" bytes): "
			   "%s",
                           out_sz, rd_strerror(errno));
                return RD_KAFKA_RESP_ERR__CRIT_SYS_RESOURCE;
        }

        r = LZ4F_createCompressionContext(&cctx, LZ4F_VERSION);
        if (LZ4F_isError(r)) {
                rd_rkb_dbg(rkb, MSG, "LZ4COMPR",
                           "Unable to create LZ4 compression context: %s",
                           LZ4F_getErrorName(r));
                return RD_KAFKA_RESP_ERR__CRIT_SYS_RESOURCE;
        }

        r = LZ4F_compressBegin(cctx, out, out_sz, &prefs);
        if (LZ4F_isError(r)) {
                rd_rkb_dbg(rkb, MSG, "LZ4COMPR",
                           "Unable to begin LZ4 compression "
			   "(out buffer is %"PRIusz" bytes): %s",
                           out_sz, LZ4F_getErrorName(r));
                err = RD_KAFKA_RESP_ERR__BAD_COMPRESSION;
                goto done;
        }

        out_of += r;

        for (i = 0 ; i < iov_len ; i++) {
                rd_kafka_assert(NULL, out_of < out_sz);
                r = LZ4F_compressUpdate(cctx, out+out_of, out_sz-out_of,
                                        iov[i].iov_base, iov[i].iov_len, NULL);
                if (unlikely(LZ4F_isError(r))) {
                        rd_rkb_dbg(rkb, MSG, "LZ4COMPR",
                                   "LZ4 compression failed (at iov %d/%d "
                                   "of %"PRIusz" bytes, with "
                                   "%"PRIusz" bytes remaining in out buffer): "
				   "%s",
                                   i, iov_len,
                                   (size_t)iov[i].iov_len, out_sz - out_of,
                                   LZ4F_getErrorName(r));
                        err = RD_KAFKA_RESP_ERR__BAD_COMPRESSION;
                        goto done;
                }

                out_of += r;
        }

        r = LZ4F_compressEnd(cctx, out+out_of, out_sz-out_of, NULL);
        if (unlikely(LZ4F_isError(r))) {
                rd_rkb_dbg(rkb, MSG, "LZ4COMPR",
                           "Failed to finalize LZ4 compression "
                           "of %"PRId32" bytes: %s",
                           MessageSetSize, LZ4F_getErrorName(r));
                err = RD_KAFKA_RESP_ERR__BAD_COMPRESSION;
                goto done;
        }

        out_of += r;

        /* For the broken legacy framing we need to mess up the header checksum
	 * so that the Kafka client / broker code accepts it. */
        if (!proper_hc)
                if ((err = rd_kafka_lz4_compress_break_framing(rkb,
							       out, out_of)))
                        goto done;


        *outbuf  = out;
        *outlenp = out_of;

done:
        LZ4F_freeCompressionContext(cctx);

        if (err)
                rd_free(out);

        return err;

}


/**
 * Compresses a MessageSet
 */
static int rd_kafka_compress_MessageSet_buf (rd_kafka_broker_t *rkb,
					     rd_kafka_toppar_t *rktp,
					     rd_kafka_buf_t *rkbuf,
					     int iov_firstmsg,
					     size_t of_firstmsg,
					     size_t of_init_firstmsg,
					     int MsgVersion,
					     int64_t timestamp_firstmsg,
					     int32_t *MessageSetSizep) {
	int32_t MessageSetSize = *MessageSetSizep;
	size_t coutlen = 0;
	int    outlen;
	int r;
	rd_kafka_resp_err_t err;
#if WITH_SNAPPY
	int    siovlen = 1;
	struct snappy_env senv;
#endif
	struct iovec siov;
#if WITH_ZLIB
	z_stream strm;
	int i;
#endif

	switch (rktp->rktp_rkt->rkt_conf.compression_codec) {
	case RD_KAFKA_COMPRESSION_NONE:
		abort(); /* unreachable */
		break;

#if WITH_ZLIB
	case RD_KAFKA_COMPRESSION_GZIP:
		/* Initialize gzip compression */
		memset(&strm, 0, sizeof(strm));
		r = deflateInit2(&strm, Z_DEFAULT_COMPRESSION,
				 Z_DEFLATED, 15+16,
				 8, Z_DEFAULT_STRATEGY);
		if (r != Z_OK) {
			rd_rkb_log(rkb, LOG_ERR, "GZIP",
				   "Failed to initialize gzip for "
				   "compressing %"PRId32" bytes in "
				   "topic %.*s [%"PRId32"]: %s (%i): "
				   "sending uncompressed",
				   MessageSetSize,
				   RD_KAFKAP_STR_PR(rktp->rktp_rkt->rkt_topic),
				   rktp->rktp_partition,
				   strm.msg ? strm.msg : "", r);
			return -1;
		}

		/* Calculate maximum compressed size and
		 * allocate an output buffer accordingly, being
		 * prefixed with the Message header. */
		siov.iov_len = deflateBound(&strm, MessageSetSize);
		siov.iov_base = rd_malloc(siov.iov_len);

		strm.next_out = (void *)siov.iov_base;
		strm.avail_out = (uInt) siov.iov_len;

		/* Iterate through each message and compress it. */
		for (i = iov_firstmsg ;
		     i < (int)rkbuf->rkbuf_msg.msg_iovlen ; i++) {

			if (rkbuf->rkbuf_msg.msg_iov[i].iov_len == 0)
				continue;

			strm.next_in = (void *)rkbuf->rkbuf_msg.
				msg_iov[i].iov_base;
			strm.avail_in = (uInt) rkbuf->rkbuf_msg.msg_iov[i].iov_len;

			/* Compress message */
			if ((r = deflate(&strm, Z_NO_FLUSH) != Z_OK)) {
				rd_rkb_log(rkb, LOG_ERR, "GZIP",
					   "Failed to gzip-compress "
					   "%"PRIusz" bytes for "
					   "topic %.*s [%"PRId32"]: "
					   "%s (%i): "
					   "sending uncompressed",
					   (size_t)rkbuf->rkbuf_msg.msg_iov[i].
					   iov_len,
					   RD_KAFKAP_STR_PR(rktp->
							    rktp_rkt->
							    rkt_topic),
					   rktp->rktp_partition,
					   strm.msg ? strm.msg : "", r);
				deflateEnd(&strm);
				rd_free(siov.iov_base);
				return -1;
			}

			rd_kafka_assert(rkb->rkb_rk, strm.avail_in == 0);
		}

		/* Finish the compression */
		if ((r = deflate(&strm, Z_FINISH)) != Z_STREAM_END) {
			rd_rkb_log(rkb, LOG_ERR, "GZIP",
				   "Failed to finish gzip compression "
				   " of %"PRId32" bytes for "
				   "topic %.*s [%"PRId32"]: "
				   "%s (%i): "
				   "sending uncompressed",
				   MessageSetSize,
				   RD_KAFKAP_STR_PR(rktp->rktp_rkt->
						    rkt_topic),
				   rktp->rktp_partition,
				   strm.msg ? strm.msg : "", r);
			deflateEnd(&strm);
			rd_free(siov.iov_base);
			return -1;
		}

		coutlen = strm.total_out;

		/* Deinitialize compression */
		deflateEnd(&strm);
		break;
#endif

#if WITH_SNAPPY
	case RD_KAFKA_COMPRESSION_SNAPPY:
		/* Initialize snappy compression environment */
		rd_kafka_snappy_init_env_sg(&senv, 1/*iov enable*/);

		/* Calculate maximum compressed size and
		 * allocate an output buffer accordingly. */
		siov.iov_len = rd_kafka_snappy_max_compressed_length(MessageSetSize);
		siov.iov_base = rd_malloc(siov.iov_len);

		/* Compress each message */
		if ((r = rd_kafka_snappy_compress_iov(&senv,
					     &rkbuf->
					     rkbuf_iov[iov_firstmsg],
					     rkbuf->rkbuf_msg.
					     msg_iovlen -
					     iov_firstmsg,
					     MessageSetSize,
					     &siov, &siovlen,
					     &coutlen)) != 0) {
			rd_rkb_log(rkb, LOG_ERR, "SNAPPY",
				   "Failed to snappy-compress "
				   "%"PRId32" bytes for "
				   "topic %.*s [%"PRId32"]: %s: "
				   "sending uncompressed",
				   MessageSetSize,
				   RD_KAFKAP_STR_PR(rktp->rktp_rkt->
						    rkt_topic),
				   rktp->rktp_partition,
				   rd_strerror(-r));
			rd_free(siov.iov_base);
			return -1;
		}

		/* rd_free snappy environment */
		rd_kafka_snappy_free_env(&senv);
		break;
#endif

        case RD_KAFKA_COMPRESSION_LZ4:
		/* Skip LZ4 compression if broker doesn't support it. */
		if (!(rkb->rkb_features & RD_KAFKA_FEATURE_LZ4))
			return 0;

                err = rd_kafka_lz4_compress(rkb,
					    /* Correct or incorrect HC */
					    MsgVersion >= 1 ? 1 : 0,
                                            &rkbuf->rkbuf_iov[iov_firstmsg],
                                            (int)rkbuf->rkbuf_msg.msg_iovlen -
					    iov_firstmsg,
                                            MessageSetSize,
                                            &siov.iov_base, &coutlen);
                if (err)
                        return -1;
                break;


	default:
		rd_kafka_assert(rkb->rkb_rk,
				!*"notreached: compression.codec");
		break;

	}

	if (unlikely(RD_KAFKAP_MESSAGE_OVERHEAD + coutlen >
		     (size_t)MessageSetSize)) {
		/* If the compressed data is larger than the
		 * uncompressed throw it away and send as uncompressed. */
		rd_free(siov.iov_base);
		return 0;
	}

	/* Rewind rkbuf to the pre-message checkpoint.
	 * This is to replace all the original Messages with just the
	 * Message containing the compressed payload. */
	rd_kafka_buf_rewind(rkbuf, iov_firstmsg, of_firstmsg, of_init_firstmsg);

	rd_kafka_assert(rkb->rkb_rk, coutlen < INT32_MAX);
	rd_kafka_buf_write_Message(rkb, rkbuf, 0, MsgVersion,
				   rktp->rktp_rkt->rkt_conf.compression_codec,
				   timestamp_firstmsg,
				   NULL, 0,
				   (void *)siov.iov_base, (int32_t) coutlen,
				   &outlen);

	/* Update enveloping MessageSet's length. */
	*MessageSetSizep = outlen;

	/* Add allocated buffer as auxbuf to rkbuf so that
	 * it will get freed with the rkbuf */
	rd_kafka_buf_auxbuf_add(rkbuf, siov.iov_base);

	return 0;
}


/**
 * Produce messages from 'rktp's queue.
 */
static int rd_kafka_broker_produce_toppar (rd_kafka_broker_t *rkb,
					   rd_kafka_toppar_t *rktp) {
	int cnt;
	rd_kafka_msg_t *rkm;
	int msgcnt = 0, msgcntmax;
	rd_kafka_buf_t *rkbuf;
	rd_kafka_itopic_t *rkt = rktp->rktp_rkt;
	int iovcnt;
	size_t iov_firstmsg;
	size_t of_firstmsg;
	size_t of_init_firstmsg;
	size_t of_MessageSetSize;
	int32_t MessageSetSize = 0;
	int outlen;
	int MsgVersion = 0;
	int use_relative_offsets = 0;
	int64_t timestamp_firstmsg = 0;
	int queued_cnt;
	size_t queued_bytes;
	size_t buffer_space;
	rd_ts_t int_latency_base;

        queued_cnt = rd_kafka_msgq_len(&rktp->rktp_xmit_msgq);
        if (queued_cnt == 0)
                return 0;

        queued_bytes = rd_kafka_msgq_size(&rktp->rktp_xmit_msgq);

	/* Internal latency calculation base.
	 * Uses rkm_ts_timeout which is enqueue time + timeout */
	int_latency_base = rd_clock() + (rkt->rkt_conf.message_timeout_ms * 1000);

	if (rkb->rkb_features & RD_KAFKA_FEATURE_MSGVER1) {
		MsgVersion = 1;
		if (rktp->rktp_rkt->rkt_conf.compression_codec)
			use_relative_offsets = 1;
	}

	/* iovs:
	 *  1 * (RequiredAcks + Timeout + Topic + Partition + MessageSetSize)
	 *  msgcntmax * messagehdr
	 *  msgcntmax * Value_len
	 *  msgcntmax * messagepayload (ext memory)
	 * = 1 + (4 * msgcntmax)
	 *
	 * We are bound by configuration.
	 */
	msgcntmax = RD_MIN(queued_cnt, rkb->rkb_rk->rk_conf.batch_num_messages);
	rd_kafka_assert(rkb->rkb_rk, msgcntmax > 0);
	iovcnt = 1 + (3 * msgcntmax);

	/* Allocate iovecs to hold all headers and messages,
	 * and allocate enough to allow copies of small messages.
	 * The allocated size is the minimum of message.max.bytes
	 * or queued_bytes + queued_cnt * per_msg_overhead */

	buffer_space =
		/* RequiredAcks + Timeout + TopicCnt */
		2 + 4 + 4 +
		/* Topic */
		RD_KAFKAP_STR_SIZE(rkt->rkt_topic) +
		/* PartitionCnt + Partition + MessageSetSize */
		4 + 4 + 4;

	if (rkb->rkb_rk->rk_conf.msg_copy_max_size > 0)
		buffer_space += queued_bytes +
			msgcntmax * RD_KAFKAP_MESSAGE_OVERHEAD;
	else
		buffer_space += msgcntmax * RD_KAFKAP_MESSAGE_OVERHEAD;


	rkbuf = rd_kafka_buf_new(rkb->rkb_rk, RD_KAFKAP_Produce, iovcnt,
				 RD_MIN((size_t)rkb->rkb_rk->rk_conf.
					max_msg_size,
					buffer_space));

	/*
	 * Insert first part of Produce header
	 */
	/* RequiredAcks */
	rd_kafka_buf_write_i16(rkbuf, rkt->rkt_conf.required_acks);
	/* Timeout */
	rd_kafka_buf_write_i32(rkbuf, rkt->rkt_conf.request_timeout_ms);
	/* TopicArrayCnt */
	rd_kafka_buf_write_i32(rkbuf, 1);

	/* Insert topic */
	rd_kafka_buf_write_kstr(rkbuf, rkt->rkt_topic);

	/*
	 * Insert second part of Produce header
	 */
	/* PartitionArrayCnt */
	rd_kafka_buf_write_i32(rkbuf, 1);
	/* Partition */
	rd_kafka_buf_write_i32(rkbuf, rktp->rktp_partition);
	/* MessageSetSize: Will be finalized later*/
	of_MessageSetSize = rd_kafka_buf_write_i32(rkbuf, 0);

	/* Push write-buffer onto iovec stack to create a clean iovec boundary
	 * for the compression codecs. */
	rd_kafka_buf_autopush(rkbuf);

	iov_firstmsg = rkbuf->rkbuf_msg.msg_iovlen;
	of_firstmsg = rkbuf->rkbuf_wof;
	of_init_firstmsg = rkbuf->rkbuf_wof_init;

	while (msgcnt < msgcntmax &&
	       (rkm = TAILQ_FIRST(&rktp->rktp_xmit_msgq.rkmq_msgs))) {

		if (of_firstmsg + MessageSetSize + rd_kafka_msg_wire_size(rkm) >
		    (size_t)rkb->rkb_rk->rk_conf.max_msg_size) {
			rd_rkb_dbg(rkb, MSG, "PRODUCE",
				   "No more space in current MessageSet "
				   "(%i messages)",
				   rd_atomic32_get(&rkbuf->rkbuf_msgq.
						   rkmq_msg_cnt));
			/* Not enough remaining size. */
			break;
		}

		rd_kafka_msgq_deq(&rktp->rktp_xmit_msgq, rkm, 1);
		rd_kafka_msgq_enq(&rkbuf->rkbuf_msgq, rkm);

		rd_avg_add(&rkb->rkb_avg_int_latency,
				   int_latency_base - rkm->rkm_ts_timeout);

		if (unlikely(msgcnt == 0 && MsgVersion == 1))
			timestamp_firstmsg = rkm->rkm_timestamp;

		/* Write message to buffer */
		rd_kafka_assert(rkb->rkb_rk, rkm->rkm_len < INT32_MAX);
		rd_kafka_buf_write_Message(rkb, rkbuf,
					   use_relative_offsets ? msgcnt : 0,
					   MsgVersion,
					   RD_KAFKA_COMPRESSION_NONE,
					   rkm->rkm_timestamp,
					   rkm->rkm_key, (int32_t)rkm->rkm_key_len,
					   rkm->rkm_payload,
					   (int32_t)rkm->rkm_len,
					   &outlen);

		msgcnt++;
		MessageSetSize += outlen;
		rd_dassert(outlen <= rd_kafka_msg_wire_size(rkm));
	}

	/* No messages added, bail out early. */
	if (unlikely(rd_atomic32_get(&rkbuf->rkbuf_msgq.rkmq_msg_cnt) == 0)) {
		rd_kafka_buf_destroy(rkbuf);
		return -1;
	}

	/* Push final (copied) message, if any. */
	rd_kafka_buf_autopush(rkbuf);

	/* Compress the message(s) */
	if (rktp->rktp_rkt->rkt_conf.compression_codec)
		rd_kafka_compress_MessageSet_buf(rkb, rktp, rkbuf,
						 (int)iov_firstmsg, (int)of_firstmsg,
						 of_init_firstmsg,
						 MsgVersion,
						 timestamp_firstmsg,
						 &MessageSetSize);

	/* Update MessageSetSize */
	rd_kafka_buf_update_i32(rkbuf, of_MessageSetSize, MessageSetSize);
	
	rd_atomic64_add(&rktp->rktp_c.tx_msgs,
			rd_atomic32_get(&rkbuf->rkbuf_msgq.rkmq_msg_cnt));
	rd_atomic64_add(&rktp->rktp_c.tx_bytes, MessageSetSize);

	rd_rkb_dbg(rkb, MSG, "PRODUCE",
		   "produce messageset with %i messages "
		   "(%"PRId32" bytes)",
		   rd_atomic32_get(&rkbuf->rkbuf_msgq.rkmq_msg_cnt),
		   MessageSetSize);

	cnt = rd_atomic32_get(&rkbuf->rkbuf_msgq.rkmq_msg_cnt);

	if (!rkt->rkt_conf.required_acks)
		rkbuf->rkbuf_flags |= RD_KAFKA_OP_F_NO_RESPONSE;

	/* Use timeout from first message. */
	rkbuf->rkbuf_ts_timeout =
		TAILQ_FIRST(&rkbuf->rkbuf_msgq.rkmq_msgs)->rkm_ts_timeout;

	if (rkb->rkb_features & RD_KAFKA_FEATURE_THROTTLETIME)
                rd_kafka_buf_ApiVersion_set(rkbuf, 1,
                                            RD_KAFKA_FEATURE_THROTTLETIME);

        rd_kafka_broker_buf_enq_replyq(rkb, rkbuf,
                                       RD_KAFKA_NO_REPLYQ,
                                       rd_kafka_produce_msgset_reply,
                                       /* refcount for msgset_reply() */
                                       rd_kafka_toppar_keep(rktp));


	return cnt;
}


/**
 * @brief Map and assign existing partitions to this broker using
 *        the leader-id.
 *
 * @locks none
 * @locality any
 */
static void rd_kafka_broker_map_partitions (rd_kafka_broker_t *rkb) {
        rd_kafka_t *rk = rkb->rkb_rk;
        rd_kafka_itopic_t *rkt;
        int cnt = 0;

        if (rkb->rkb_nodeid == -1)
                return;

        rd_kafka_rdlock(rk);
        TAILQ_FOREACH(rkt, &rk->rk_topics, rkt_link) {
                int i;

                rd_kafka_topic_wrlock(rkt);
                for (i = 0 ; i < rkt->rkt_partition_cnt ; i++) {
                        shptr_rd_kafka_toppar_t *s_rktp = rkt->rkt_p[i];
                        rd_kafka_toppar_t *rktp = rd_kafka_toppar_s2i(s_rktp);

                        /* Only map unassigned partitions matching this broker*/
                        rd_kafka_toppar_lock(rktp);
                        if (rktp->rktp_leader_id == rkb->rkb_nodeid &&
                            !(rktp->rktp_leader && rktp->rktp_next_leader)) {
                                rd_kafka_toppar_leader_update(
                                        rktp, rktp->rktp_leader_id, rkb);
                                cnt++;
                        }
                        rd_kafka_toppar_unlock(rktp);
                }
                rd_kafka_topic_wrunlock(rkt);
        }
        rd_kafka_rdunlock(rk);

        rd_rkb_dbg(rkb, TOPIC|RD_KAFKA_DBG_BROKER, "LEADER",
                   "Mapped %d partition(s) to broker", cnt);
}


/**
 * Serve a broker op (an op posted by another thread to be handled by
 * this broker's thread).
 *
 * Locality: broker thread
 * Locks: none
 */
static void rd_kafka_broker_op_serve (rd_kafka_broker_t *rkb,
				      rd_kafka_op_t *rko) {
        shptr_rd_kafka_toppar_t *s_rktp;
        rd_kafka_toppar_t *rktp;

	rd_kafka_assert(rkb->rkb_rk, thrd_is_current(rkb->rkb_thread));

	switch (rko->rko_type)
	{
        case RD_KAFKA_OP_NODE_UPDATE:
        {
                enum {
                        _UPD_NAME = 0x1,
                        _UPD_ID = 0x2
                } updated = 0;
                char brokername[RD_KAFKA_NODENAME_SIZE];

                rd_kafka_broker_lock(rkb);

                if (strcmp(rkb->rkb_nodename,
                           rko->rko_u.node.nodename)) {
                        rd_rkb_dbg(rkb, BROKER, "UPDATE",
                                   "Nodename changed from %s to %s",
                                   rkb->rkb_nodename,
                                   rko->rko_u.node.nodename);
                        strncpy(rkb->rkb_nodename,
                                rko->rko_u.node.nodename,
                                sizeof(rkb->rkb_nodename)-1);
                        updated |= _UPD_NAME;
                }

                if (rko->rko_u.node.nodeid != -1 &&
                    rko->rko_u.node.nodeid != rkb->rkb_nodeid) {
                        rd_rkb_dbg(rkb, BROKER, "UPDATE",
                                   "NodeId changed from %"PRId32" to %"PRId32,
                                   rkb->rkb_nodeid,
				   rko->rko_u.node.nodeid);
                        rkb->rkb_nodeid = rko->rko_u.node.nodeid;
                        updated |= _UPD_ID;
                }

                rd_kafka_mk_brokername(brokername, sizeof(brokername),
                                       rkb->rkb_proto,
				       rkb->rkb_nodename, rkb->rkb_nodeid,
				       RD_KAFKA_LEARNED);
                if (strcmp(rkb->rkb_name, brokername)) {
                        /* Udate the name copy used for logging. */
                        mtx_lock(&rkb->rkb_logname_lock);
                        rd_free(rkb->rkb_logname);
                        rkb->rkb_logname = rd_strdup(brokername);
                        mtx_unlock(&rkb->rkb_logname_lock);

                        rd_rkb_dbg(rkb, BROKER, "UPDATE",
                                   "Name changed from %s to %s",
                                   rkb->rkb_name, brokername);
                        strncpy(rkb->rkb_name, brokername,
                                sizeof(rkb->rkb_name)-1);
                }
                rd_kafka_broker_unlock(rkb);

                if (updated & _UPD_NAME)
                        rd_kafka_broker_fail(rkb, LOG_NOTICE,
                                             RD_KAFKA_RESP_ERR__NODE_UPDATE,
                                             "Broker hostname updated");
                else if (updated & _UPD_ID) {
                        /* Map existing partitions to this broker. */
                        rd_kafka_broker_map_partitions(rkb);

			/* If broker is currently in state up we need
			 * to trigger a state change so it exits its
			 * state&type based .._serve() loop. */
                        rd_kafka_broker_lock(rkb);
			if (rkb->rkb_state == RD_KAFKA_BROKER_STATE_UP)
				rd_kafka_broker_set_state(
					rkb, RD_KAFKA_BROKER_STATE_UPDATE);
                        rd_kafka_broker_unlock(rkb);
                }
                break;
        }

        case RD_KAFKA_OP_XMIT_BUF:
                rd_kafka_broker_buf_enq2(rkb, rko->rko_u.xbuf.rkbuf);
                rko->rko_u.xbuf.rkbuf = NULL; /* buffer now owned by broker */
                if (rko->rko_replyq.q) {
                        /* Op will be reused for forwarding response. */
                        rko = NULL;
                }
                break;

        case RD_KAFKA_OP_XMIT_RETRY:
                rd_kafka_broker_buf_retry(rkb, rko->rko_u.xbuf.rkbuf);
                rko->rko_u.xbuf.rkbuf = NULL;
                break;

        case RD_KAFKA_OP_PARTITION_JOIN:
                /*
		 * Add partition to broker toppars
		 */
                rktp = rd_kafka_toppar_s2i(rko->rko_rktp);
                rd_kafka_toppar_lock(rktp);

                /* Abort join if instance is terminating */
                if (rd_kafka_terminating(rkb->rkb_rk) ||
		    (rktp->rktp_flags & RD_KAFKA_TOPPAR_F_REMOVE)) {
                        rd_rkb_dbg(rkb, BROKER | RD_KAFKA_DBG_TOPIC, "TOPBRK",
                                   "Topic %s [%"PRId32"]: not joining broker: "
                                   "%s",
                                   rktp->rktp_rkt->rkt_topic->str,
                                   rktp->rktp_partition,
				   rd_kafka_terminating(rkb->rkb_rk) ?
				   "instance is terminating" :
				   "partition removed");

                        rd_kafka_broker_destroy(rktp->rktp_next_leader);
                        rktp->rktp_next_leader = NULL;
                        rd_kafka_toppar_unlock(rktp);
                        break;
                }

                /* See if we are still the next leader */
                if (rktp->rktp_next_leader != rkb) {
                        rd_rkb_dbg(rkb, BROKER | RD_KAFKA_DBG_TOPIC, "TOPBRK",
                                   "Topic %s [%"PRId32"]: not joining broker "
                                   "(next leader %s)",
                                   rktp->rktp_rkt->rkt_topic->str,
                                   rktp->rktp_partition,
                                   rktp->rktp_next_leader ?
                                   rd_kafka_broker_name(rktp->rktp_next_leader):
                                   "(none)");

                        /* Need temporary refcount so we can safely unlock
                         * after q_enq(). */
                        s_rktp = rd_kafka_toppar_keep(rktp);

                        /* No, forward this op to the new next leader. */
                        rd_kafka_q_enq(rktp->rktp_next_leader->rkb_ops, rko);
                        rko = NULL;

                        rd_kafka_toppar_unlock(rktp);
                        rd_kafka_toppar_destroy(s_rktp);

                        break;
                }

                rd_rkb_dbg(rkb, BROKER | RD_KAFKA_DBG_TOPIC, "TOPBRK",
                           "Topic %s [%"PRId32"]: joining broker (rktp %p)",
                           rktp->rktp_rkt->rkt_topic->str,
                           rktp->rktp_partition, rktp);

                rd_kafka_assert(NULL, rktp->rktp_s_for_rkb == NULL);
		rktp->rktp_s_for_rkb = rd_kafka_toppar_keep(rktp);
                rd_kafka_broker_lock(rkb);
		TAILQ_INSERT_TAIL(&rkb->rkb_toppars, rktp, rktp_rkblink);
		rkb->rkb_toppar_cnt++;
                rd_kafka_broker_unlock(rkb);
		rktp->rktp_leader = rkb;
                rktp->rktp_msgq_wakeup_fd = rkb->rkb_toppar_wakeup_fd;
                rd_kafka_broker_keep(rkb);

                rd_kafka_broker_destroy(rktp->rktp_next_leader);
                rktp->rktp_next_leader = NULL;

                rd_kafka_toppar_unlock(rktp);

		rd_kafka_brokers_broadcast_state_change(rkb->rkb_rk);
                break;

        case RD_KAFKA_OP_PARTITION_LEAVE:
                /*
		 * Remove partition from broker toppars
		 */
                rktp = rd_kafka_toppar_s2i(rko->rko_rktp);

		rd_kafka_toppar_lock(rktp);

		/* Multiple PARTITION_LEAVEs are possible during partition
		 * migration, make sure we're supposed to handle this one. */
		if (unlikely(rktp->rktp_leader != rkb)) {
			rd_rkb_dbg(rkb, BROKER | RD_KAFKA_DBG_TOPIC, "TOPBRK",
				   "Topic %s [%"PRId32"]: "
				   "ignoring PARTITION_LEAVE: "
				   "broker is not leader (%s)",
				   rktp->rktp_rkt->rkt_topic->str,
				   rktp->rktp_partition,
				   rktp->rktp_leader ?
				   rd_kafka_broker_name(rktp->rktp_leader) :
				   "none");
			rd_kafka_toppar_unlock(rktp);
			break;
		}
		rd_kafka_toppar_unlock(rktp);

		/* Remove from fetcher list */
		rd_kafka_toppar_fetch_decide(rktp, rkb, 1/*force remove*/);

		rd_kafka_toppar_lock(rktp);

		rd_rkb_dbg(rkb, BROKER | RD_KAFKA_DBG_TOPIC, "TOPBRK",
			   "Topic %s [%"PRId32"]: leaving broker "
			   "(%d messages in xmitq, next leader %s, rktp %p)",
			   rktp->rktp_rkt->rkt_topic->str, rktp->rktp_partition,
			   rd_kafka_msgq_len(&rktp->rktp_xmit_msgq),
			   rktp->rktp_next_leader ?
			   rd_kafka_broker_name(rktp->rktp_next_leader) :
			   "(none)", rktp);

		/* Prepend xmitq(broker-local) messages to the msgq(global).
		 * There is no msgq_prepend() so we append msgq to xmitq
		 * and then move the queue altogether back over to msgq. */
		rd_kafka_msgq_concat(&rktp->rktp_xmit_msgq,
				     &rktp->rktp_msgq);
		rd_kafka_msgq_move(&rktp->rktp_msgq, &rktp->rktp_xmit_msgq);

                rd_kafka_broker_lock(rkb);
		TAILQ_REMOVE(&rkb->rkb_toppars, rktp, rktp_rkblink);
		rkb->rkb_toppar_cnt--;
                rd_kafka_broker_unlock(rkb);
                rd_kafka_broker_destroy(rktp->rktp_leader);
                rktp->rktp_msgq_wakeup_fd = -1;
		rktp->rktp_leader = NULL;

                /* Need to hold on to a refcount past q_enq() and
                 * unlock() below */
                s_rktp = rktp->rktp_s_for_rkb;
                rktp->rktp_s_for_rkb = NULL;

                if (rktp->rktp_next_leader) {
                        /* There is a next leader we need to migrate to. */
                        rko->rko_type = RD_KAFKA_OP_PARTITION_JOIN;
                        rd_kafka_q_enq(rktp->rktp_next_leader->rkb_ops, rko);
                        rko = NULL;
                } else {
			rd_rkb_dbg(rkb, BROKER | RD_KAFKA_DBG_TOPIC, "TOPBRK",
				   "Topic %s [%"PRId32"]: no next leader, "
				   "failing %d message(s) in partition queue",
				   rktp->rktp_rkt->rkt_topic->str,
				   rktp->rktp_partition,
				   rd_kafka_msgq_len(&rktp->rktp_msgq));
			rd_kafka_assert(NULL, rd_kafka_msgq_len(&rktp->rktp_xmit_msgq) == 0);
			rd_kafka_dr_msgq(rktp->rktp_rkt, &rktp->rktp_msgq,
					 rd_kafka_terminating(rkb->rkb_rk) ?
					 RD_KAFKA_RESP_ERR__DESTROY :
					 RD_KAFKA_RESP_ERR__UNKNOWN_PARTITION);

		}

                rd_kafka_toppar_unlock(rktp);
                rd_kafka_toppar_destroy(s_rktp);

		rd_kafka_brokers_broadcast_state_change(rkb->rkb_rk);
                break;

        case RD_KAFKA_OP_TERMINATE:
                /* nop: just a wake-up. */
                if (rkb->rkb_blocking_max_ms > 1)
                        rkb->rkb_blocking_max_ms = 1; /* Speed up termination*/
                break;

        default:
                rd_kafka_assert(rkb->rkb_rk, !*"unhandled op type");
                break;
        }

        if (rko)
                rd_kafka_op_destroy(rko);
}


/**
 * Serve broker ops and IOs.
 *
 * NOTE: timeout_ms decides on maximum blocking time for serving the ops queue,
 *       not IO poll timeout.
 *
 * Locality: broker thread
 * Locks: none
 */
static void rd_kafka_broker_serve (rd_kafka_broker_t *rkb, int timeout_ms) {
	rd_kafka_op_t *rko;
	rd_ts_t now;

	/* Serve broker ops */
        while ((rko = rd_kafka_q_pop(rkb->rkb_ops, timeout_ms, 0)))
                rd_kafka_broker_op_serve(rkb, rko);

	/* Serve IO events */
        if (likely(rkb->rkb_transport != NULL))
                rd_kafka_transport_io_serve(rkb->rkb_transport,
                                            rkb->rkb_blocking_max_ms);

        /* Scan wait-response queue for timeouts. */
        now = rd_clock();
        if (rd_interval(&rkb->rkb_timeout_scan_intvl, 1000000, now) > 0)
                rd_kafka_broker_timeout_scan(rkb, now);
}


/**
 * Serve the toppar's assigned to this broker.
 *
 * Locality: broker thread
 */
static void rd_kafka_broker_toppars_serve (rd_kafka_broker_t *rkb) {
        rd_kafka_toppar_t *rktp, *rktp_tmp;

        TAILQ_FOREACH_SAFE(rktp, &rkb->rkb_toppars, rktp_rkblink, rktp_tmp) {
		/* Serve toppar to update desired rktp state */
		rd_kafka_broker_consumer_toppar_serve(rkb, rktp);
        }
}


/**
 * Idle function for unassigned brokers
 * If \p timeout_ms is non-zero the serve loop will be exited regardless
 * of state after this long (approximately).
 */
static void rd_kafka_broker_ua_idle (rd_kafka_broker_t *rkb, int timeout_ms) {
	int initial_state = rkb->rkb_state;
        rd_ts_t ts_end = timeout_ms ? rd_clock() + timeout_ms * 1000 : 0;

	/* Since ua_idle is used during connection setup 
	 * in state ..BROKER_STATE_CONNECT we only run this loop
	 * as long as the state remains the same as the initial, on a state
	 * change - most likely to UP, a correct serve() function
	 * should be used instead. */
	while (!rd_kafka_broker_terminating(rkb) &&
	       (int)rkb->rkb_state == initial_state &&
               (!ts_end || ts_end > rd_clock())) {

                rd_kafka_broker_toppars_serve(rkb);

		rd_kafka_broker_serve(rkb, 10);
        }
}


/**
 * @brief Serve a toppar for producing.
 *
 * @param next_wakeup will be updated to when the next wake-up/attempt is
 *                    desired, only lower (sooner) values will be set.
 *
 * Locks: toppar_lock(rktp) MUST be held. 
 * Returns the number of messages produced.
 */
static int rd_kafka_toppar_producer_serve (rd_kafka_broker_t *rkb,
                                           rd_kafka_toppar_t *rktp,
                                           int do_timeout_scan,
                                           rd_ts_t now,
                                           rd_ts_t *next_wakeup) {
        int cnt = 0;
        int r;

        rd_rkb_dbg(rkb, QUEUE, "TOPPAR",
                   "%.*s [%"PRId32"] %i+%i msgs",
                   RD_KAFKAP_STR_PR(rktp->rktp_rkt->
                                    rkt_topic),
                   rktp->rktp_partition,
                   rd_atomic32_get(&rktp->rktp_msgq.rkmq_msg_cnt),
                   rd_atomic32_get(&rktp->rktp_xmit_msgq.
                                   rkmq_msg_cnt));

        if (rd_atomic32_get(&rktp->rktp_msgq.rkmq_msg_cnt) > 0)
                rd_kafka_msgq_concat(&rktp->rktp_xmit_msgq, &rktp->rktp_msgq);

        /* Timeout scan */
        if (unlikely(do_timeout_scan)) {
                rd_kafka_msgq_t timedout = RD_KAFKA_MSGQ_INITIALIZER(timedout);

                if (rd_kafka_msgq_age_scan(&rktp->rktp_xmit_msgq,
                                           &timedout, now)) {
                        /* Trigger delivery report for timed out messages */
                        rd_kafka_dr_msgq(rktp->rktp_rkt, &timedout,
                                         RD_KAFKA_RESP_ERR__MSG_TIMED_OUT);
                }
        }

        r = rd_atomic32_get(&rktp->rktp_xmit_msgq.rkmq_msg_cnt);
        if (r == 0)
                return 0;

        /* Attempt to fill the batch size, but limit
         * our waiting to queue.buffering.max.ms
         * and batch.num.messages. */
        if (r < rkb->rkb_rk->rk_conf.batch_num_messages) {
                rd_kafka_msg_t *rkm_oldest;
                rd_ts_t wait_max;

                rkm_oldest = TAILQ_FIRST(&rktp->rktp_xmit_msgq.rkmq_msgs);
                if (unlikely(!rkm_oldest))
                        return 0;

                /* Calculate maximum wait-time to
                 * honour queue.buffering.max.ms contract. */
                wait_max = rd_kafka_msg_enq_time(rktp->rktp_rkt, rkm_oldest) +
                        (rkb->rkb_rk->rk_conf.buffering_max_ms * 1000);
                if (wait_max > now) {
                        if (wait_max < *next_wakeup)
                                *next_wakeup = wait_max;
                        /* Wait for more messages or queue.buffering.max.ms
                         * to expire. */
                        return 0;
                }
        }

        /* Send Produce requests for this toppar */
        while (1) {
                r = rd_kafka_broker_produce_toppar(rkb, rktp);
                if (likely(r > 0))
                        cnt += r;
                else
                        break;
        }

        return cnt;
}


/**
 * Producer serving
 */
static void rd_kafka_broker_producer_serve (rd_kafka_broker_t *rkb) {
        rd_interval_t timeout_scan;

        rd_interval_init(&timeout_scan);

        rd_kafka_assert(rkb->rkb_rk, thrd_is_current(rkb->rkb_thread));

	rd_kafka_broker_lock(rkb);

	while (!rd_kafka_broker_terminating(rkb) &&
	       rkb->rkb_state == RD_KAFKA_BROKER_STATE_UP) {
		rd_kafka_toppar_t *rktp;
		int cnt;
		rd_ts_t now;
                rd_ts_t next_wakeup;
                int do_timeout_scan = 0;

		rd_kafka_broker_unlock(rkb);

		now = rd_clock();
                next_wakeup = now + (rkb->rkb_rk->rk_conf.
                                     socket_blocking_max_ms * 1000);

                if (rd_interval(&timeout_scan, 1000*1000, now) >= 0)
                        do_timeout_scan = 1;

		do {
			cnt = 0;

                        /* Serve each toppar */
			TAILQ_FOREACH(rktp, &rkb->rkb_toppars, rktp_rkblink) {
                                /* Serve toppar op queue */
                                rd_kafka_toppar_lock(rktp);
                                if (unlikely(rktp->rktp_leader != rkb)) {
                                        /* Currently migrating away from this
                                         * broker. */
                                        rd_kafka_toppar_unlock(rktp);
                                        continue;
                                }
				if (unlikely(RD_KAFKA_TOPPAR_IS_PAUSED(rktp))) {
					/* Partition is paused */
					rd_kafka_toppar_unlock(rktp);
					continue;
				}
                                /* Try producing toppar */
                                cnt += rd_kafka_toppar_producer_serve(
                                        rkb, rktp, do_timeout_scan, now,
                                        &next_wakeup);

                                rd_kafka_toppar_unlock(rktp);
			}

		} while (cnt);

		/* Check and move retry buffers */
		if (unlikely(rd_atomic32_get(&rkb->rkb_retrybufs.rkbq_cnt) > 0))
			rd_kafka_broker_retry_bufs_move(rkb);

                rkb->rkb_blocking_max_ms = (int)
                        (next_wakeup > now ? (next_wakeup - now) / 1000 : 0);
		rd_kafka_broker_serve(rkb, RD_POLL_NOWAIT);

		rd_kafka_broker_lock(rkb);
	}

	rd_kafka_broker_unlock(rkb);
}


#if WITH_SNAPPY
/**
 * Decompress Snappy message with Snappy-java framing.
 * Returns a malloced buffer with the uncompressed data, or NULL on failure.
 */
static char *rd_kafka_snappy_java_decompress (rd_kafka_broker_t *rkb,
					      int64_t Offset,
					      const char *inbuf,
					      size_t inlen,
					      size_t *outlenp) {
	int pass;
	char *outbuf = NULL;

	/**
	 * Traverse all chunks in two passes:
	 *  pass 1: calculate total uncompressed length
	 *  pass 2: uncompress
	 *
	 * Each chunk is prefixed with 4: length */

	for (pass = 1 ; pass <= 2 ; pass++) {
		ssize_t of = 0;  /* inbuf offset */
		ssize_t uof = 0; /* outbuf offset */

		while (of + 4 <= (ssize_t)inlen) {
			/* compressed length */
			uint32_t clen;
			/* uncompressed length */
			size_t ulen;
			int r;

                        memcpy(&clen, inbuf+of, 4);
                        clen = be32toh(clen);
			of += 4;

			if (unlikely(clen > inlen - of)) {
				rd_rkb_dbg(rkb, MSG, "SNAPPY",
					   "Invalid snappy-java chunk length for "
					   "message at offset %"PRId64" "
					   "(%"PRIu32">%"PRIusz": ignoring message",
					   Offset, clen, inlen - of);
				return NULL;
			}

			/* Acquire uncompressed length */
			if (unlikely(!rd_kafka_snappy_uncompressed_length(inbuf+of,
								 clen, &ulen))) {
				rd_rkb_dbg(rkb, MSG, "SNAPPY",
					   "Failed to get length of "
					   "(snappy-java framed) Snappy "
					   "compressed payload for message at "
					   "offset %"PRId64" (%"PRId32" bytes): "
					   "ignoring message",
					   Offset, clen);
				return NULL;
			}

			if (pass == 1) {
				/* pass 1: calculate total length */
				of  += clen;
				uof += ulen;
				continue;
			}

			/* pass 2: Uncompress to outbuf */
			if (unlikely((r = rd_kafka_snappy_uncompress(inbuf+of, clen,
							    outbuf+uof)))) {
				rd_rkb_dbg(rkb, MSG, "SNAPPY",
					   "Failed to decompress Snappy-java framed "
					   "payload for message at offset %"PRId64
					   " (%"PRId32" bytes): %s: ignoring message",
					   Offset, clen,
					   rd_strerror(-r/*negative errno*/));
				rd_free(outbuf);
				return NULL;
			}

			of  += clen;
			uof += ulen;
		}

		if (unlikely(of != (ssize_t)inlen)) {
			rd_rkb_dbg(rkb, MSG, "SNAPPY",
				   "%"PRIusz" trailing bytes in Snappy-java framed compressed "
				   "data at offset %"PRId64": ignoring message",
				   inlen - of, Offset);
			return NULL;
		}

		if (pass == 1) {
			if (uof <= 0) {
				rd_rkb_dbg(rkb, MSG, "SNAPPY",
					   "Empty Snappy-java framed data "
					   "at offset %"PRId64" (%"PRIusz" bytes): "
					   "ignoring message",
					   Offset, uof);
				return NULL;
			}

			/* Allocate memory for uncompressed data */
			outbuf = rd_malloc(uof);
			if (unlikely(!outbuf)) {
				rd_rkb_dbg(rkb, MSG, "SNAPPY",
					   "Failed to allocate memory for uncompressed "
					   "Snappy data at offset %"PRId64
					   " (%"PRIusz" bytes): %s",
					   Offset, uof, rd_strerror(errno));
				return NULL;
			}

		} else {
			/* pass 2 */
			*outlenp = uof;
		}
	}

	return outbuf;
}
#endif



/**
 * Parses a MessageSet and enqueues internal ops on the local
 * application queue for each Message.
 * @remark rkq must be a temporary unforwarded queue
 *         that requires no locking.
 */
static rd_kafka_resp_err_t
rd_kafka_messageset_handle (rd_kafka_broker_t *rkb,
			    rd_kafka_toppar_t *rktp,
			    rd_kafka_q_t *rkq,
			    struct rd_kafka_toppar_ver *tver,
			    int16_t ApiVersion,
                            rd_kafka_timestamp_type_t outer_tstype,
                            int64_t outer_timestamp,
			    rd_kafka_buf_t *rkbuf_orig,
			    void *buf, size_t size) {
        rd_kafka_buf_t *rkbuf; /* Slice of rkbuf_orig */
	rd_kafka_buf_t *rkbufz;
        /* Dont log decode errors since Fetch replies may be partial. */
        const int log_decode_errors = 0;

        /* Set up a shadow rkbuf for parsing the slice of rkbuf_orig
         * pointed out by buf,size. */
        rkbuf = rd_kafka_buf_new_shadow(buf, size);
	rkbuf->rkbuf_rkb = rkb;
	rd_kafka_broker_keep(rkb);

	if (rd_kafka_buf_remain(rkbuf) == 0)
		rd_kafka_buf_parse_fail(rkbuf,
					"%s [%"PRId32"] empty messageset",
					rktp->rktp_rkt->rkt_topic->str,
					rktp->rktp_partition);

	while (rd_kafka_buf_remain(rkbuf) > 0) {
		struct {
			int64_t Offset;
			int32_t MessageSize;
			uint32_t Crc;
			int8_t  MagicByte; /* MsgVersion */
			int8_t  Attributes;
			int64_t Timestamp;
		} hdr;
		rd_kafkap_bytes_t Key;
		rd_kafkap_bytes_t Value;
		int32_t Value_len;
		rd_kafka_op_t *rko;
		size_t outlen;
		void *outbuf = NULL; /* Uncompressed output buffer. */
		size_t hdrsize = 6; /* Header size following MessageSize */
		int relative_offsets;
                size_t crc_start_of;
                int32_t crc_start_remain;
                rd_kafka_resp_err_t err RD_UNUSED = RD_KAFKA_RESP_ERR_NO_ERROR;

		rd_kafka_buf_read_i64(rkbuf, &hdr.Offset);
		rd_kafka_buf_read_i32(rkbuf, &hdr.MessageSize);
		rd_kafka_buf_read_i32(rkbuf, &hdr.Crc);
                crc_start_of = rkbuf->rkbuf_of;
                crc_start_remain = rd_kafka_buf_remain(rkbuf);
		rd_kafka_buf_read_i8(rkbuf, &hdr.MagicByte);
		rd_kafka_buf_read_i8(rkbuf, &hdr.Attributes);


                if (rkb->rkb_rk->rk_conf.check_crcs &&
                    !(hdr.Attributes & RD_KAFKA_MSG_ATTR_COMPRESSION_MASK)) {
                        /* Verify CRC if desired.
                         * Since compression algorithms have their own
                         * checksums we dont bother checking CRCs of compressed
                         * messagesets. */
                        uint32_t calc_crc;

                        if (hdr.MessageSize - 4 > crc_start_remain)
                                goto err; /* Ignore partial messages */

                        calc_crc = rd_crc32(rkbuf->rkbuf_rbuf+crc_start_of,
                                            hdr.MessageSize-4);

                        if (unlikely(hdr.Crc != calc_crc)) {
                                rd_kafka_q_op_err(rkq, RD_KAFKA_OP_CONSUMER_ERR,
                                                  RD_KAFKA_RESP_ERR__BAD_MSG,
                                                  tver->version,
                                                  rktp,
                                                  hdr.Offset,
                                                  "Message at offset %"PRId64
                                                  " of %"PRId32" bytes "
                                                  "failed CRC check "
                                                  "(original 0x%"PRIx32" != "
                                                  "calculated 0x%"PRIx32")",
                                                  hdr.Offset, hdr.MessageSize,
                                                  hdr.Crc, calc_crc);
                                rd_kafka_buf_skip(rkbuf, hdr.MessageSize-4);
                                continue;
                        }
                }


		if (hdr.MagicByte == 1) { /* MsgVersion */
			rd_kafka_buf_read_i64(rkbuf, &hdr.Timestamp);
			hdrsize += 8;
		} else
			hdr.Timestamp = 0;

                if (hdr.MessageSize - (ssize_t)hdrsize >
		    rd_kafka_buf_remain(rkbuf)) {
                        /* Broker may send partial messages.
                         * Bail out silently.
			 * "A Guide To The Kafka Protocol" states:
			 *   "As an optimization the server is allowed to
			 *    return a partial message at the end of the
			 *    message set.
			 *    Clients should handle this case."
			 * We're handling it by not passing the error upstream.
			 */
                        goto err;
                }

		/* Extract key */
		rd_kafka_buf_read_bytes(rkbuf, &Key);

		/* Extract Value */
		rd_kafka_buf_read_bytes(rkbuf, &Value);

		Value_len = RD_KAFKAP_BYTES_LEN(&Value);

		/* Check for message compression.
		 * The Key is ignored for compressed messages. */
		switch (hdr.Attributes & RD_KAFKA_MSG_ATTR_COMPRESSION_MASK)
		{
		case RD_KAFKA_COMPRESSION_NONE:
		{
			rd_kafka_msg_t *rkm;

			/* Pure uncompressed message, this is the innermost
			 * handler after all compression and cascaded
			 * messagesets have been peeled off. */

                        /* MessageSets may contain offsets earlier than we
                         * requested (compressed messagesets in particular),
                         * drop the earlier messages.
			 * Note: the inner offset may only be trusted for
			 *       absolute offsets. KIP-31 introduced
			 *       ApiVersion 2 that maintains relative offsets
			 *       of compressed messages and the base offset
			 *       in the outer message is the offset of
			 *       the *LAST* message in the MessageSet.
			 *       This requires us to assign offsets
			 *       after all messages have been read from
			 *       the messageset, and it also means
			 *       we cant perform this offset check here
			 *       in that case. */
			relative_offsets = ApiVersion == 2;

                        if (!relative_offsets &&
			    hdr.Offset < rktp->rktp_offsets.fetch_offset)
                                continue;

			rko = rd_kafka_op_new(RD_KAFKA_OP_FETCH);
			rko->rko_rktp = rd_kafka_toppar_keep(rktp);
			rko->rko_version = tver->version;
			rkm = &rko->rko_u.fetch.rkm;

			/* Since all the ops share the same payload buffer
			 * (rkbuf->rkbuf_buf2) a refcnt is used on the rkbuf
			 * that makes sure all consume_cb() will have been
			 * called for each of these ops before the rkbuf
			 * and its rkbuf_buf2 are freed. */
			rko->rko_u.fetch.rkbuf = rkbuf_orig; /* original rkbuf */
			rd_kafka_buf_keep(rkbuf_orig);

                        if (outer_tstype != RD_KAFKA_TIMESTAMP_NOT_AVAILABLE) {
                                rkm->rkm_timestamp = outer_timestamp;
                                rkm->rkm_tstype = outer_tstype;
                        } else if (hdr.MagicByte >= 1 && hdr.Timestamp) {
				rkm->rkm_timestamp = hdr.Timestamp;
				if (hdr.Attributes & RD_KAFKA_MSG_ATTR_LOG_APPEND_TIME)
					rkm->rkm_tstype = RD_KAFKA_TIMESTAMP_LOG_APPEND_TIME;
				else
					rkm->rkm_tstype = RD_KAFKA_TIMESTAMP_CREATE_TIME;
			}

			if (!RD_KAFKAP_BYTES_IS_NULL(&Key)) {
				rkm->rkm_key     = (void *)Key.data;
				rkm->rkm_key_len = RD_KAFKAP_BYTES_LEN(&Key);
			}

                        /* Forward NULL message notation to application. */
			rkm->rkm_payload = RD_KAFKAP_BYTES_IS_NULL(&Value) ?
                                NULL : (void *)Value.data;
			rkm->rkm_len = Value_len;
			rko->rko_len = (int32_t)rkm->rkm_len;

			rkm->rkm_offset    = hdr.Offset;
			rkm->rkm_partition = rktp->rktp_partition;

			if (0)
			rd_rkb_dbg(rkb, MSG, "MSG",
				   "Pushed message at offset %"PRId64
				   " onto queue", hdr.Offset);

			rd_kafka_q_enq(rkq, rko);
		}
		break;

#if WITH_ZLIB
		case RD_KAFKA_COMPRESSION_GZIP:
		{
			uint64_t outlenx = 0;

			/* Decompress Message payload */
			outbuf = rd_gz_decompress(Value.data, Value_len,
						  &outlenx);
			if (unlikely(!outbuf)) {
				rd_rkb_dbg(rkb, MSG, "GZIP",
					   "Failed to decompress Gzip "
					   "message at offset %"PRId64
					   " of %"PRId32" bytes: "
					   "ignoring message",
					   hdr.Offset, Value_len);
                                err = RD_KAFKA_RESP_ERR__BAD_COMPRESSION;
                                goto per_msg_err;
			}

			outlen = (size_t)outlenx;
		}
		break;
#endif

#if WITH_SNAPPY
		case RD_KAFKA_COMPRESSION_SNAPPY:
		{
			const char *inbuf = Value.data;
			int r;
			static const unsigned char snappy_java_magic[] =
				{ 0x82, 'S','N','A','P','P','Y', 0 };
			static const int snappy_java_hdrlen = 8+4+4;

			/* snappy-java adds its own header (SnappyCodec)
			 * which is not compatible with the official Snappy
			 * implementation.
			 *   8: magic, 4: version, 4: compatible
			 * followed by any number of chunks:
			 *   4: length
			 * ...: snappy-compressed data. */
			if (likely(Value_len > snappy_java_hdrlen + 4 &&
				   !memcmp(inbuf, snappy_java_magic, 8))) {
				/* snappy-java framing */

				inbuf = inbuf + snappy_java_hdrlen;
				Value_len -= snappy_java_hdrlen;
				outbuf = rd_kafka_snappy_java_decompress(rkb,
									 hdr.Offset,
									 inbuf,
									 Value_len,
									 &outlen);
				if (unlikely(!outbuf)) {
                                        err = RD_KAFKA_RESP_ERR__BAD_COMPRESSION;
                                        goto per_msg_err;
                                }


			} else {
				/* no framing */

				/* Acquire uncompressed length */
				if (unlikely(!rd_kafka_snappy_uncompressed_length(inbuf,
									 Value_len,
									 &outlen))) {
					rd_rkb_dbg(rkb, MSG, "SNAPPY",
						   "Failed to get length of Snappy "
						   "compressed payload "
						   "for message at offset %"PRId64
						   " (%"PRId32" bytes): "
						   "ignoring message",
						   hdr.Offset, Value_len);
                                        err = RD_KAFKA_RESP_ERR__BAD_COMPRESSION;
                                        goto per_msg_err;
				}

				/* Allocate output buffer for uncompressed data */
				outbuf = rd_malloc(outlen);

				/* Uncompress to outbuf */
				if (unlikely((r = rd_kafka_snappy_uncompress(inbuf,
								    Value_len,
								    outbuf)))) {
					rd_rkb_dbg(rkb, MSG, "SNAPPY",
						   "Failed to decompress Snappy "
						   "payload for message at offset "
						   "%"PRId64
						   " (%"PRId32" bytes): %s: "
						   "ignoring message",
						   hdr.Offset, Value_len,
						   rd_strerror(-r/*negative errno*/));
					rd_free(outbuf);
                                        err = RD_KAFKA_RESP_ERR__BAD_COMPRESSION;
                                        goto per_msg_err;
				}
			}

		}
		break;
#endif

                case RD_KAFKA_COMPRESSION_LZ4:
		{
                        err = rd_kafka_lz4_decompress(rkb,
						      /* Proper HC? */
						      hdr.MagicByte >= 1 ? 1 : 0,
						      hdr.Offset,
                                                      (void *)Value.data,
                                                      Value_len,
						      &outbuf, &outlen);
                        if (err)
                                goto per_msg_err;
                }
                break;


		default:
                        err = RD_KAFKA_RESP_ERR__BAD_COMPRESSION;
			rd_rkb_dbg(rkb, MSG, "CODEC",
				   "%s [%"PRId32"]: Message at offset %"PRId64
				   " with unsupported "
				   "compression codec 0x%x: message ignored",
				   rktp->rktp_rkt->rkt_topic->str,
				   rktp->rktp_partition,
				   hdr.Offset, (int)hdr.Attributes);

		per_msg_err:
			/* Enqueue error messsage */
			/* Create op and push on temporary queue. */
			rd_kafka_q_op_err(rkq, RD_KAFKA_OP_CONSUMER_ERR,
					  RD_KAFKA_RESP_ERR__NOT_IMPLEMENTED,
					  tver->version, rktp,
					  hdr.Offset,
					  "Unsupported compression codec "
					  "0x%x", (int)hdr.Attributes);
			break;
		}


		if (outbuf) {
			rd_kafka_q_t relq; /* Temporary queue for use with
					    * relative offsets. */
			int relative_offsets = ApiVersion == 2;
                        rd_kafka_timestamp_type_t use_tstype =
                                RD_KAFKA_TIMESTAMP_NOT_AVAILABLE;
                        int64_t use_timestamp = 0;

                        if (hdr.MagicByte >= 1 &&
                            (hdr.Attributes&RD_KAFKA_MSG_ATTR_LOG_APPEND_TIME)){
                                use_tstype = RD_KAFKA_TIMESTAMP_LOG_APPEND_TIME;
                                use_timestamp = hdr.Timestamp;
                        }

			/* With a new allocated buffer (outbuf) we need
			 * a separate rkbuf for it to allow multiple fetch ops
			 * to share the same payload buffer. */
			rkbufz = rd_kafka_buf_new_shadow(outbuf, outlen);

			if (relative_offsets) {
				rd_kafka_q_init(&relq, rkb->rkb_rk);
                                /* Make sure enqueued ops get the
                                 * correct serve/opaque reflecting the
                                 * original queue. */
                                relq.rkq_serve = rkq->rkq_serve;
                                relq.rkq_opaque = rkq->rkq_opaque;
                        }

			/* Now parse the contained Messages */
			rd_kafka_messageset_handle(rkb, rktp,
						   relative_offsets ?
						   &relq : rkq, tver,
						   ApiVersion,
                                                   use_tstype,
                                                   use_timestamp,
						   rkbufz, outbuf, outlen);


			if (relative_offsets) {
				/* Update messages to absolute offsets
				 * and purge any messages older than the current
				 * fetch offset. */
				rd_kafka_q_fix_offsets(
					&relq, rktp->rktp_offsets.fetch_offset,
					hdr.Offset - rd_kafka_q_len(&relq) + 1);

				/* Append messages to proper queue. */
				rd_kafka_q_concat0(rkq, &relq, 0/*no-lock*/);
				rd_kafka_q_destroy(&relq);
			}

			/* Loose our refcnt of the rkbuf.
			 * Individual rko's will have their own. */
			rd_kafka_buf_destroy(rkbufz);
		}

	}

	goto done;
 err:
        /* Count all errors as partial message errors. */
        rd_atomic64_add(&rkb->rkb_c.rx_partial, 1);

 done:
        /* rkbuf is a temporary shadow of rkbuf_orig, reset buf2 pointer
         * to avoid it being freed now. */
        rkbuf->rkbuf_buf2 = NULL;
        rd_kafka_buf_destroy(rkbuf);

	return RD_KAFKA_RESP_ERR_NO_ERROR;
}


/**
 * Backoff the next Fetch request (due to error).
 */
static void rd_kafka_broker_fetch_backoff (rd_kafka_broker_t *rkb) {
        int backoff_ms = rkb->rkb_rk->rk_conf.fetch_error_backoff_ms;
        rkb->rkb_ts_fetch_backoff = rd_clock() + (backoff_ms * 1000);

        if (rkb->rkb_blocking_max_ms > backoff_ms)
                rkb->rkb_blocking_max_ms = backoff_ms;
}


/**
 * Parses and handles a Fetch reply.
 * Returns 0 on success or an error code on failure.
 */
static rd_kafka_resp_err_t
rd_kafka_fetch_reply_handle (rd_kafka_broker_t *rkb,
			     rd_kafka_buf_t *rkbuf, rd_kafka_buf_t *request) {
	int32_t TopicArrayCnt;
	int i;
        const int log_decode_errors = 1;
        shptr_rd_kafka_itopic_t *s_rkt = NULL;

	if (request->rkbuf_reqhdr.ApiVersion >= 1) { /* v1 & v2 */
		int32_t Throttle_Time;
		rd_kafka_buf_read_i32(rkbuf, &Throttle_Time);

		rd_kafka_op_throttle_time(rkb, rkb->rkb_rk->rk_rep,
					  Throttle_Time);
	}

	rd_kafka_buf_read_i32(rkbuf, &TopicArrayCnt);
	/* Verify that TopicArrayCnt seems to be in line with remaining size */
	rd_kafka_buf_check_len(rkbuf,
			       TopicArrayCnt * (3/*topic min size*/ +
						4/*PartitionArrayCnt*/ +
						4+2+8+4/*inner header*/));

	for (i = 0 ; i < TopicArrayCnt ; i++) {
		rd_kafkap_str_t topic;
		rd_kafka_toppar_t *rktp;
		shptr_rd_kafka_toppar_t *s_rktp = NULL;
		int32_t fetch_version;
		int32_t PartitionArrayCnt;
		struct {
			int32_t Partition;
			int16_t ErrorCode;
			int64_t HighwaterMarkOffset;
			int32_t MessageSetSize;
		} hdr;
		rd_kafka_resp_err_t err2;
		int j;

		rd_kafka_buf_read_str(rkbuf, &topic);
		rd_kafka_buf_read_i32(rkbuf, &PartitionArrayCnt);

		rd_kafka_buf_check_len(rkbuf,
				       PartitionArrayCnt *
				       (4+2+8+4/*inner header*/));

                s_rkt = rd_kafka_topic_find0(rkb->rkb_rk, &topic);

		for (j = 0 ; j < PartitionArrayCnt ; j++) {
			rd_kafka_q_t tmp_opq; /* Temporary queue for ops */
			struct rd_kafka_toppar_ver *tver, tver_skel;

			rd_kafka_buf_read_i32(rkbuf, &hdr.Partition);
			rd_kafka_buf_read_i16(rkbuf, &hdr.ErrorCode);
			rd_kafka_buf_read_i64(rkbuf, &hdr.HighwaterMarkOffset);
			rd_kafka_buf_read_i32(rkbuf, &hdr.MessageSetSize);

                        if (hdr.MessageSetSize < 0)
                                rd_kafka_buf_parse_fail(
                                        rkbuf,
                                        "%.*s [%"PRId32"]: "
                                        "invalid MessageSetSize %"PRId32,
                                        RD_KAFKAP_STR_PR(&topic),
                                        hdr.Partition,
                                        hdr.MessageSetSize);

			/* Look up topic+partition */
                        if (likely(s_rkt != NULL)) {
                                rd_kafka_itopic_t *rkt;
                                rkt = rd_kafka_topic_s2i(s_rkt);
                                rd_kafka_topic_rdlock(rkt);
                                s_rktp = rd_kafka_toppar_get(
                                        rkt, hdr.Partition, 0/*no ua-on-miss*/);
                                rd_kafka_topic_rdunlock(rkt);
                        }

			if (unlikely(!s_rkt || !s_rktp)) {
				rd_rkb_dbg(rkb, TOPIC, "UNKTOPIC",
					   "Received Fetch response "
					   "(error %hu) for unknown topic "
					   "%.*s [%"PRId32"]: ignoring",
					   hdr.ErrorCode,
					   RD_KAFKAP_STR_PR(&topic),
					   hdr.Partition);
				rd_kafka_buf_skip(rkbuf, hdr.MessageSetSize);
				continue;
			}

                        rktp = rd_kafka_toppar_s2i(s_rktp);

                        rd_kafka_toppar_lock(rktp);
                        /* Make sure toppar hasn't moved to another broker
                         * during the lifetime of the request. */
                        if (unlikely(rktp->rktp_leader != rkb)) {
                                rd_kafka_toppar_unlock(rktp);
                                rd_rkb_dbg(rkb, MSG, "FETCH",
                                           "%.*s [%"PRId32"]: "
                                           "partition leadership changed: "
                                           "discarding fetch response",
                                           RD_KAFKAP_STR_PR(&topic),
                                           hdr.Partition);
                                rd_kafka_toppar_destroy(s_rktp); /* from get */
                                rd_kafka_buf_skip(rkbuf, hdr.MessageSetSize);
                                continue;
                        }
			fetch_version = rktp->rktp_fetch_version;
                        rd_kafka_toppar_unlock(rktp);

			/* Check if this Fetch is for an outdated fetch version,
                         * if so ignore it. */
			tver_skel.s_rktp = s_rktp;
			tver = rd_list_find(request->rkbuf_rktp_vers,
					    &tver_skel,
					    rd_kafka_toppar_ver_cmp);
			rd_kafka_assert(NULL, tver &&
					rd_kafka_toppar_s2i(tver->s_rktp) ==
					rktp);
			if (tver->version < fetch_version) {
				rd_rkb_dbg(rkb, MSG, "DROP",
					   "%s [%"PRId32"]: "
					   "dropping outdated fetch response "
					   "(v%d < %d)",
					   rktp->rktp_rkt->rkt_topic->str,
					   rktp->rktp_partition,
					   tver->version, fetch_version);
                                rd_atomic64_add(&rktp->rktp_c. rx_ver_drops, 1);
                                rd_kafka_toppar_destroy(s_rktp); /* from get */
                                rd_kafka_buf_skip(rkbuf, hdr.MessageSetSize);
                                continue;
                        }

			rd_rkb_dbg(rkb, MSG, "FETCH",
				   "Topic %.*s [%"PRId32"] MessageSet "
				   "size %"PRId32", error \"%s\", "
				   "MaxOffset %"PRId64", "
                                   "Ver %"PRId32"/%"PRId32,
				   RD_KAFKAP_STR_PR(&topic), hdr.Partition,
				   hdr.MessageSetSize,
				   rd_kafka_err2str(hdr.ErrorCode),
				   hdr.HighwaterMarkOffset,
                                   tver->version, fetch_version);


                        /* Update hi offset to be able to compute
                         * consumer lag. */
                        rktp->rktp_offsets.hi_offset = hdr.HighwaterMarkOffset;


			/* High offset for get_watermark_offsets() */
			rd_kafka_toppar_lock(rktp);
			rktp->rktp_hi_offset = hdr.HighwaterMarkOffset;
			rd_kafka_toppar_unlock(rktp);

			/* If this is the last message of the queue,
			 * signal EOF back to the application. */
			if (hdr.HighwaterMarkOffset ==
                            rktp->rktp_offsets.fetch_offset
			    &&
			    rktp->rktp_offsets.eof_offset !=
                            rktp->rktp_offsets.fetch_offset) {
				hdr.ErrorCode =
					RD_KAFKA_RESP_ERR__PARTITION_EOF;
				rktp->rktp_offsets.eof_offset =
                                        rktp->rktp_offsets.fetch_offset;
			}

			/* Handle partition-level errors. */
			if (unlikely(hdr.ErrorCode !=
				     RD_KAFKA_RESP_ERR_NO_ERROR)) {
				/* Some errors should be passed to the
				 * application while some handled by rdkafka */
				switch (hdr.ErrorCode)
				{
					/* Errors handled by rdkafka */
				case RD_KAFKA_RESP_ERR_UNKNOWN_TOPIC_OR_PART:
				case RD_KAFKA_RESP_ERR_LEADER_NOT_AVAILABLE:
				case RD_KAFKA_RESP_ERR_NOT_LEADER_FOR_PARTITION:
				case RD_KAFKA_RESP_ERR_BROKER_NOT_AVAILABLE:
                                        /* Request metadata information update*/
                                        rd_kafka_toppar_leader_unavailable(
                                                rktp, "fetch", hdr.ErrorCode);
                                        break;

					/* Application errors */
				case RD_KAFKA_RESP_ERR_OFFSET_OUT_OF_RANGE:
                                {
                                        int64_t err_offset =
                                                rktp->rktp_offsets.fetch_offset;
                                        rktp->rktp_offsets.fetch_offset =
                                                RD_KAFKA_OFFSET_INVALID;
					rd_kafka_offset_reset(
						rktp, err_offset,
						hdr.ErrorCode,
						rd_kafka_err2str(hdr.
								 ErrorCode));
                                }
                                break;
				case RD_KAFKA_RESP_ERR__PARTITION_EOF:
					if (!rkb->rkb_rk->rk_conf.enable_partition_eof)
						break;
					/* FALLTHRU */
				case RD_KAFKA_RESP_ERR_MSG_SIZE_TOO_LARGE:
				default: /* and all other errors */
					rd_dassert(tver->version > 0);
					rd_kafka_q_op_err(
						rktp->rktp_fetchq,
						RD_KAFKA_OP_CONSUMER_ERR,
						hdr.ErrorCode, tver->version,
						rktp,
						rktp->rktp_offsets.fetch_offset,
						"%s",
						rd_kafka_err2str(hdr.ErrorCode));
					break;
				}

                                /* FIXME: only back off this rktp */
				rd_kafka_broker_fetch_backoff(rkb);

				rd_kafka_toppar_destroy(s_rktp);/* from get()*/

                                rd_kafka_buf_skip(rkbuf, hdr.MessageSetSize);
				continue;
			}

			if (hdr.MessageSetSize <= 0) {
				rd_kafka_toppar_destroy(s_rktp); /*from get()*/
				continue;
			}

			/* All parsed messages are put on this temporary op
			 * queue first and then moved in one go to the
			 * real op queue. */
			rd_kafka_q_init(&tmp_opq, rkb->rkb_rk);
                        /* Make sure enqueued ops get the
                         * correct serve/opaque reflecting the
                         * original queue. */
                        tmp_opq.rkq_serve = rktp->rktp_fetchq->rkq_serve;
                        tmp_opq.rkq_opaque = rktp->rktp_fetchq->rkq_opaque;

			/* Parse and handle the message set */
			err2 = rd_kafka_messageset_handle(
				rkb, rktp, &tmp_opq, tver,
				request->rkbuf_reqhdr.ApiVersion,
                                RD_KAFKA_TIMESTAMP_NOT_AVAILABLE, -1,
				rkbuf, rkbuf->rkbuf_rbuf+rkbuf->rkbuf_of,
				hdr.MessageSetSize);
			if (err2) {
				rd_kafka_q_destroy(&tmp_opq);
				rd_kafka_toppar_destroy(s_rktp);/* from get()*/
				rd_kafka_buf_parse_fail(rkbuf, "messageset handle failed");
                                RD_NOTREACHED();
			}

			/* Concat all messages onto the real op queue */
			rd_rkb_dbg(rkb, MSG | RD_KAFKA_DBG_FETCH, "CONSUME",
				   "Enqueue %i messages on %s [%"PRId32"] "
				   "fetch queue (qlen %i, v%d)",
				   rd_kafka_q_len(&tmp_opq),
				   rktp->rktp_rkt->rkt_topic->str,
				   rktp->rktp_partition,
				   rd_kafka_q_len(rktp->rktp_fetchq),
				   tver->version);

			if (rd_kafka_q_len(&tmp_opq) > 0) {
				/* Update partitions fetch offset based on
				 * last message's offest. */
				int64_t last_offset = -1;
				rd_kafka_op_t *rko =
					rd_kafka_q_last(&tmp_opq,
							RD_KAFKA_OP_FETCH,
							0 /* no error ops */);

				if (rko)
					last_offset = rko->rko_u.fetch.rkm.rkm_offset;

				if (rd_kafka_q_concat(rktp->rktp_fetchq,
						      &tmp_opq) == -1) {
					/* rktp fetchq disabled, probably
					 * shutting down. Drop messages. */
					rd_kafka_q_purge0(&tmp_opq,
							  0/*no-lock*/);
				} else {
					if (last_offset != -1)
						rktp->rktp_offsets.fetch_offset =
							last_offset + 1;
					rd_atomic64_add(&rktp->rktp_c.msgs,
							rd_kafka_q_len(&tmp_opq));
				}
                        } else {
                                /* The message set didn't contain any full
                                 * message. This means that the size limit was
                                 * too tight. */
                                if (rktp->rktp_fetch_msg_max_bytes < (1 << 30)) {
                                        rktp->rktp_fetch_msg_max_bytes *= 2;
					rd_rkb_dbg(rkb, FETCH, "CONSUME",
						   "Topic %s [%"PRId32"]: Increasing "
						   "max fetch bytes to %"PRId32,
						   rktp->rktp_rkt->rkt_topic->str,
						   rktp->rktp_partition,
						   rktp->rktp_fetch_msg_max_bytes);
				} else {
					rd_kafka_q_op_err(
						rktp->rktp_fetchq,
						RD_KAFKA_OP_CONSUMER_ERR,
						RD_KAFKA_RESP_ERR_MSG_SIZE_TOO_LARGE,
						tver->version,
						rktp,
						rktp->rktp_offsets.fetch_offset,
						"Message at offset %"PRId64" "
						"is too large to fetch, try increasing "
						"receive.message.max.bytes",
						rktp->rktp_offsets.fetch_offset);
				}
                        }

			rd_kafka_toppar_destroy(s_rktp); /* from get() */

			rd_kafka_q_destroy(&tmp_opq);

			rd_kafka_buf_skip(rkbuf, hdr.MessageSetSize);
		}

                if (s_rkt) {
                        rd_kafka_topic_destroy0(s_rkt);
                        s_rkt = NULL;
                }
	}

	if (rd_kafka_buf_remain(rkbuf) != 0) {
		rd_kafka_buf_parse_fail(rkbuf,
					"Remaining data after message set "
					"parse: %i bytes",
					rd_kafka_buf_remain(rkbuf));
		RD_NOTREACHED();
	}

	return 0;

err:
        if (s_rkt)
                rd_kafka_topic_destroy0(s_rkt);
	rd_rkb_dbg(rkb, MSG, "BADMSG", "Bad message (Fetch v%d): "
		   "is broker.version.fallback incorrectly set?",
		   (int)request->rkbuf_reqhdr.ApiVersion);
	return RD_KAFKA_RESP_ERR__BAD_MSG;
}



static void rd_kafka_broker_fetch_reply (rd_kafka_t *rk,
					 rd_kafka_broker_t *rkb,
					 rd_kafka_resp_err_t err,
					 rd_kafka_buf_t *reply,
					 rd_kafka_buf_t *request,
					 void *opaque) {
	rd_kafka_assert(rkb->rkb_rk, rkb->rkb_fetching > 0);
	rkb->rkb_fetching = 0;

	/* Parse and handle the messages (unless the request errored) */
	if (!err && reply)
		err = rd_kafka_fetch_reply_handle(rkb, reply, request);

	if (unlikely(err)) {
                char tmp[128];

                rd_rkb_dbg(rkb, MSG, "FETCH", "Fetch reply: %s",
                           rd_kafka_err2str(err));
		switch (err)
		{
		case RD_KAFKA_RESP_ERR_UNKNOWN_TOPIC_OR_PART:
		case RD_KAFKA_RESP_ERR_LEADER_NOT_AVAILABLE:
		case RD_KAFKA_RESP_ERR_NOT_LEADER_FOR_PARTITION:
		case RD_KAFKA_RESP_ERR_BROKER_NOT_AVAILABLE:
		case RD_KAFKA_RESP_ERR_REPLICA_NOT_AVAILABLE:
                        /* Request metadata information update */
                        rd_snprintf(tmp, sizeof(tmp),
                                    "FetchRequest failed: %s",
                                    rd_kafka_err2str(err));
                        rd_kafka_metadata_refresh_known_topics(rkb->rkb_rk,
                                                               NULL, 1/*force*/,
                                                               tmp);
                        /* FALLTHRU */

		case RD_KAFKA_RESP_ERR__TRANSPORT:
		case RD_KAFKA_RESP_ERR_REQUEST_TIMED_OUT:
                case RD_KAFKA_RESP_ERR__MSG_TIMED_OUT:
			/* The fetch is already intervalled from
                         * consumer_serve() so dont retry. */
			break;

		default:
			break;
		}

		rd_kafka_broker_fetch_backoff(rkb);
		/* FALLTHRU */
	}
}











/**
 * Build and send a Fetch request message for all underflowed toppars
 * for a specific broker.
 */
static int rd_kafka_broker_fetch_toppars (rd_kafka_broker_t *rkb) {
	rd_kafka_toppar_t *rktp;
	rd_kafka_buf_t *rkbuf;
	rd_ts_t now = rd_clock();
	int cnt = 0;
	size_t of_TopicArrayCnt = 0;
	int TopicArrayCnt = 0;
	size_t of_PartitionArrayCnt = 0;
	int PartitionArrayCnt = 0;
	rd_kafka_itopic_t *rkt_last = NULL;

	/* Create buffer and iovecs:
	 *   1 x ReplicaId MaxWaitTime MinBytes TopicArrayCnt
	 *   N x topic name
	 *   N x PartitionArrayCnt Partition FetchOffset MaxBytes
	 * where N = number of toppars.
	 * Since we dont keep track of the number of topics served by
	 * this broker, only the partition count, we do a worst-case calc
	 * when allocation iovecs and assume each partition is on its own topic
	 */

        if (unlikely(rkb->rkb_fetch_toppar_cnt == 0))
                return 0;

	rkbuf = rd_kafka_buf_new_growable(
                rkb->rkb_rk, RD_KAFKAP_Fetch, 1,
                /* ReplicaId+MaxWaitTime+MinBytes+TopicCnt */
                4+4+4+4+
                /* N x PartCnt+Partition+FetchOffset+MaxBytes+?TopicNameLen?*/
                (rkb->rkb_fetch_toppar_cnt * (4+4+8+4+40)));

	/* FetchRequest header */
	/* ReplicaId */
	rd_kafka_buf_write_i32(rkbuf, -1);
	/* MaxWaitTime */
	rd_kafka_buf_write_i32(rkbuf, rkb->rkb_rk->rk_conf.fetch_wait_max_ms);
	/* MinBytes */
	rd_kafka_buf_write_i32(rkbuf, rkb->rkb_rk->rk_conf.fetch_min_bytes);

	/* Write zero TopicArrayCnt but store pointer for later update */
	of_TopicArrayCnt = rd_kafka_buf_write_i32(rkbuf, 0);

        /* Prepare map for storing the fetch version for each partition,
         * this will later be checked in Fetch response to purge outdated
         * responses (e.g., after a seek). */
        rkbuf->rkbuf_rktp_vers = rd_list_new(
                0, (void *)rd_kafka_toppar_ver_destroy);
        rd_list_prealloc_elems(rkbuf->rkbuf_rktp_vers,
                               sizeof(struct rd_kafka_toppar_ver),
                               rkb->rkb_fetch_toppar_cnt);

	/* Round-robin start of the list. */
        rktp = rkb->rkb_fetch_toppar_next;
        do {
		struct rd_kafka_toppar_ver *tver;

		if (rkt_last != rktp->rktp_rkt) {
			if (rkt_last != NULL) {
				/* Update PartitionArrayCnt */
				rd_kafka_buf_update_i32(rkbuf,
							of_PartitionArrayCnt,
							PartitionArrayCnt);
			}

                        /* Topic name */
			rd_kafka_buf_write_kstr(rkbuf,
                                                rktp->rktp_rkt->rkt_topic);
			TopicArrayCnt++;
			rkt_last = rktp->rktp_rkt;
                        /* Partition count */
			of_PartitionArrayCnt = rd_kafka_buf_write_i32(rkbuf, 0);
			PartitionArrayCnt = 0;
		}

		PartitionArrayCnt++;
		/* Partition */
		rd_kafka_buf_write_i32(rkbuf, rktp->rktp_partition);
		/* FetchOffset */
		rd_kafka_buf_write_i64(rkbuf, rktp->rktp_offsets.fetch_offset);
		/* MaxBytes */
		rd_kafka_buf_write_i32(rkbuf, rktp->rktp_fetch_msg_max_bytes);

		rd_rkb_dbg(rkb, FETCH, "FETCH",
			   "Fetch topic %.*s [%"PRId32"] at offset %"PRId64
			   " (v%d)",
			   RD_KAFKAP_STR_PR(rktp->rktp_rkt->rkt_topic),
			   rktp->rktp_partition,
                           rktp->rktp_offsets.fetch_offset,
			   rktp->rktp_fetch_version);

		/* Add toppar + op version mapping. */
		tver = rd_list_add(rkbuf->rkbuf_rktp_vers, NULL);
		tver->s_rktp = rd_kafka_toppar_keep(rktp);
		tver->version = rktp->rktp_fetch_version;

		cnt++;
	} while ((rktp = CIRCLEQ_LOOP_NEXT(&rkb->rkb_fetch_toppars,
                                           rktp, rktp_fetchlink)) !=
                 rkb->rkb_fetch_toppar_next);

        /* Update next toppar to fetch in round-robin list. */
        rd_kafka_broker_fetch_toppar_next(rkb,
                                          rktp ?
                                          CIRCLEQ_LOOP_NEXT(&rkb->
                                                            rkb_fetch_toppars,
                                                            rktp, rktp_fetchlink):
                                          NULL);

	rd_rkb_dbg(rkb, FETCH, "FETCH", "Fetch %i/%i/%i toppar(s)",
                   cnt, rkb->rkb_fetch_toppar_cnt, rkb->rkb_toppar_cnt);
	if (!cnt) {
		rd_kafka_buf_destroy(rkbuf);
		return cnt;
	}

	if (rkt_last != NULL) {
		/* Update last topic's PartitionArrayCnt */
		rd_kafka_buf_update_i32(rkbuf,
					of_PartitionArrayCnt,
					PartitionArrayCnt);
	}

	/* Update TopicArrayCnt */
	rd_kafka_buf_update_i32(rkbuf, of_TopicArrayCnt, TopicArrayCnt);

	/* Use configured timeout */
	rkbuf->rkbuf_ts_timeout = now +
		((rkb->rkb_rk->rk_conf.socket_timeout_ms +
		  rkb->rkb_rk->rk_conf.fetch_wait_max_ms) * 1000);

        if (rkb->rkb_features & RD_KAFKA_FEATURE_MSGVER1)
                rd_kafka_buf_ApiVersion_set(rkbuf, 2,
                                            RD_KAFKA_FEATURE_MSGVER1);
        else if (rkb->rkb_features & RD_KAFKA_FEATURE_THROTTLETIME)
                rd_kafka_buf_ApiVersion_set(rkbuf, 1,
                                            RD_KAFKA_FEATURE_THROTTLETIME);

        rd_kafka_buf_autopush(rkbuf);

	/* Sort toppar versions for quicker lookups in Fetch response. */
	rd_list_sort(rkbuf->rkbuf_rktp_vers, rd_kafka_toppar_ver_cmp);

	rkb->rkb_fetching = 1;
        rd_kafka_broker_buf_enq1(rkb, rkbuf, rd_kafka_broker_fetch_reply, NULL);

	return cnt;
}




/**
 * Consumer serving
 */
static void rd_kafka_broker_consumer_serve (rd_kafka_broker_t *rkb) {

	rd_kafka_assert(rkb->rkb_rk, thrd_is_current(rkb->rkb_thread));

	rd_kafka_broker_lock(rkb);

	while (!rd_kafka_broker_terminating(rkb) &&
	       rkb->rkb_state == RD_KAFKA_BROKER_STATE_UP) {
		rd_ts_t now;

		rd_kafka_broker_unlock(rkb);

		now = rd_clock();

                /* Serve toppars */
                rd_kafka_broker_toppars_serve(rkb);

		/* Send Fetch request message for all underflowed toppars */
		if (!rkb->rkb_fetching) {
                        if (rkb->rkb_ts_fetch_backoff < now) {
                                rd_kafka_broker_fetch_toppars(rkb);
                                rkb->rkb_blocking_max_ms =
                                        rkb->rkb_rk->
                                        rk_conf.socket_blocking_max_ms;
                        } else
                                rd_rkb_dbg(rkb, QUEUE, "FETCH",
                                           "Fetch backoff for %"PRId64"ms",
                                           (rkb->rkb_ts_fetch_backoff-now)/
                                           1000);
                }

		/* Check and move retry buffers */
		if (unlikely(rd_atomic32_get(&rkb->rkb_retrybufs.rkbq_cnt) > 0))
			rd_kafka_broker_retry_bufs_move(rkb);

		rd_kafka_broker_serve(rkb, RD_POLL_NOWAIT);

		rd_kafka_broker_lock(rkb);
	}

	rd_kafka_broker_unlock(rkb);
}


static int rd_kafka_broker_thread_main (void *arg) {
	rd_kafka_broker_t *rkb = arg;
	rd_kafka_t *rk = rkb->rkb_rk;

        rd_snprintf(rd_kafka_thread_name, sizeof(rd_kafka_thread_name),
		    "%s", rkb->rkb_name);

	(void)rd_atomic32_add(&rd_kafka_thread_cnt_curr, 1);

        /* Our own refcount was increased just prior to thread creation,
         * when refcount drops to 1 it is just us left and the broker 
         * thread should terminate. */

	/* Acquire lock (which was held by thread creator during creation)
	 * to synchronise state. */
	rd_kafka_broker_lock(rkb);
	rd_kafka_broker_unlock(rkb);

	rd_rkb_dbg(rkb, BROKER, "BRKMAIN", "Enter main broker thread");

	while (!rd_kafka_broker_terminating(rkb)) {
                rd_ts_t backoff;

		switch (rkb->rkb_state)
		{
		case RD_KAFKA_BROKER_STATE_INIT:
			/* The INIT state exists so that an initial connection
			 * failure triggers a state transition which might
			 * trigger a ALL_BROKERS_DOWN error. */
		case RD_KAFKA_BROKER_STATE_DOWN:
			if (rkb->rkb_source == RD_KAFKA_INTERNAL) {
                                rd_kafka_broker_lock(rkb);
				rd_kafka_broker_set_state(rkb,
							  RD_KAFKA_BROKER_STATE_UP);
                                rd_kafka_broker_unlock(rkb);
				break;
			}

                        /* Throttle & jitter reconnects to avoid
                         * thundering horde of reconnecting clients after
                         * a broker / network outage. Issue #403 */
                        if (rkb->rkb_rk->rk_conf.reconnect_jitter_ms &&
                            (backoff =
                             rd_interval_immediate(
                                     &rkb->rkb_connect_intvl,
                                     rd_jitter(rkb->rkb_rk->rk_conf.
                                               reconnect_jitter_ms*500,
                                               rkb->rkb_rk->rk_conf.
                                               reconnect_jitter_ms*1500),
                                     0)) <= 0) {
                                backoff = -backoff/1000;
                                rd_rkb_dbg(rkb, BROKER, "RECONNECT",
                                           "Delaying next reconnect by %dms",
                                           (int)backoff);
                                rd_kafka_broker_ua_idle(rkb, (int)backoff);
                                continue;
                        }

			/* Initiate asynchronous connection attempt.
			 * Only the host lookup is blocking here. */
			if (rd_kafka_broker_connect(rkb) == -1) {
				/* Immediate failure, most likely host
				 * resolving failed.
				 * Try the next resolve result until we've
				 * tried them all, in which case we sleep a
				 * short while to avoid busy looping. */
				if (!rkb->rkb_rsal ||
                                    rkb->rkb_rsal->rsal_cnt == 0 ||
                                    rkb->rkb_rsal->rsal_curr + 1 ==
                                    rkb->rkb_rsal->rsal_cnt)
                                        rd_kafka_broker_ua_idle(rkb, 1000);
			}
			break;

		case RD_KAFKA_BROKER_STATE_CONNECT:
		case RD_KAFKA_BROKER_STATE_AUTH:
		case RD_KAFKA_BROKER_STATE_AUTH_HANDSHAKE:
		case RD_KAFKA_BROKER_STATE_APIVERSION_QUERY:
			/* Asynchronous connect in progress. */
			rd_kafka_broker_ua_idle(rkb, 0);

			if (rkb->rkb_state == RD_KAFKA_BROKER_STATE_DOWN) {
				/* Connect failure.
				 * Try the next resolve result until we've
				 * tried them all, in which case we sleep a
				 * short while to avoid busy looping. */
				if (!rkb->rkb_rsal ||
                                    rkb->rkb_rsal->rsal_cnt == 0 ||
                                    rkb->rkb_rsal->rsal_curr + 1 ==
                                    rkb->rkb_rsal->rsal_cnt)
                                        rd_kafka_broker_ua_idle(rkb, 1000);
			}
			break;

                case RD_KAFKA_BROKER_STATE_UPDATE:
                        /* FALLTHRU */
		case RD_KAFKA_BROKER_STATE_UP:
			if (rkb->rkb_nodeid == RD_KAFKA_NODEID_UA)
				rd_kafka_broker_ua_idle(rkb, 0);
			else if (rk->rk_type == RD_KAFKA_PRODUCER)
				rd_kafka_broker_producer_serve(rkb);
			else if (rk->rk_type == RD_KAFKA_CONSUMER)
				rd_kafka_broker_consumer_serve(rkb);

			if (rkb->rkb_state == RD_KAFKA_BROKER_STATE_UPDATE) {
                                rd_kafka_broker_lock(rkb);
				rd_kafka_broker_set_state(rkb, RD_KAFKA_BROKER_STATE_UP);
                                rd_kafka_broker_unlock(rkb);
			} else {
				/* Connection torn down, sleep a short while to
				 * avoid busy-looping on protocol errors */
				rd_usleep(100*1000/*100ms*/, &rk->rk_terminate);
			}
			break;
		}

                if (rd_kafka_terminating(rkb->rkb_rk)) {
                        /* Handle is terminating: fail the send+retry queue
                         * to speed up termination, otherwise we'll
                         * need to wait for request timeouts. */
                        int r;

                        r = rd_kafka_broker_bufq_timeout_scan(
                                rkb, 0, &rkb->rkb_outbufs, NULL,
                                RD_KAFKA_RESP_ERR__DESTROY, 0);
                        r += rd_kafka_broker_bufq_timeout_scan(
                                rkb, 0, &rkb->rkb_retrybufs, NULL,
                                RD_KAFKA_RESP_ERR__DESTROY, 0);
                        rd_rkb_dbg(rkb, BROKER, "TERMINATE",
                                   "Handle is terminating: "
                                   "failed %d request(s) in "
                                   "retry+outbuf", r);
                }
	}

	if (rkb->rkb_source != RD_KAFKA_INTERNAL) {
		rd_kafka_wrlock(rkb->rkb_rk);
		TAILQ_REMOVE(&rkb->rkb_rk->rk_brokers, rkb, rkb_link);
		(void)rd_atomic32_sub(&rkb->rkb_rk->rk_broker_cnt, 1);
		rd_kafka_wrunlock(rkb->rkb_rk);
	}

	rd_kafka_broker_fail(rkb, LOG_DEBUG, RD_KAFKA_RESP_ERR__DESTROY, NULL);
	rd_kafka_broker_destroy(rkb);

	rd_atomic32_sub(&rd_kafka_thread_cnt_curr, 1);

	return 0;
}


/**
 * Final destructor. Refcnt must be 0.
 */
void rd_kafka_broker_destroy_final (rd_kafka_broker_t *rkb) {

        rd_kafka_assert(rkb->rkb_rk, thrd_is_current(rkb->rkb_thread));
        rd_kafka_assert(rkb->rkb_rk, TAILQ_EMPTY(&rkb->rkb_outbufs.rkbq_bufs));
        rd_kafka_assert(rkb->rkb_rk, TAILQ_EMPTY(&rkb->rkb_waitresps.rkbq_bufs));
        rd_kafka_assert(rkb->rkb_rk, TAILQ_EMPTY(&rkb->rkb_retrybufs.rkbq_bufs));
	rd_kafka_assert(rkb->rkb_rk, TAILQ_EMPTY(&rkb->rkb_toppars));

#if WITH_SASL
	if (rkb->rkb_source != RD_KAFKA_INTERNAL &&
            (rkb->rkb_rk->rk_conf.security_protocol ==
             RD_KAFKA_PROTO_SASL_PLAINTEXT ||
             rkb->rkb_rk->rk_conf.security_protocol ==
             RD_KAFKA_PROTO_SASL_SSL))
		rd_kafka_broker_sasl_term(rkb);
#endif

        if (rkb->rkb_wakeup_fd[0] != -1)
                rd_close(rkb->rkb_wakeup_fd[0]);
        if (rkb->rkb_wakeup_fd[1] != -1)
                rd_close(rkb->rkb_wakeup_fd[1]);

	if (rkb->rkb_recv_buf)
		rd_kafka_buf_destroy(rkb->rkb_recv_buf);

	if (rkb->rkb_rsal)
		rd_sockaddr_list_destroy(rkb->rkb_rsal);

	if (rkb->rkb_ApiVersions)
		rd_free(rkb->rkb_ApiVersions);
        rd_free(rkb->rkb_origname);

	rd_kafka_q_purge(rkb->rkb_ops);
	rd_kafka_q_destroy(rkb->rkb_ops);

	    rd_avg_destroy(&rkb->rkb_avg_int_latency);
        rd_avg_destroy(&rkb->rkb_avg_rtt);
	rd_avg_destroy(&rkb->rkb_avg_throttle);

        mtx_lock(&rkb->rkb_logname_lock);
        rd_free(rkb->rkb_logname);
        rkb->rkb_logname = NULL;
        mtx_unlock(&rkb->rkb_logname_lock);
        mtx_destroy(&rkb->rkb_logname_lock);

	mtx_destroy(&rkb->rkb_lock);

        rd_refcnt_destroy(&rkb->rkb_refcnt);

	rd_free(rkb);
}

/**
 * Returns the internal broker with refcnt increased.
 */
rd_kafka_broker_t *rd_kafka_broker_internal (rd_kafka_t *rk) {
	rd_kafka_broker_t *rkb;

        mtx_lock(&rk->rk_internal_rkb_lock);
	rkb = rk->rk_internal_rkb;
	if (rkb)
		rd_kafka_broker_keep(rkb);
        mtx_unlock(&rk->rk_internal_rkb_lock);

	return rkb;
}


/**
 * Adds a broker with refcount set to 1.
 * If 'source' is RD_KAFKA_INTERNAL an internal broker is added
 * that does not actually represent or connect to a real broker, it is used
 * for serving unassigned toppar's op queues.
 *
 * Locks: rd_kafka_wrlock(rk) must be held
 */
rd_kafka_broker_t *rd_kafka_broker_add (rd_kafka_t *rk,
					rd_kafka_confsource_t source,
					rd_kafka_secproto_t proto,
					const char *name, uint16_t port,
					int32_t nodeid) {
	rd_kafka_broker_t *rkb;
#ifndef _MSC_VER
        int r;
        sigset_t newset, oldset;
#endif

	rkb = rd_calloc(1, sizeof(*rkb));

        rd_kafka_mk_nodename(rkb->rkb_nodename, sizeof(rkb->rkb_nodename),
                             name, port);
        rd_kafka_mk_brokername(rkb->rkb_name, sizeof(rkb->rkb_name),
                               proto, rkb->rkb_nodename, nodeid, source);

	rkb->rkb_source = source;
	rkb->rkb_rk = rk;
	rkb->rkb_nodeid = nodeid;
	rkb->rkb_proto = proto;
        rkb->rkb_port = port;
        rkb->rkb_origname = rd_strdup(name);

	mtx_init(&rkb->rkb_lock, mtx_plain);
        mtx_init(&rkb->rkb_logname_lock, mtx_plain);
        rkb->rkb_logname = rd_strdup(rkb->rkb_name);
	TAILQ_INIT(&rkb->rkb_toppars);
        CIRCLEQ_INIT(&rkb->rkb_fetch_toppars);
	rd_kafka_bufq_init(&rkb->rkb_outbufs);
	rd_kafka_bufq_init(&rkb->rkb_waitresps);
	rd_kafka_bufq_init(&rkb->rkb_retrybufs);
	rkb->rkb_ops = rd_kafka_q_new(rk);
        rd_interval_init(&rkb->rkb_connect_intvl);
	rd_avg_init(&rkb->rkb_avg_int_latency, RD_AVG_GAUGE);
	rd_avg_init(&rkb->rkb_avg_rtt, RD_AVG_GAUGE);
	rd_avg_init(&rkb->rkb_avg_throttle, RD_AVG_GAUGE);
        rd_refcnt_init(&rkb->rkb_refcnt, 0);
        rd_kafka_broker_keep(rkb); /* rk_broker's refcount */

        rkb->rkb_blocking_max_ms = rk->rk_conf.socket_blocking_max_ms;

	/* ApiVersion fallback interval */
	if (rkb->rkb_rk->rk_conf.api_version_request) {
		rd_interval_init(&rkb->rkb_ApiVersion_fail_intvl);
		rd_interval_fixed(&rkb->rkb_ApiVersion_fail_intvl,
				  rkb->rkb_rk->rk_conf.api_version_fallback_ms*1000);
	}

	/* Set next intervalled metadata refresh, offset by a random
	 * value to avoid all brokers to be queried simultaneously. */
	if (rkb->rkb_rk->rk_conf.metadata_refresh_interval_ms >= 0)
		rkb->rkb_ts_metadata_poll = rd_clock() +
			(rkb->rkb_rk->rk_conf.
			 metadata_refresh_interval_ms * 1000) +
			(rd_jitter(500,1500) * 1000);
	else /* disabled */
		rkb->rkb_ts_metadata_poll = UINT64_MAX;

#ifndef _MSC_VER
        /* Block all signals in newly created thread.
         * To avoid race condition we block all signals in the calling
         * thread, which the new thread will inherit its sigmask from,
         * and then restore the original sigmask of the calling thread when
         * we're done creating the thread.
	 * NOTE: term_sig remains unblocked since we use it on termination
	 *       to quickly interrupt system calls. */
        sigemptyset(&oldset);
        sigfillset(&newset);
	if (rkb->rkb_rk->rk_conf.term_sig)
		sigdelset(&newset, rkb->rkb_rk->rk_conf.term_sig);
        pthread_sigmask(SIG_SETMASK, &newset, &oldset);
#endif

        /*
         * Fd-based queue wake-ups using a non-blocking pipe.
         * Writes are best effort, if the socket queue is full
         * the write fails (silently) but this has no effect on latency
         * since the POLLIN flag will already have been raised for fd.
         */
        rkb->rkb_wakeup_fd[0]     = -1;
        rkb->rkb_wakeup_fd[1]     = -1;
        rkb->rkb_toppar_wakeup_fd = -1;

#ifndef _MSC_VER /* pipes cant be mixed with WSAPoll on Win32 */
        if ((r = rd_pipe_nonblocking(rkb->rkb_wakeup_fd)) == -1) {
                rd_rkb_log(rkb, LOG_ERR, "WAKEUPFD",
                           "Failed to setup broker queue wake-up fds: "
                           "%s: disabling low-latency mode",
                           rd_strerror(r));

        } else if (source == RD_KAFKA_INTERNAL) {
                /* nop: internal broker has no IO transport. */

        } else {
                char onebyte = 1;

                /* Since there is a small syscall penalty,
                 * only enable partition message queue wake-ups
                 * if latency contract demands it.
                 * rkb_ops queue wakeups are always enabled though,
                 * since they are much more infrequent. */
                if (rk->rk_conf.buffering_max_ms <
                    rk->rk_conf.socket_blocking_max_ms) {
                        rd_rkb_dbg(rkb, QUEUE, "WAKEUPFD",
                                   "Enabled low-latency partition "
                                   "queue wake-ups");
                        rkb->rkb_toppar_wakeup_fd = rkb->rkb_wakeup_fd[1];
                }


                rd_rkb_dbg(rkb, QUEUE, "WAKEUPFD",
                           "Enabled low-latency ops queue wake-ups");
                rd_kafka_q_io_event_enable(rkb->rkb_ops, rkb->rkb_wakeup_fd[1],
                                           &onebyte, sizeof(onebyte));
        }
#endif

        /* Lock broker's lock here to synchronise state, i.e., hold off
	 * the broker thread until we've finalized the rkb. */
	rd_kafka_broker_lock(rkb);
        rd_kafka_broker_keep(rkb); /* broker thread's refcnt */
	if (thrd_create(&rkb->rkb_thread,
			rd_kafka_broker_thread_main, rkb) != thrd_success) {
		char tmp[512];
		rd_snprintf(tmp, sizeof(tmp),
			 "Unable to create broker thread: %s (%i)",
			 rd_strerror(errno), errno);
		rd_kafka_log(rk, LOG_CRIT, "THREAD", "%s", tmp);

		rd_kafka_broker_unlock(rkb);

		/* Send ERR op back to application for processing. */
		rd_kafka_op_err(rk, RD_KAFKA_RESP_ERR__CRIT_SYS_RESOURCE,
				"%s", tmp);

		rd_free(rkb);

#ifndef _MSC_VER
		/* Restore sigmask of caller */
		pthread_sigmask(SIG_SETMASK, &oldset, NULL);
#endif

		return NULL;
	}

	if (rkb->rkb_source != RD_KAFKA_INTERNAL) {
#if WITH_SASL
                if (rk->rk_conf.security_protocol ==
                    RD_KAFKA_PROTO_SASL_PLAINTEXT ||
                    rk->rk_conf.security_protocol == RD_KAFKA_PROTO_SASL_SSL)
                        rd_kafka_broker_sasl_init(rkb);
#endif

		TAILQ_INSERT_TAIL(&rkb->rkb_rk->rk_brokers, rkb, rkb_link);
		(void)rd_atomic32_add(&rkb->rkb_rk->rk_broker_cnt, 1);
		rd_rkb_dbg(rkb, BROKER, "BROKER",
			   "Added new broker with NodeId %"PRId32,
			   rkb->rkb_nodeid);
	}

	rd_kafka_broker_unlock(rkb);

#ifndef _MSC_VER
	/* Restore sigmask of caller */
	pthread_sigmask(SIG_SETMASK, &oldset, NULL);
#endif

	return rkb;
}

/**
 * Locks: rd_kafka_rdlock()
 * NOTE: caller must release rkb reference by rd_kafka_broker_destroy()
 */
rd_kafka_broker_t *rd_kafka_broker_find_by_nodeid0 (rd_kafka_t *rk,
                                                    int32_t nodeid,
                                                    int state) {
	rd_kafka_broker_t *rkb;

	TAILQ_FOREACH(rkb, &rk->rk_brokers, rkb_link) {
		rd_kafka_broker_lock(rkb);
		if (!rd_atomic32_get(&rk->rk_terminate) &&
		    rkb->rkb_nodeid == nodeid) {
                        if (state != -1 && (int)rkb->rkb_state != state) {
                                rd_kafka_broker_unlock(rkb);
                                break;
                        }
			rd_kafka_broker_keep(rkb);
			rd_kafka_broker_unlock(rkb);
			return rkb;
		}
		rd_kafka_broker_unlock(rkb);
	}

	return NULL;

}

/**
 * Locks: rd_kafka_rdlock(rk) must be held
 * NOTE: caller must release rkb reference by rd_kafka_broker_destroy()
 */
static rd_kafka_broker_t *rd_kafka_broker_find (rd_kafka_t *rk,
						rd_kafka_secproto_t proto,
						const char *name,
						uint16_t port) {
	rd_kafka_broker_t *rkb;
	char nodename[RD_KAFKA_NODENAME_SIZE];

        rd_kafka_mk_nodename(nodename, sizeof(nodename), name, port);

	TAILQ_FOREACH(rkb, &rk->rk_brokers, rkb_link) {
		rd_kafka_broker_lock(rkb);
		if (!rd_atomic32_get(&rk->rk_terminate) &&
		    rkb->rkb_proto == proto &&
		    !strcmp(rkb->rkb_nodename, nodename)) {
			rd_kafka_broker_keep(rkb);
			rd_kafka_broker_unlock(rkb);
			return rkb;
		}
		rd_kafka_broker_unlock(rkb);
	}

	return NULL;
}


/**
 * Parse a broker host name.
 * The string 'name' is modified and null-terminated portions of it
 * are returned in 'proto', 'host', and 'port'.
 *
 * Returns 0 on success or -1 on parse error.
 */
static int rd_kafka_broker_name_parse (rd_kafka_t *rk,
				       char **name,
				       rd_kafka_secproto_t *proto,
				       const char **host,
				       uint16_t *port) {
	char *s = *name;
	char *orig;
	char *n, *t, *t2;

	/* Save a temporary copy of the original name for logging purposes */
	rd_strdupa(&orig, *name);

	/* Find end of this name (either by delimiter or end of string */
	if ((n = strchr(s, ',')))
		*n = '\0';
	else
		n = s + strlen(s)-1;


	/* Check if this looks like an url. */
	if ((t = strstr(s, "://"))) {
		int i;
		/* "proto://host[:port]" */

		if (t == s) {
			rd_kafka_log(rk, LOG_WARNING, "BROKER",
				     "Broker name \"%s\" parse error: "
				     "empty protocol name", orig);
			return -1;
		}

		/* Make protocol uppercase */
		for (t2 = s ; t2 < t ; t2++)
			*t2 = toupper(*t2);

		*t = '\0';

		/* Find matching protocol by name. */
		for (i = 0 ; i < RD_KAFKA_PROTO_NUM ; i++)
			if (!rd_strcasecmp(s, rd_kafka_secproto_names[i]))
				break;

		/* Unsupported protocol */
		if (i == RD_KAFKA_PROTO_NUM) {
			rd_kafka_log(rk, LOG_WARNING, "BROKER",
				     "Broker name \"%s\" parse error: "
				     "unsupported protocol \"%s\"", orig, s);

			return -1;
		}

		*proto = i;

                /* Enforce protocol */
		if (rk->rk_conf.security_protocol != *proto) {
			rd_kafka_log(rk, LOG_WARNING, "BROKER",
				     "Broker name \"%s\" parse error: "
				     "protocol \"%s\" does not match "
				     "security.protocol setting \"%s\"",
				     orig, s,
				     rd_kafka_secproto_names[
					     rk->rk_conf.security_protocol]);
			return -1;
		}

		/* Hostname starts here */
		s = t+3;

		/* Ignore anything that looks like the path part of an URL */
		if ((t = strchr(s, '/')))
			*t = '\0';

	} else
		*proto = rk->rk_conf.security_protocol; /* Default protocol */


	*port = RD_KAFKA_PORT;
	/* Check if port has been specified, but try to identify IPv6
	 * addresses first:
	 *  t = last ':' in string
	 *  t2 = first ':' in string
	 *  If t and t2 are equal then only one ":" exists in name
	 *  and thus an IPv4 address with port specified.
	 *  Else if not equal and t is prefixed with "]" then it's an
	 *  IPv6 address with port specified.
	 *  Else no port specified. */
	if ((t = strrchr(s, ':')) &&
	    ((t2 = strchr(s, ':')) == t || *(t-1) == ']')) {
		*t = '\0';
		*port = atoi(t+1);
	}

	/* Empty host name -> localhost */
	if (!*s) 
		s = "localhost";

	*host = s;
	*name = n+1;  /* past this name. e.g., next name/delimiter to parse */

	return 0;
}


/**
 * Adds a (csv list of) broker(s).
 * Returns the number of brokers succesfully added.
 *
 * Locality: any thread
 * Lock prereqs: none
 */
int rd_kafka_brokers_add0 (rd_kafka_t *rk, const char *brokerlist) {
	char *s_copy = rd_strdup(brokerlist);
	char *s = s_copy;
	int cnt = 0;
	rd_kafka_broker_t *rkb;

	/* Parse comma-separated list of brokers. */
	while (*s) {
		uint16_t port;
		const char *host;
		rd_kafka_secproto_t proto;

		if (*s == ',' || *s == ' ') {
			s++;
			continue;
		}

		if (rd_kafka_broker_name_parse(rk, &s, &proto,
					       &host, &port) == -1)
			break;

		rd_kafka_wrlock(rk);

		if ((rkb = rd_kafka_broker_find(rk, proto, host, port)) &&
		    rkb->rkb_source == RD_KAFKA_CONFIGURED) {
			cnt++;
		} else if (rd_kafka_broker_add(rk, RD_KAFKA_CONFIGURED,
					       proto, host, port,
					       RD_KAFKA_NODEID_UA) != NULL)
			cnt++;

		/* If rd_kafka_broker_find returned a broker its
		 * reference needs to be released 
		 * See issue #193 */
		if (rkb)
			rd_kafka_broker_destroy(rkb);

		rd_kafka_wrunlock(rk);
	}

	rd_free(s_copy);

	return cnt;
}


int rd_kafka_brokers_add (rd_kafka_t *rk, const char *brokerlist) {
        return rd_kafka_brokers_add0(rk, brokerlist);
}


/**
 * Adds a new broker or updates an existing one.
 *
 */
void rd_kafka_broker_update (rd_kafka_t *rk, rd_kafka_secproto_t proto,
                             const struct rd_kafka_metadata_broker *mdb) {
	rd_kafka_broker_t *rkb;
        char nodename[RD_KAFKA_NODENAME_SIZE];
        int needs_update = 0;

        rd_kafka_mk_nodename(nodename, sizeof(nodename), mdb->host, mdb->port);

	rd_kafka_wrlock(rk);
	if (unlikely(rd_atomic32_get(&rk->rk_terminate))) {
		/* Dont update metadata while terminating, do this
		 * after acquiring lock for proper synchronisation */
		rd_kafka_wrunlock(rk);
		return;
	}

	if ((rkb = rd_kafka_broker_find_by_nodeid(rk, mdb->id))) {
                /* Broker matched by nodeid, see if we need to update
                 * the hostname. */
                if (strcmp(rkb->rkb_nodename, nodename))
                        needs_update = 1;
        } else if ((rkb = rd_kafka_broker_find(rk, proto,
					       mdb->host, mdb->port))) {
                /* Broker matched by hostname (but not by nodeid),
                 * update the nodeid. */
                needs_update = 1;

        } else {
		rd_kafka_broker_add(rk, RD_KAFKA_LEARNED,
				    proto, mdb->host, mdb->port, mdb->id);
	}

	rd_kafka_wrunlock(rk);

        if (rkb) {
                /* Existing broker */
                if (needs_update) {
                        rd_kafka_op_t *rko;

                        rko = rd_kafka_op_new(RD_KAFKA_OP_NODE_UPDATE);
                        strncpy(rko->rko_u.node.nodename, nodename,
				sizeof(rko->rko_u.node.nodename)-1);
                        rko->rko_u.node.nodeid   = mdb->id;
                        rd_kafka_q_enq(rkb->rkb_ops, rko);
                }
                rd_kafka_broker_destroy(rkb);
        }
}


/**
 * Returns a thread-safe temporary copy of the broker name.
 * Must not be called more than 4 times from the same expression.
 *
 * Locks: none
 * Locality: any thread
 */
const char *rd_kafka_broker_name (rd_kafka_broker_t *rkb) {
        static RD_TLS char ret[4][RD_KAFKA_NODENAME_SIZE];
        static RD_TLS int reti = 0;

        reti = (reti + 1) % 4;
        mtx_lock(&rkb->rkb_logname_lock);
        rd_snprintf(ret[reti], sizeof(ret[reti]), "%s", rkb->rkb_logname);
        mtx_unlock(&rkb->rkb_logname_lock);

        return ret[reti];
}


void rd_kafka_brokers_init (void) {
}








