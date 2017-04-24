/*
 * librdkafka - The Apache Kafka C/C++ library
 *
 * Copyright (c) 2015 Magnus Edenhill
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

#include "rdkafka_int.h"
#include "rdkafka_transport.h"
#include "rdkafka_transport_int.h"
#include "rdkafka_sasl.h"
#include "rdkafka_sasl_int.h"


 /**
 * Send auth message with framing.
 * This is a blocking call.
 */
int rd_kafka_sasl_send (rd_kafka_transport_t *rktrans,
                        const void *payload, int len,
                        char *errstr, int errstr_size) {
	struct msghdr msg = RD_ZERO_INIT;
	struct iovec iov[1];
	int32_t hdr;
	char *frame;
	int total_len = sizeof(hdr) + len;

	rd_rkb_dbg(rktrans->rktrans_rkb, SECURITY, "SASL",
		   "Send SASL frame to broker (%d bytes)", len);

	frame = rd_malloc(total_len);

	hdr = htobe32(len);
	memcpy(frame, &hdr, sizeof(hdr));
	if (payload)
		memcpy(frame+sizeof(hdr), payload, len);

	msg.msg_iov = iov;
	msg.msg_iovlen = 1;
	iov[0].iov_base = frame;
	iov[0].iov_len  = total_len;

	/* Simulate blocking behaviour on non-blocking socket..
	 * FIXME: This isn't optimal but is highly unlikely to stall since
	 *        the socket buffer will most likely not be exceeded. */
	do {
		int r;

		r = (int)rd_kafka_transport_sendmsg(rktrans, &msg,
                                                    errstr, errstr_size);
		if (r == -1) {
			rd_rkb_dbg(rktrans->rktrans_rkb, SECURITY, "SASL",
				   "SASL send of %d bytes failed: %s",
				   total_len, errstr);
			free(frame);
			return -1;
		}

		if (r == total_len)
			break;

		/* Avoid busy-looping */
		rd_usleep(10*1000, NULL);

		/* Shave off written bytes */
		total_len -= r;
		iov[0].iov_base = ((char *)(iov[0].iov_base)) + r;
		iov[0].iov_len  = total_len;

	} while (total_len > 0);

	free(frame);

	return 0;
}


/**
 * @brief Authentication succesful
 *
 * Transition to next connect state.
 */
void rd_kafka_sasl_auth_done (rd_kafka_transport_t *rktrans) {
        /* Authenticated */
        rd_kafka_broker_connect_up(rktrans->rktrans_rkb);
}


int rd_kafka_sasl_io_event (rd_kafka_transport_t *rktrans, int events,
			    char *errstr, int errstr_size) {
        rd_kafka_buf_t *rkbuf;
        int r;

        if (!(events & POLLIN))
                return 0;

        r = rd_kafka_transport_framed_recvmsg(rktrans, &rkbuf,
                                              errstr, errstr_size);
        if (r == -1)
                return -1;
        else if (r == 0) /* not fully received yet */
                return 0;

        rd_rkb_dbg(rktrans->rktrans_rkb, SECURITY, "SASL",
                   "Received SASL frame from broker (%"PRIusz" bytes)",
                   rkbuf ? rkbuf->rkbuf_len : 0);

        return rktrans->rktrans_sasl.recv(rktrans,
                                          rkbuf ?
                                          rkbuf->rkbuf_rbuf : NULL,
                                          rkbuf ?
                                          rkbuf->rkbuf_len : 0,
                                          errstr, errstr_size);
}



/**
 * Initialize and start SASL authentication.
 *
 * Returns 0 on successful init and -1 on error.
 *
 * Locality: broker thread
 */
int rd_kafka_sasl_client_new (rd_kafka_transport_t *rktrans,
			      char *errstr, int errstr_size) {
	int r;
	rd_kafka_broker_t *rkb = rktrans->rktrans_rkb;
	rd_kafka_t *rk = rkb->rkb_rk;
        char *hostname, *t;

        /* Verify broker support:
         * - RD_KAFKA_FEATURE_SASL_GSSAPI - GSSAPI supported
         * - RD_KAFKA_FEATURE_SASL_HANDSHAKE - GSSAPI, PLAIN and possibly
         *   other mechanisms supported. */
        if (!strcmp(rk->rk_conf.sasl.mechanisms, "GSSAPI")) {
                if (!(rkb->rkb_features & RD_KAFKA_FEATURE_SASL_GSSAPI)) {
                        rd_snprintf(errstr, errstr_size,
                                    "SASL GSSAPI authentication not supported "
                                    "by broker");
                        return -1;
                }
        } else if (!(rkb->rkb_features & RD_KAFKA_FEATURE_SASL_HANDSHAKE)) {
                rd_snprintf(errstr, errstr_size,
                            "SASL Handshake not supported by broker "
                            "(required by mechanism %s)%s",
                            rk->rk_conf.sasl.mechanisms,
                            rk->rk_conf.api_version_request ? "" :
                            ": try api.version.request=true");
                return -1;
        }

        rd_strdupa(&hostname, rktrans->rktrans_rkb->rkb_nodename);
        if ((t = strchr(hostname, ':')))
                *t = '\0';  /* remove ":port" */

        rd_rkb_dbg(rkb, SECURITY, "SASL",
                   "Initializing SASL client: service name %s, "
                   "hostname %s, mechanisms %s",
                   rk->rk_conf.sasl.service_name, hostname,
                   rk->rk_conf.sasl.mechanisms);

#ifndef _MSC_VER
        r = rd_kafka_sasl_cyrus_client_new(rktrans, hostname,
                                           errstr, errstr_size);
#else
        r = rd_kafka_sasl_win32_client_new(rktrans, hostname,
                                           errstr, errstr_size);
#endif

        if (r != -1)
                rd_kafka_transport_poll_set(rktrans, POLLIN);

        return r;
}







/**
 * Per handle SASL term.
 *
 * Locality: broker thread
 */
void rd_kafka_broker_sasl_term (rd_kafka_broker_t *rkb) {
#ifndef _MSC_VER
        rd_kafka_broker_sasl_cyrus_term(rkb);
#endif
}

/**
 * Broker SASL init.
 *
 * Locality: broker thread
 */
void rd_kafka_broker_sasl_init (rd_kafka_broker_t *rkb) {
#ifndef _MSC_VER
        rd_kafka_broker_sasl_cyrus_init(rkb);
#endif
}



int rd_kafka_sasl_conf_validate (rd_kafka_t *rk,
				 char *errstr, size_t errstr_size) {

	if (strcmp(rk->rk_conf.sasl.mechanisms, "GSSAPI"))
		return 0;

#ifndef _MSC_VER
        return rd_kafka_sasl_cyrus_conf_validate(rk, errstr, errstr_size);
#else
        return 0;
#endif
}


/**
 * Global SASL termination.
 */
void rd_kafka_sasl_global_term (void) {
#ifndef _MSC_VER
        rd_kafka_sasl_cyrus_global_term();
#endif
}


/**
 * Global SASL init, called once per runtime.
 */
int rd_kafka_sasl_global_init (void) {
#ifndef _MSC_VER
        return rd_kafka_sasl_cyrus_global_init();
#else
        return 0;
#endif
}

