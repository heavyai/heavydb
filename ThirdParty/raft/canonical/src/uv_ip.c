#include <stdlib.h>
#include <string.h>

#include <uv.h>

#include "../include/raft.h"

#include "uv_ip.h"

int uvIpParse(const char *address, struct sockaddr_in *addr)
{
    char buf[256];
    char *host;
    char *port;
    char *colon = ":";
    int rv;

    /* TODO: turn this poor man parsing into proper one */
    strcpy(buf, address);
    host = strtok(buf, colon);
    port = strtok(NULL, ":");
    if (port == NULL) {
        port = "8080";
    }

    rv = uv_ip4_addr(host, atoi(port), addr);
    if (rv != 0) {
        return RAFT_NOCONNECTION;
    }

    return 0;
}
