/* IP-related utils. */

#ifndef UV_IP_H_
#define UV_IP_H_

#include <netinet/in.h>

/* Split @address into @host and @port and populate @addr accordingly. */
int uvIpParse(const char *address, struct sockaddr_in *addr);

#endif /* UV_IP_H */
