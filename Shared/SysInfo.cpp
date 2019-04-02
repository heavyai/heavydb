#include "SysInfo.h"

#include <unistd.h>
#include <climits>

std::string get_hostname() {
  char hostname[_POSIX_HOST_NAME_MAX];

  gethostname(hostname, _POSIX_HOST_NAME_MAX);

  return {hostname};
}
