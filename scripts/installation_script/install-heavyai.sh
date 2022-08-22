#! /bin/bash
LOGFILE=${LOGFILE:=heavy_install.log}
exec 3>&1 4>&2

#set -o history -o histexpand
VERBOSE=$1
if [[ $VERBOSE = "" ]]; then
  exec 1>>$LOGFILE 2>&1
fi
set -x
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
source "${SCRIPT_DIR}"/functions.sh
###############################################################################
#                                                                             #
#                              GLOBAL                                         #
#                                                                             #
###############################################################################

# Currently support operating system versions
# These two lists are accessed indirectly using the ${!var} pattern
# as in:
#     supported_versions="SUPPORTED_VERSIONS_${linux_distro}"
#     ${!supported_versions}
SUPPORTED_VERSIONS_Ubuntu=18.04,20.04
SUPPORTED_VERSIONS_CentOS=7,8

# Can be overridden through environment variables
HEAVYAI_USER=${HEAVYAI_USER:=heavyai}
HEAVYAI_GROUP=${HEAVYAI_GROUP:=heavyai}
HEAVYAI_STORAGE=${HEAVYAI_STORAGE:=/var/lib/heavyai}
HEAVYAI_PATH=${HEAVYAI_PATH:=/opt/heavyai}
HEAVYAI_LOG=${HEAVYAI_LOG:=/var/lib/heavyai/data/mapd_log}
HEAVYAI_INSTALL=${HEAVYAI_INSTALL:=/opt/heavyai-installs}

# Directory of all the temp/state files are located.
export TMPDIR=$(pwd)/tmpInstall

echo "
###############################################################################
#                                                                             #
#                      HEAVYAI Easy Installation Script                       #
#                                                                             #
###############################################################################

These instructions assume the following:
* You are installing on a clean Linux host machine with only the operating system installed.
* Your HEAVY.AI host only runs the daemons and services required to support HEAVY.AI.
* Your HEAVY.AI host is connected to the internet.
*
* Installing on: "$(date)" " >&3

printf "
\nTHE SCRIPT IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.\n\n" >&3

  echo "BEFORE  IT"
if  [ ! -d "$TMPDIR" ]; then
  echo "DONING IT"
  mkdir "$TMPDIR"
fi

  echo "AFTER  IT"
if  [ !  -f "$TMPDIR"/READY_TO_INSTALL ]; then
    set_installation_environment
    install_base_packages_$OS_TYPE
    [ -f "$TMPDIR"/FIREWALL ] && enable_firewall
    install_nvidia_drivers
    initialize_systemd || exit_on_error $? !:0
    [ -f "$TMPDIR"/OS_INSTALL ] && rm "$TMPDIR"/OS_INSTALL

    touch "$TMPDIR"/READY_TO_INSTALL
    restart_host
else
echo "
This stage includes loading of your license
and activation of HEAVYAI software. " >&3
    activation
    verify_installation
    import_sample_data_and_query
    rm -r "$TMPDIR"
echo "
###############################################################################
#                                                                             #
#                  Installation and Activation Completed                      #
#                                                                             #
###############################################################################

Installation and verification complete. 

Return to the installation documentation and follow the instructions create a
new dashboard and chart." >&3

    exit 0
fi
