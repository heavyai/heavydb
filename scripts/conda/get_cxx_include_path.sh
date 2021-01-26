#
# Usage:
#
#   source this file and call ${get_cxx_include_path} to get a full
#   include path of ${CXX}
#
# Author: Pearu Peterson
# Created: November 2019
#
function get_cxx_include_path() {
    local cplus_include_path=""
    local start_include_search=false
    local sep=""
    (${CXX} -E -x c++ - -v < /dev/null) 2>&1 |
        while read -r LINE
        do
            case "$LINE" in
                '#include <...> search starts here:' | '#include "..." search starts here:')
                    start_include_search=true
                    ;;
                'End of search list.')
                    echo "$cplus_include_path"
                    break
                    ;;
                *)
                    if [ "$start_include_search" = "true" ];
                    then
                        cplus_include_path="$cplus_include_path${sep}$(readlink -f ${LINE})"
                        sep=":"
                    fi
                    ;;
            esac
        done
}
