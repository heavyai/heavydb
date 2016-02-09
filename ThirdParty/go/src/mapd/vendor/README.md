Vendored dependencies managed by
[govendor](https://github.com/kardianos/govendor) and `GO15VENDOREXPERIMENT`.

Note: the CMake-managed build does not use `GO15VENDOREXPERIMENT` or the
`ThirdParty/go/src/mapd/vendor/` directory as-is because Go 1.5 is not
available on all target build platforms. Instead, CMake copies
`ThirdParty/go/src/mapd/vendor/` to `${CMAKE_BUILD_DIR}/go/src` and uses
`GOPATH=${CMAKE_BUILD_DIR}/go`.

To manage dependencies:

Set `$GOPATH`, add to `$PATH`, install `govendor`:

    export GOPATH=/path/to/map-d/mapd2/ThirdParty/go
    export PATH=$GOPATH/bin:$PATH
    go get github.com/kardianos/govendor

Enable `GO15VENDOREXPERIMENT` (this is enabled by default in Go 1.6):

    export GO15VENDOREXPERIMENT=1

To add a new dependency:

    cd $GOPATH/src/mapd/
    go get url/to/dep
    govendor add url/to/dep

To update a dependency:

    cd $GOPATH/src/mapd/
    go get -u url/to/dep
    govendor update url/to/dep

To add dependencies by hand, copy the directory to `$GOPATH/src/mapd/vendor`
and remove the `.git` directory and all files related to unused build flags.

Only commit files under the `$GOPATH/src/mapd/vendor`. Do not commit anything under
`$GOPATH/bin` or `$GOPATH/pkg`.
