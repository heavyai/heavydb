#!/bin/bash

set -e
set -x

SCRIPTS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# install homebrew
if ! hash brew &> /dev/null; then
  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install.sh)"
else
  brew update
  brew upgrade || true
fi

## install deps
brew install cmake
brew install wget
brew install jq
brew install c-blosc
brew install golang
brew install libpng
brew install libarchive
brew install maven
brew install ninja
brew install snappy
brew install double-conversion
brew install llvm

function build_pin_dep {
  brew unpin $1 || true
  brew unlink $1 || true
  brew install -s --debug $SCRIPTS_DIR/macos/$1.rb
  brew switch $1 $2
  brew pin $1
}

## # Disabled due to build issues on macOS
## build_pin_dep aws-sdk-cpp 1.7.230

build_pin_dep boost 1.72.0
build_pin_dep gflags 2.2.2
build_pin_dep glog 0.4.0
build_pin_dep thrift 0.11.0
build_pin_dep aws-sdk-cpp 1.7.280
build_pin_dep apache-arrow-omnisci 0.16.0
build_pin_dep bisonpp 1.21-45
build_pin_dep librdkafka 1.2.2
build_pin_dep xerces-c 3.2.2
build_pin_dep xml-tooling-c 3.0.4_1
build_pin_dep opensaml 3.0.1_1
build_pin_dep uriparser 0.9.3
build_pin_dep expat 2.2.9
build_pin_dep minizip 1.2.11
build_pin_dep libkml-master 1.4.0
build_pin_dep proj5 5.2.0
build_pin_dep gdal 2.4.4
build_pin_dep geos 3.8.1

# Finally, add a few components of llvm to your path PATH.
# Not adding full llvm/bin to PATH since brew's `clang` breaks CUDA
mkdir -p ~/bin/
for i in llvm-config; do
  ln -sf "$(brew --prefix llvm)/bin/$i" ~/bin/$i
done
export PATH=~/bin:$PATH

cat >> ~/.bash_profile <<EOF
#mapd-deps
PATH=\$HOME/bin:\$PATH
DYLD_LIBRARY_PATH=/usr/local/opt/openssl/lib:\$DYLD_LIBRARY_PATH
export PATH DYLD_LIBRARY_PATH
EOF

source ~/.bash_profile

cat >> ~/.zlogin <<EOF
#mapd-deps
PATH=\$HOME/bin:\$PATH
DYLD_LIBRARY_PATH=/usr/local/opt/openssl/lib:\$DYLD_LIBRARY_PATH
export PATH DYLD_LIBRARY_PATH
EOF

source ~/.zlogin

#done!
#git clone omniscidb && cd omniscidb && mkdir build && cd build && ccmake ..
