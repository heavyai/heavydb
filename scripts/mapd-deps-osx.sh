#!/bin/bash

set -e
set -x

PREFIX=/usr/local
SCRIPTS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $SCRIPTS_DIR/common-functions.sh

# install homebrew
if ! hash brew &> /dev/null; then
  ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
else
  brew update
  brew upgrade
fi

# install deps
brew install cmake
brew install folly
brew install gflags
brew install glog
brew install wget
brew install jq
brew install c-blosc

#brew install thrift
# custom thrift formula pinned to specific supported version
brew install -s $SCRIPTS_DIR/../ThirdParty/Thrift/thrift.rb
brew switch thrift 0.11.0

brew install cryptopp
brew install llvm@6

#install_arrow
brew install snappy
brew install -s ../ThirdParty/Arrow/apache-arrow.rb
brew switch apache-arrow 0.11.1

brew install golang
brew install libpng
brew install libarchive
brew install opensaml

brew cask install java
brew install gdal
brew install maven

# compile and install bison++ (default location under /usr/local is fine)
curl -O https://storage.googleapis.com/google-code-archive-downloads/v2/code.google.com/flexpp-bisonpp/bisonpp-1.21-45.tar.gz
tar xvf bisonpp-1.21-45.tar.gz
pushd bison++-1.21
./configure && make && sudo make install
popd

# install AWS core and s3 sdk
# remove -j $(proc) to avoid "clang: error: unable to execute command: posix_spawn failed: Resource temporarily unavailable""
install_awscpp

# install CUDA
brew tap caskroom/drivers
brew cask install nvidia-cuda
CUDA_ROOT=$(ls -d /Developer/NVIDIA/CUDA-* | tail -n 1)
export PATH=$CUDA_ROOT/bin/:$PATH

# Finally, add a few components of llvm to your path PATH.
# Not adding full llvm/bin to PATH since brew's `clang` breaks CUDA
mkdir -p ~/bin/
for i in llvm-config; do
  ln -sf "$(brew --prefix llvm@6)/bin/$i" ~/bin/$i
done
export PATH=~/bin:$PATH

cat >> ~/.bash_profile <<EOF
#mapd-deps cuda
CUDA_ROOT=\$(ls -d /Developer/NVIDIA/CUDA-* | tail -n 1)
DYLD_LIBRARY_PATH=\$CUDA_ROOT/lib:\$DYLD_LIBRARY_PATH
PATH=\$CUDA_ROOT/bin:\$PATH
PATH=\$HOME/bin:\$PATH
export DYLD_LIBRARY_PATH PATH
EOF

source ~/.bash_profile

#done!
#git clone mapd2 && cd mapd2 && mkdir build && cd build && ccmake ..
