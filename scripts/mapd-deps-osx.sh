#!/bin/bash

set -e
set -x

# install homebrew
if ! hash brew &> /dev/null; then
  ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
else
  brew update
  brew upgrade
fi

# install deps
brew install cmake
brew install gflags
brew install glog
brew install thrift
brew install cryptopp
brew install llvm
brew install folly

# install CUDA (even if you don't have an nvidia GPU - some headers req'd for compilation)
brew tap caskroom/drivers
brew cask install cuda
CUDA_ROOT=$(ls -d /Developer/NVIDIA/CUDA-* | tail -n 1)
export PATH=$CUDA_ROOT/bin/:$PATH

# compile and install bison++ (default location under /usr/local is fine)
curl -O https://storage.googleapis.com/google-code-archive-downloads/v2/code.google.com/flexpp-bisonpp/bisonpp-1.21-45.tar.gz
tar xvf bisonpp-1.21-45.tar.gz
cd bison++-1.21
./configure && make && make install

# Finally, add a few components of llvm to your path PATH.
# Not adding full llvm/bin to PATH since brew's `clang` breaks CUDA
mkdir -p ~/bin/
for i in clang++ llc llvm-config clang-format; do
  ln -sf "$(brew --prefix llvm)/bin/$i" ~/bin/$i
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

brew install nodejs
brew install golang
brew install glfw3
brew install glew

brew install gdal --with-libkml

brew cask install java
cat >> ~/.bash_profile <<EOF
# mapd-deps java
DYLD_LIBRARY_PATH=$(/usr/libexec/java_home)/jre/lib/server:$DYLD_LIBRARY_PATH
JAVA_HOME=$(/usr/libexec/java_home)
export DYLD_LIBRARY_PATH JAVA_HOME
EOF
source ~/.bash_profile
brew install maven

#done!
#git clone mapd2 && cd mapd2 && mkdir build && cd build && ccmake ..
