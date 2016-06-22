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

# install CUDA (even if you don't have an nvidia GPU - some headers req'd for compilation)
brew tap Caskroom/cask
brew install Caskroom/cask/cuda
export PATH=/Developer/NVIDIA/CUDA-7.5/bin/:$PATH

# compile and install bison++ (default location under /usr/local is fine)
curl -O https://flexpp-bisonpp.googlecode.com/files/bisonpp-1.21-45.tar.gz
tar xvf bisonpp-1.21-45.tar.gz
cd bison++-1.21
./configure && make && make install

# LLVM 3.5
# We currently require LLVM 3.5 as we use the older JIT api (replaced by MCJIT
# in 3.6). This version is provided by Homebrew's `versions` tap.

brew tap homebrew/versions

cat << EOS

llvm35 requires a slight modification to the build config. See the README for details.

Add the following to the args, somewhere around lines 196-203:
   "--with-c-include-dirs=/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX10.11.sdk/usr/include",

Hit enter to edit the config.

EOS
read -r

brew edit llvm35
brew install llvm35 --with-all-targets

# Finally, add a few components of llvm to your path PATH.
mkdir -p ~/bin/
for i in clang++ llc llvm-config; do
  ln -sf "$(brew --prefix llvm35)/bin/$i-3.5" ~/bin/$i
done
export PATH=~/bin:$PATH

cat >> ~/.bash_profile <<EOF
DYLD_LIBRARY_PATH=/Developer/NVIDIA/CUDA-7.5/lib:\$DYLD_LIBRARY_PATH
PATH=/Developer/NVIDIA/CUDA-7.5/bin:\$PATH
PATH=\$HOME/bin:\$PATH
export DYLD_LIBRARY_PATH PATH
EOF

brew install nodejs
brew install golang
brew install glfw3
brew install glew

#done!
#git clone mapd2 && cd mapd2 && mkdir build && cd build && ccmake ..
