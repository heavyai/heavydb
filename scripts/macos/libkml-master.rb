class LibkmlMaster < Formula
  desc "Library to parse, generate and operate on KML"
  homepage "https://github.com/libkml/libkml"
  url "https://dependencies.mapd.com/thirdparty/libkml-libkml-master.zip"

  depends_on "cmake" => :build
  depends_on "expat"
  depends_on "minizip"

  conflicts_with "libkml", :because => "libkml from homebrew is majorly out of date"

  def install
    system "mkdir", "build"

    cd "build" do
      system "cmake", "..", "-DCMAKE_BUILD_TYPE=Release", "-DBUILD_SHARED_LIBS=off", *std_cmake_args
      system "make"
      system "make", "install"
    end
  end
end
