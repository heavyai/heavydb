class ApacheArrowOmnisci < Formula
  desc "Columnar in-memory analytics layer designed to accelerate big data"
  homepage "https://arrow.apache.org/"
  url "https://github.com/apache/arrow/archive/apache-arrow-0.16.0.tar.gz"
  head "https://github.com/apache/arrow.git"
  sha256 "d7b3838758a365c8c47d55ab0df1006a70db951c6964440ba354f81f518b8d8d"

  depends_on "cmake" => :build
  depends_on "boost"

  def install
    ENV.cxx11
    args = [
      "-DCMAKE_BUILD_TYPE=Release",
      "-DCMAKE_INSTALL_PREFIX=#{prefix}",
      "-DARROW_BUILD_SHARED=ON",
      "-DARROW_BUILD_STATIC=ON",
      "-DARROW_BUILD_TESTS=OFF",
      "-DARROW_BUILD_BENCHMARKS=OFF",
      "-DARROW_WITH_BROTLI=OFF",
      "-DARROW_WITH_ZLIB=ON",
      "-DARROW_USE_GLOG=OFF",
      "-DARROW_GFLAGS_USE_SHARED=OFF",
      "-DARROW_WITH_LZ4=OFF",
      "-DARROW_WITH_SNAPPY=ON",
      "-DARROW_WITH_ZSTD=ON",
      "-DARROW_CSV=ON",
      "-DARROW_JSON=ON",
      "-DARROW_BOOST_USE_SHARED=ON",
      "-DARROW_PARQUET=ON",
      "-DARROW_JEMALLOC=OFF",
      "-DTHRIFT_HOME=#{Formula["thrift"].opt_prefix}"
    ]

    cd "cpp" do
      system "cmake", ".", *std_cmake_args, *args
      system "make"
      system "make", "install"
    end
  end

  test do
    (testpath/"test.cpp").write <<~EOS
      #include "arrow/api.h"
      int main(void)
      {
        arrow::int64();
        return 0;
      }
    EOS
    system ENV.cxx, "test.cpp", "-std=c++11", "-I#{include}", "-L#{lib}", "-larrow", "-o", "test"
    system "./test"
  end
end
