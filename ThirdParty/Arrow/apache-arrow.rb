class ApacheArrow < Formula
  desc "Columnar in-memory analytics layer designed to accelerate big data"
  homepage "https://arrow.apache.org/"
  url "https://github.com/apache/arrow/archive/apache-arrow-0.11.1.tar.gz"
  head "https://github.com/apache/arrow.git"
  sha256 "3219c4e87e7cf979017f0cc5bc5dd6a3611d0fc750e821911fab998599dc125b"

  depends_on "cmake" => :build
  depends_on "boost"
  depends_on "jemalloc"
  depends_on "python" => :optional
  depends_on "python@2" => :optional

  needs :cxx11

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
      "-DARROW_WITH_ZLIB=OFF",
      "-DARROW_WITH_LZ4=OFF",
      "-DARROW_WITH_SNAPPY=ON",
      "-DARROW_WITH_ZSTD=OFF",
      "-DARROW_BOOST_USE_SHARED=ON",
      "-DARROW_PARQUET=ON",
      "-DTHRIFT_HOME=#{Formula["thrift"].opt_prefix}"
    ]

    if build.with?("python") && build.with?("python@2")
      odie "Cannot provide both --with-python and --with-python@2"
    end
    Language::Python.each_python(build) do |python, _version|
      args << "-DARROW_PYTHON=1" << "-DPYTHON_EXECUTABLE=#{which python}"
    end

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
