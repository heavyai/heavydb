class Minizip < Formula
  desc "C library for zip/unzip via zLib"
  homepage "https://www.winimage.com/zLibDll/minizip.html"
  url "https://zlib.net/zlib-1.2.11.tar.gz"
  sha256 "c3e5e9fdd5004dcb542feda5ee4f0ff0744628baf8ed2dd5d66f8ca1197cb1a1"

  depends_on "autoconf" => :build
  depends_on "automake" => :build
  depends_on "libtool" => :build

  conflicts_with "minizip2",
    :because => "both install a `libminizip.a` library"

  def install
    system "./configure", "--prefix=#{prefix}"
    system "make"

    cd "contrib/minizip" do
      # edits to statically link to libz.a
      #inreplace "Makefile.am" do |s|
      #  s.sub! "-L$(zlib_top_builddir)", "$(zlib_top_builddir)/libz.a"
      #  s.sub! "-version-info 1:0:0 -lz", "-version-info 1:0:0"
      #  s.sub! "libminizip.la -lz", "libminizip.la"
      #end
      system "autoreconf", "-fi"
      system "./configure", "--prefix=#{prefix}", "--enable-static", "--disable-shared"
      system "make", "install"
    end
  end

  def caveats
    <<~EOS
      Minizip headers installed in 'minizip' subdirectory, since they conflict
      with the venerable 'unzip' library.
    EOS
  end
end
