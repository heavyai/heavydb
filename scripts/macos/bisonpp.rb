class Bisonpp < Formula
  desc ""
  homepage ""
  url "https://dependencies.mapd.com/thirdparty/bisonpp-1.21-45.tar.gz"
  sha256 "a6d2b04aa22c91c8b09cb2205c0a4148e289ea8e4127d8f05fb2e4c5b45ed791"

  def install
    system "./configure", "--prefix=#{prefix}"
    system "mkdir", "-p", "#{prefix}/lib"
    system "make", "install"
  end

  test do
    system "#{bin}/bison++", "--version"
  end
end

