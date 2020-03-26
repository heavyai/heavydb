class XmlToolingC < Formula
  desc "Provides a higher level interface to XML processing"
  homepage "https://wiki.shibboleth.net/confluence/display/OpenSAML/XMLTooling-C"
  url "https://dependencies.mapd.com/thirdparty/xmltooling-3.0.4-nolog4shib.tar.gz"
  revision 1

  depends_on "pkg-config" => :build
  depends_on "boost"
  depends_on "log4shib"
  depends_on "openssl@1.1"
  depends_on "xerces-c"
  depends_on "xml-security-c"

  resource "curl" do
    url "https://curl.haxx.se/download/curl-7.65.1.tar.bz2"
    mirror "https://curl.askapache.com/download/curl-7.65.1.tar.bz2"
    sha256 "cbd36df60c49e461011b4f3064cff1184bdc9969a55e9608bf5cadec4686e3f7"
  end

  def install
    ENV.cxx11

    ENV.prepend_path "PKG_CONFIG_PATH", "#{Formula["openssl@1.1"].opt_lib}/pkgconfig"

    resource("curl").stage do
      system "./configure", "--disable-debug",
                            "--disable-dependency-tracking",
                            "--disable-silent-rules",
                            "--prefix=#{libexec}/curl",
                            "--with-ssl=#{Formula["openssl@1.1"].opt_prefix}",
                            "--with-ca-bundle=#{etc}/openssl/cert.pem",
                            "--with-ca-path=#{etc}/openssl/certs",
                            "--without-libssh2",
                            "--without-libmetalink",
                            "--without-gssapi",
                            "--without-librtmp",
                            "--disable-ares",
                            "--disable-shared",
                            "--enable-static"
      system "make", "install"
    end

    ENV.prepend_path "PKG_CONFIG_PATH", libexec/"curl/lib/pkgconfig"

    system "./configure", "--disable-debug",
                          "--disable-dependency-tracking",
                          "--prefix=#{prefix}",
                          "--disable-shared",
                          "--enable-static"
    system "make", "install"
  end
end
