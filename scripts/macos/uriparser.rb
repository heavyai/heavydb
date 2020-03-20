class Uriparser < Formula
  desc "URI parsing library (strictly RFC 3986 compliant)"
  homepage "https://uriparser.github.io/"
  head "https://github.com/uriparser/uriparser.git"

  stable do
    url "https://github.com/uriparser/uriparser/releases/download/uriparser-0.9.3/uriparser-0.9.3.tar.bz2"
    sha256 "28af4adb05e811192ab5f04566bebc5ebf1c30d9ec19138f944963d52419e28f"

    # Upstream fix, will be integrated in next release
    # https://github.com/uriparser/uriparser/issues/67
    patch do
      url "https://github.com/uriparser/uriparser/commit/f870e6c68696a6018702caa5c8a2feba9b0f99fa.diff?full_index=1"
      sha256 "c609224fc996b6231781e1beba4424c2237fc5e49e2de049b344d926db0630f7"
    end
  end

  depends_on "cmake" => :build

  conflicts_with "libkml", :because => "both install `liburiparser.dylib`"

  def install
    system "cmake", ".", "-DURIPARSER_BUILD_TESTS=OFF", "-DURIPARSER_BUILD_DOCS=OFF", "-DBUILD_SHARED_LIBS=off", *std_cmake_args
    system "make"
    system "make", "install"
  end

  test do
    expected = <<~EOS
      uri:          https://brew.sh
      scheme:       https
      hostText:     brew.sh
      absolutePath: false
                    (always false for URIs with host)
    EOS
    assert_equal expected, shell_output("#{bin}/uriparse https://brew.sh").chomp
  end
end
