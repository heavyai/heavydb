class XercesC < Formula
  desc "Validating XML parser"
  homepage "https://xerces.apache.org/xerces-c/"
  url "https://dependencies.mapd.com/thirdparty/xerces-c-3.2.2.tar.gz"
  sha256 "dd6191f8aa256d3b4686b64b0544eea2b450d98b4254996ffdfe630e0c610413"

  depends_on "cmake" => :build

  def install
    ENV.cxx11

    mkdir "build" do
      system "cmake", "..", "-DBUILD_SHARED_LIBS=OFF", "-Dnetwork=off", "-Dtranscoder=iconv", *std_cmake_args
      system "make"
      system "ctest", "-V"
      system "make", "install"
      system "make", "clean"
      system "cmake", "..", "-DBUILD_SHARED_LIBS=OFF", "-Dnetwork=off", "-Dtranscoder=iconv", *std_cmake_args
      system "make"
      lib.install Dir["src/*.a"]
    end
    # Remove a sample program that conflicts with libmemcached
    # on case-insensitive file systems
    (bin/"MemParse").unlink
  end

  test do
    (testpath/"ducks.xml").write <<~EOS
      <?xml version="1.0" encoding="iso-8859-1"?>

      <ducks>
        <person id="Red.Duck" >
          <name><family>Duck</family> <given>One</given></name>
          <email>duck@foo.com</email>
        </person>
      </ducks>
    EOS

    output = shell_output("#{bin}/SAXCount #{testpath}/ducks.xml")
    assert_match "(6 elems, 1 attrs, 0 spaces, 37 chars)", output
  end
end
