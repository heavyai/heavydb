class Geos < Formula
  desc "GEOS - Geometry Engine, Open Source"
  homepage "https://trac.osgeo.org/geos/"
  url "https://download.osgeo.org/geos/geos-3.8.1.tar.bz2"

  head do
    url "https://github.com/OSGeo/geos.git"
    depends_on "doxygen" => :build
  end

  depends_on "cmake" => :build

  def install
    mkdir "cmake-build" do
      system "cmake", "..", "-DCMAKE_BUILD_TYPE=Release", *std_cmake_args
      system "make", "install"
    end

  test do
    system "#{bin}/geos-config", "--prefix"
    system "#{bin}/geos-config", "--version"
  end
end
