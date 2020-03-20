class Gdal < Formula
  desc "Geospatial Data Abstraction Library"
  homepage "https://www.gdal.org/"
  url "https://download.osgeo.org/gdal/2.4.4/gdal-2.4.4.tar.xz"

  head do
    url "https://github.com/OSGeo/gdal.git"
    depends_on "doxygen" => :build
  end

  depends_on "libdap"
  depends_on "proj5"
  depends_on "expat"
  depends_on "libkml-master"
  #depends_on "libiconv"

  conflicts_with "cpl", :because => "both install cpl_error.h"

  def install
    args = [
      # Base configuration
      "--prefix=#{prefix}",
      "--mandir=#{man}",
      "--disable-debug",
      "--with-libtool",
      "--with-local=#{prefix}",
      "--with-threads",

      # Homebrew backends
      "--with-curl=/usr/bin/curl-config",
      "--with-libkml=#{Formula["libkml-master"].prefix}",
      "--with-expat=#{Formula["expat"].prefix}",
      "--with-proj=#{Formula["proj5"].opt_prefix}",
      #"--with-libiconv=#{Formula["libiconv"].prefix}",

      # Explicitly disable some features
      "--with-armadillo=no",
      "--with-qhull=no",
      "--with-odbc=no",
      "--with-webp=no",
      "--with-crypto=no",
      "--with-jpeg=internal",
      "--with-png=internal",
      "--with-gif=no",
      "--with-freexl=no",
      "--with-cfitsio=no",
      "--with-libtiff=internal",
      "--with-geotiff=internal",
      "--with-libjson-c=internal",
      "--without-opencl",
      "--without-sqlite3",
      "--without-grass",
      "--without-jpeg12",
      "--without-libgrass",
      "--without-mysql",
      "--without-perl",
      "--without-python",
      "--without-geos",
      "--without-xml",
      "--without-xml2",
      "--without-jasper",
      "--without-xerces",

      # Unsupported backends are either proprietary or have no compatible version
      # in Homebrew. Podofo is disabled because Poppler provides the same
      # functionality and then some.
      "--without-gta",
      "--without-ogdi",
      "--without-fme",
      "--without-hdf4",
      "--without-hdf5",
      "--without-openjpeg",
      "--without-netcdf",
      "--without-fgdb",
      "--without-ecw",
      "--without-kakadu",
      "--without-mrsid",
      "--without-jp2mrsid",
      "--without-mrsid_lidar",
      "--without-msg",
      "--without-oci",
      "--without-ingres",
      "--without-idb",
      "--without-sde",
      "--without-podofo",
      "--without-rasdaman",
      "--without-sosi",
    ]

    # Work around "error: no member named 'signbit' in the global namespace"
    # Remove once support for macOS 10.12 Sierra is dropped
    if DevelopmentTools.clang_build_version >= 900
      ENV.delete "SDKROOT"
      ENV.delete "HOMEBREW_SDKROOT"
    end

    system "./configure", *args
    system "make", "-j", "4"
    system "make", "install"

    # Build Python bindings
    cd "swig/python" do
      system "python3", *Language::Python.setup_install_args(prefix)
    end
    bin.install Dir["swig/python/scripts/*.py"]

    system "make", "man" if build.head?
    # Force man installation dir: https://trac.osgeo.org/gdal/ticket/5092
    system "make", "install-man", "INST_MAN=#{man}"
    # Clean up any stray doxygen files
    Dir.glob("#{bin}/*.dox") { |p| rm p }
  end

  test do
    # basic tests to see if third-party dylibs are loading OK
    system "#{bin}/gdalinfo", "--formats"
    system "#{bin}/ogrinfo", "--formats"
    system "python3", "-c", "import gdal"
  end
end
