# Requires installation of:
#   CMake
#   Git for windows (git.exe MUST be in the system PATH in order for update_glslang_sources.py to update properly)
#   Python 2.x (also for update_glslang_sources.py)
#   Visual Studio command line build tools (Visual Studio 2019 or greater)
#
# Default is to build and install into "USERNAME/Documents/OmniSci/omnisci-deps"
# Specify a different path via parameter: `.\mapd-deps-windows.ps1 C:\foo`
param([string]$TargetPath = "$env:USERPROFILE\Documents\OmniSci\omnisci-deps",
             [string]$vcpkg_release ="dd462392f4651dcbce3051225a20b161035bef5e",
             [switch]$clobber=$false,
             [switch]$exclude_static=$false)

write-host "Param [TargetPath = $TargetPath vcpkg_release = $vcpkg_release clobber = $clobber exclude_static = $exclude_static ]"

$script_path = Split-Path $script:MyInvocation.MyCommand.Path
. $script_path\windows\deps-utils.src.ps1

if ((Test-Prerequisites) -eq $false) {
  Write-Host "Missing prerequisites."
  exit
}

$ErrorActionPreference = "Stop"

$deps_path = $TargetPath
# $deps_build_path = "$deps_path\build"

Write-Host "Building dependencies to $deps_path"

# Enable ssl / tls
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

if ($clobberExisting) {
  if ((Remove-Tree($deps_path)) -ne $false) {
    Write-Error "Failed to remove existing deps directory because Windows. Aborting script."
    exit
  }
  
  # create the main deps folder and a build folder
  New-Directory-Quiet $deps_path
  # New-Directory-Quiet $deps_build_path

  # Prepare vkpkg for use
  Push-Location $deps_path
  Write-Host "Cloning vcpkg"
  git clone https://github.com/Microsoft/vcpkg.git
  Write-Host "Boostrapping vcpkg"
  Push-Location vcpkg

  if($vcpkg_release) {
    write-host "Cloning specific vcpkg commit [$vcpkg_release]"

    git -c advice.detachedHead=false checkout $vcpkg_release
  }
  .\bootstrap-vcpkg.bat
} else {
  Write-Host "deps_path $deps_path"
  Push-Location $deps_path\vcpkg
}

Write-Host "Installing vcpkg dependencies (this will take a long time)..."
$static_package_list = @("glog",
                  "thrift",
                  "openssl", 
                  "zlib", 
                  "libpng",
                  "curl",
                  "gdal",
                  "geos",
                  "blosc", 
                  "folly", 
                  "llvm",
                  "tbb",
                  "boost-log",
                  "boost-timer",
                  "boost-stacktrace",
                  "boost-geometry",
                  "boost-circular-buffer",
                  "boost-graph",
                  "boost-process",
                  "boost-sort",
                  "boost-uuid",
                  "boost-iostreams",
                  "aws-sdk-cpp",
                  "librdkafka",
                  "libarchive",
                  "xerces-c",
                  "arrow",
                  "proj4[tools]"
                  "proj"
                  "expat"
                  "libkml"
                  "uriparser"
                  )
$package_list = $static_package_list + "pdcurses"
foreach ($package in $package_list) {
  $package_config = $package+":x64-windows"
  .\vcpkg install $package_config
}

if(-Not $exclude_static) {
  foreach ($static_package in $static_package_list) {
    $package_config = $static_package+":x64-windows-static"
    .\vcpkg install $package_config
    if(-Not $?) {
      .\vcpkg install --recurse $package_config
    }
  }
}


Pop-Location
Pop-Location
