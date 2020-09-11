# Requires installation of:
#   CMake
#   Git for windows (git.exe MUST be in the system PATH in order for update_glslang_sources.py to update properly)
#   Python 2.x (also for update_glslang_sources.py)
#   Visual Studio command line build tools (Visual Studio 2019 or greater)
#
# Default is to build and install into "USERNAME/Documents/OmniSci/omnisci-deps"
# Specify a different path via parameter: `.\mapd-deps-windows.ps1 C:\foo`
param([string]$TargetPath = "$env:USERPROFILE\Documents\OmniSci\omnisci-deps")

$script_path = Split-Path $script:MyInvocation.MyCommand.Path
. $script_path\windows\deps-utils.src.ps1

if ((Test-Prerequisites) -eq $false) {
  Write-Host "Missing prerequisites."
  exit
}

$clobberExisting = $true
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
}

# Prepare vkpkg for use
Push-Location $deps_path
Write-Host "Cloning vcpkg"
git clone https://github.com/Microsoft/vcpkg.git
Write-Host "Boostrapping vcpkg"
Push-Location vcpkg
.\bootstrap-vcpkg.bat

Write-Host "Installing vcpkg dependencies (this will take a long time)..."
$package_list = @("glog", 
                  "thrift",
                  "openssl", 
                  "zlib", 
                  "libpng",
                  "pdcurses",
                  "curl",
                  "gdal",
                  "geos",
                  "blosc", 
                  "folly", 
                  "llvm",
                  "boost-log",
                  "boost-timer",
                  "boost-stacktrace",
                  "arrow",
                  "aws-sdk-cpp",
                  "librdkafka",
                  "libarchive"
                  )

foreach ($package in $package_list) {
  $package_config = $package+":x64-windows"
  .\vcpkg install $package_config
}

Pop-Location
Pop-Location
