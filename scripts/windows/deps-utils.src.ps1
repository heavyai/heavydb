
function Test-CommandExists($command, $errorMessage) {
  try {
    if (Get-Command $command -ErrorAction Stop) {
      Write-Host " Found $command." -ForegroundColor Green
      return $true 
    }
  } catch {
    $Host.UI.WriteErrorLine(" $command not found.")
    $Host.UI.WriteErrorLine("   $errorMessage.")
    return $false
  }
}

function Find-MsBuild([int] $MaxVersion = 2019) {
  $agentPath = "$Env:programfiles (x86)\Microsoft Visual Studio\2017\BuildTools\MSBuild\15.0\Bin\msbuild.exe"
  $devPath = "$Env:programfiles (x86)\Microsoft Visual Studio\2017\Enterprise\MSBuild\15.0\Bin\msbuild.exe"
  $proPath = "$Env:programfiles (x86)\Microsoft Visual Studio\2017\Professional\MSBuild\15.0\Bin\msbuild.exe"
  $communityPath = "$Env:programfiles (x86)\Microsoft Visual Studio\2017\Community\MSBuild\15.0\Bin\msbuild.exe"
  $fallback2015Path = "${Env:ProgramFiles(x86)}\MSBuild\14.0\Bin\MSBuild.exe"
  $fallback2013Path = "${Env:ProgramFiles(x86)}\MSBuild\12.0\Bin\MSBuild.exe"
  $fallbackPath = "C:\Windows\Microsoft.NET\Framework\v4.0.30319"
  
  If ((2017 -le $MaxVersion) -And (Test-Path $agentPath)) { return $agentPath } 
  If ((2017 -le $MaxVersion) -And (Test-Path $devPath)) { return $devPath } 
  If ((2017 -le $MaxVersion) -And (Test-Path $proPath)) { return $proPath } 
  If ((2017 -le $MaxVersion) -And (Test-Path $communityPath)) { return $communityPath } 
  If ((2015 -le $MaxVersion) -And (Test-Path $fallback2015Path)) { return $fallback2015Path } 
  If ((2013 -le $MaxVersion) -And (Test-Path $fallback2013Path)) { return $fallback2013Path } 
  If (Test-Path $fallbackPath) { return $fallbackPath } 
  return $false
}     
 
function Test-Prerequisites {
  Write-Host "Checking for prerequisites."
  $all_found = $true
  $all_found = (Test-CommandExists git "Git must be installed and in your path environment") -and $all_found
  $all_found = (Test-CommandExists cmake "Cmake must be installed and in your Path environment") -and $all_found
  # $all_found = (Test-CommandExists python "Python 2.7+ must be installed and in your Path environment") -and $all_found
  if ((Find-MsBuild) -eq $false) {
    $all_found = $false
    $Host.UI.WriteErrorLine(" msbuild.exe not found, Visual Studio command line tools must be installed.")
  } else {
    Write-Host " Found msbuild." -ForegroundColor Green
  }
  return $all_found
}

function New-Directory-Quiet($Name) {
  Write-Host "Creating directory $Name"
  New-Item $Name -ItemType "directory" > $null
}
  
function Get-Tree($Path, $Include = '*') { 
  @(Get-Item $Path -Include $Include -Force) + 
  (Get-ChildItem $Path -Recurse -Include $Include -Force) | 
    sort pspath -Descending -unique
} 

function Remove-Tree($Path, $Include = '*') {
  if (Test-Path($Path)) {
    Get-Tree $Path $Include | Remove-Item -Force -Recurse
    return Test-Path($Path)
  } else {
    return $false
  }
}

function Build-Solution {
  param($Name, $Target, [string[]]$Configurations)
  Foreach ($config in $Configurations) { 
    Write-Host "Building configuration $config"
    Write-Host "Building $($Name)" -foregroundcolor green
    & "$($msBuildExe)" "$($Name)" -t:"$($Target)" -p:Configuration="$($config)" -p:CL_MPCount=3 -maxcpucount -verbosity:minimal
    cmake -DBUILD_TYPE="$config" -P cmake_install.cmake  
  }
}
