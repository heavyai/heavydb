#!/bin/bash

DEBIAN_FRONTEND=noninteractive apt update

# Generate list of upgradable packages
apt list --upgradable > upgradable.tmp

# Generate list of held packages (we need to skip these)
apt-mark showheld > held.tmp

# Remove held packages from upgradable list
grep -v -F -f held.tmp upgradable.tmp > cleaned.tmp

# Extract package name and version and format for group apt install command
awk -F'[/ ]' '
  BEGIN {
    print "#!/bin/bash\n"
    print "DEBIAN_FRONTEND=noninteractive apt-get update\n"
    print "DEBIAN_FRONTEND=noninteractive apt-get install -y --only-upgrade \\"
  }

  {
    # Extract package name and version and format for group apt install command
    if(NR>1) {print "  "$1"="$3" \\"}
  }

' cleaned.tmp > cudagl_package_updater.sh

# Remove the last continuation backslash
sed -i '$ s/ \\//' cudagl_package_updater.sh

# Make executable and clean up
chmod +x cudagl_package_updater.sh
rm held.tmp upgradable.tmp cleaned.tmp
