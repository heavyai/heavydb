

###############################################################################
#                                                                             #
#                              FUNCTION CALLS                                 #
#                                                                             #
###############################################################################



report_error()
{
    exit_code=$1
    last_command=${@:2}
    
    if [ "$exit_code" != 0 ]; then
                             
echo "

###############################################################################
                                                                             
 ERROR DURING INSTALLATION
 Timestamp: "$(date)"                                                   
 Last command failed with exit code ${exit_code}                            
 Last command run was [${last_command}].
 Check the install.log for error conditions.                             
                                                                           
###############################################################################" >&3
    fi
}

exit_on_error()
{
    exit_code=$1
    last_command=${@:2}
    
    if [ "$exit_code" != 0 ]; then
                             
echo "

###############################################################################
                                                                             
 ERROR DURING INSTALLATION
 Timestamp: "$(date)"                                                   
 Last command failed with exit code ${exit_code}                            
 Last command run was [${last_command}].
 Check the install.log for error conditions.
                                                                           
###############################################################################" >&3
        exit "$exit_code"
    fi
}

spinner()
{
    set +x
    local pid=$!
    local delay=0.75
    local spinstr='|/-\'
    while [ "$(ps a | awk '{print $1}' | grep $pid)" ]; do
        local temp=${spinstr#?}
        printf " [%c]  " "$spinstr" >&3
        local spinstr=$temp${spinstr%"$temp"}
        sleep $delay
        printf "\b\b\b\b\b\b"  >&3
    done
    printf "    \b\b\b\b"  >&3
    set -x
}

query_graphics_install() {
    echo "
* 1) Proceed without CUDA support or
* 2) Manually install CUDA drivers. or
* c) Cancel installation.
* [1,2,c] " >&3; read -r CUDA_CHOICE

    while [ "$CUDA_CHOICE" != "1" ] && [ "$CUDA_CHOICE" != "2" ] && [ "$CUDA_CHOICE" != "c" ]; do
        echo "** Only 1,2, or c is a valid response." >&3
        read -r CUDA_CHOICE
    done

    if [ "$CUDA_CHOICE" = "1" ]; then
        touch "$TMPDIR"/GPU_NO
    elif [ "$CUDA_CHOICE" = "2" ]; then
        touch "$TMPDIR"/GPU_YES
        touch "$TMPDIR"/CUDA_DRIVERS_MANUAL_INSTALL
    elif [ "$CUDA_CHOICE" = "c" ]; then
        exit 1
    fi
}

test_secure_boot() {
  SECURE_BOOT=$(sudo mokutil --sb-state)
  if [[ "SecureBoot enabled" = $SECURE_BOOT ]] ; then
    echo "
WARNING. Secure boot enabled in BIOS/UEFI. You can either install
the graphic's card driver manually or disable secure boot in
the BIOS/UEFI and then rerun the script." >&3
    query_graphics_driver_install
    return 1
  fi
  return 0
}

not_Supported_OperatingSystem_Version_with_GPU()
{
    local linux_distro=$1 # Should be CentOS or Ubuntu
    local supported_versions="SUPPORTED_VERSIONS_${linux_distro}"
    if [[ ${!supported_versions} = "" ]]; then
      # At this point if the operating system isn't CentOS or Ubuntu
      # it is an error
      echo "ERROR unsupported [${linux_distro}]" >&3
      return 1
    fi
    echo " * This script supports CUDA drivers for ${linux_distro} ${!supported_versions}."
    query_graphics_driver_install
}
restart_host()
{
    printf "\n * System is ready for reboot. Proceed with reboot? [y/n] " >&3; read -r REBOOT_RESPONSE

    while [ "$REBOOT_RESPONSE" != "y" ] && [ "$REBOOT_RESPONSE" != "n" ]; do
        echo "* Waiting to reboot. [y/n]" >&3
        read -r REBOOT_RESPONSE
    done

    if [ "$REBOOT_RESPONSE" = "y" ]; then
        sudo reboot
    else 
        echo "* Reboot system before proceeding."
        return 0
    fi
}

install_nvidia_drivers()
{
    rm "$TMPDIR/${INSTALL_METHOD^^}_INSTALL" # remove state file

    if [ -f "$TMPDIR"/GPU_YES ]; then
      echo "
###############################################################################
#                                                                             #
#                          Installing nvidia driver                           #
#                                                                             #
###############################################################################
" >&3
      list_available_nvidia_cards
      if [ "$CARD_LIST" = "" ] ; then
        echo "WARNING. NVIDIA graphics cards not found.  Automated install of graphics drivers will not proceed"  >&3
        echo "You will be provided with the option to manually install." >&3
        touch "$TMPDIR"/CUDA_DRIVERS_MANUAL_INSTALL
      else
        local_number_card_types=$(echo "$CARD_LIST" | uniq --skip-fields 1 | wc -l)
         if [ "$number_card_types" -gt "1" ] ; then
            echo "WARNING. Multiple types of NVIDIA graphics cards detected. Automated install of graphics drivers will not proceed" >&3
            echo "You will be provided with the option to manually install." >&3
            touch "$TMPDIR"/CUDA_DRIVERS_MANUAL_INSTALL
         else
            echo " * Found the following NVIDIA graphic cards [$CARD_LIST]; proceeding with automated install of graphics drivers"  >&3
            install_nvidia_driver_$OS_TYPE  $(cat $TMPDIR/$OS_TYPE)
        fi
      fi
    fi

    if [ -f "$TMPDIR"/CUDA_DRIVERS_MANUAL_INSTALL ]; then
      echo "
STOPPING HERE for manual download of CUDA  drivers for your Operating System.

Install CUDA per the instructions on the NVIDIA web site in another terminal.
When complete return here and press the enter key to complete installation." >&3
      read manual_install_completed
      mv "$TMPDIR"/CUDA_DRIVERS_MANUAL_INSTALL  "$TMPDIR"/CUDA_DRIVERS_MANUAL_INSTALLED
    fi
    install_heavyai_$INSTALL_METHOD # install_heavy_apt, install_heavy_yum or install_heavy_yum
}

get_release_id_for_OS()
{
  # set result in global $RELEASE_ID   
  local os_type=$1
  local release_set=$2
  local version_id=$(cat /etc/os-release | grep '^VERSION_ID=' | cut -d '=' -f2 | sed 's/"//g')
  if [[ $release_set =~ $version_id ]]; then
    RELEASE_ID=$version_id
    return 0
  fi
  version_id="not set"
  while [[ ! $release_set =~ $version_id ]]; do
    echo "* Unable to detect $os_type release. Please enter one of  [${release_set}] or Q to quit"; read -r version_id
    if [ "$version_id" = "Q" ] ; then 
      return 1 
    fi 
  done
  RELEASE_ID=$version_id
  return 0
}

get_linux_release()
{
  # set result in global $RELEASE_ID   

  local linux_type=$1
  local  supported_versions
  if [  "$linux_type" = "CentOS" ]; then 
    supported_versions=7,8
  elif [ "$linux_type" = "Ubuntu" ]; then
    supported_versions=18.04,20.04
  fi
  get_release_id_for_OS $linux_type $supported_versions; RC=$?
  return $RC 
}

get_linux_type()
{
    # set result in global $DISTRIBUTION_ID   

    local distribution_id=$(cat /etc/os-release | grep '^ID=' | cut -d '=' -f2 | sed 's/"//g')
    
     if [ "$distribution_id" == "ubuntu" ]; then
        distribution_id="Ubuntu"
    elif [ "$distribution_id" == "centos" ]; then
        distribution_id="CentOS"
    else
        echo "* Unable to autodetect Linux distro."

        LINUX_DISTRO="not set"
        while [ "$LINUX_DISTRO" != "1" ] && [ "$LINUX_DISTRO" != "2" ] && [ "$LINUX_DISTRO" != "3" ]; do
            echo "* Please enter your Linux distribution  1) Ubuntu, 2) CentOS , or 3) to exit? [1,2,3]"; read -r LINUX_DISTRO
        done

        if [ "$LINUX_DISTRO" = "1" ]; then
            distribution_id="Ubuntu"
        elif [ "$LINUX_DISTRO" = "2" ]; then
            distribution_id="CentOS"
        elif [ "$LINUX_DISTRO" = "3" ]; then
            echo "* This HEAVY.AI installation script only supports Ubuntu or CentOS." 
            return 1 
        fi
  fi
  DISTRIBUTION_ID=$distribution_id
  return 0
}

get_heavy_tarball() {
  echo "
###############################################################################
#                                                                             #
#                          Installing HEAVYAI tar file                        #
#                                                                             #
###############################################################################
" >&3
  local target_dir=$1
  local release_type=$2 # os or ee
  local processor_type=$3 # render or cpu
  # Note 'os' and 'render' is not a valid combination
  if [ "$release_type" = "os" -a "$processor_type" = "render" ] ; then
    printf "Error, the opensource builds do not have a gpu/render verion" >&3
    return 1
  fi
  source_tar_file="https://releases.heavy.ai/$release_type/tar/heavyai-${release_type}-latest-Linux-x86_64-$processor_type.tar.gz"
  printf " * Retrieving tar [%s] file\n" "$source_tar_file" >&3

  target_tar_file=$TMPDIR/heavyai-${release_type}-latest-Linux-x86_64-${processor_type}.tar.gz
  # keep curl silent to avoid writing progress meter to log,
  # but log size of file.
  curl -s ${source_tar_file} --output  ${target_tar_file}
  # log how big the down file is
  wc -c  ${target_tar_file}


  tar_root=$(dirname $(tar -tf ${target_tar_file}| head -1))

  tar -pxvf  ${target_tar_file}
  sudo mv $tar_root $target_dir
  chmod 755 $target_dir
  echo "sudo chown -R $HEAVYAI_USER:$HEAVYAI_GROUP $target_dir"
  sudo chown -R $HEAVYAI_USER:$HEAVYAI_GROUP $target_dir
  return 0
}


# Note the calls to _CentOS and _Ubuntu functions
# are constructed from shell variables - hence the odd
# case.

install_base_packages_Ubuntu() {
  printf " * Installing base packages - gnupg, default jre, curl and apt-transport-https\n" >&3
  sudo apt -y update &
  spinner

  sudo apt -y upgrade &
  spinner

  sudo apt -y install default-jre-headless \
                      curl \
                      apt-transport-https \
                      gnupg \
  spinner
  if [ -f $TMPDIR/GPU_YES ]; then
    printf " * Installing base package - ubuntu-drivers-common\n" >&3
    sudo apt -y install ubuntu-drivers-common &
    spinner
  fi
}

install_base_packages_CentOS() {
  printf " * Installing base packages - epel-release, pciutils and java" >&3
  sudo yum -y update &
  spinner
  sudo yum -y install epel-release pciutils java &
  spinner
}

list_available_nvidia_cards() {
  # Returns vaules in CARD_LIST
  # Requires pciutils package.

  # nvidia id =  10de;
  #     video controller = 03;
  #       VGA = 00; 
  #       XGA = 01
  #       3D Controller  = 02
  #       Display controller  = 80
  # for example 01:00.0 VGA compatible controller [0300]: NVIDIA Corporation TU106M [GeForce RTX 2070 Mobile] [10de:1f10] (rev a1)
  # is an nvidia (10de) video controller/VGA (0300) card

  CARD_LIST=$(lspci -nn  | grep -i 10de  | egrep "03(00|01|02|80)")
}

install_nvidia_driver_CentOS()
{
  # Reference
  # https://docs.nvidia.com/datacenter/tesla/tesla-installation-notes/index.html and
  # https://docs.nvidia.com/datacenter/tesla/pdf/NVIDIA_Driver_Installation_Quickstart.pdf
  print " ** Attempting graphics cards discovery and driver install **\n"  >&3

  local_number_card_types=$(echo "$CARD_LIST" | uniq --skip-fields 1 | wc -l)
  if [ "$number_card_types" -gt "1" ] ; then
    echo "WARNING. Multiple types of NVIDIA graphics cards detected. Automated install of graphics drivers will not proceed" >&3
    return 1
  fi
  echo " * Installing kernel-devel-$(uname -r) kernel-headers-$(uname -r) and yum-utils" >&3
  local centos_id="$1"
  local arch=$( /bin/arch )
  local dist="rhel${centos_id}"
  sudo yum -y install kernel-devel-$(uname -r) kernel-headers-$(uname -r) yum-utils  &
  spinner
  sudo yum-config-manager -y  --add-repo http://developer.download.nvidia.com/compute/cuda/repos/${dist}/${arch}/cuda-${dist}.repo 
  echo " * Installing dkms and updating yum" >&3
  sudo yum -y install dkms &
  spinner
  sudo yum -y update &
  spinner
  echo " * Installing nvidia-driver-lates-dkms and vulkan" >&3
  sudo yum install -y nvidia-driver-latest-dkms vulkan &
  spinner
  return 0
}


install_nvidia_driver_Ubuntu()
{
  echo " * Installing nvidia driver for Ubuntu" >&3
  local ignore_id="$1" # install_nvidia_driver_CentOS needs its version. Ignore for Ubuntu
  sudo apt -y install linux-headers-generic &
  spinner
  
  sudo apt -y install libopengl0 &
  spinner
	recommended_driver=$(ubuntu-drivers devices 2>/dev/null | grep recommended | cut -d" " -f5)
  echo " *  System recommended driver is  [$recommended_driver]" >&3
	sudo apt -y --no-install-recommends install ${recommended_driver} &
	spinner
}


firewall_setup_CentOS()
{
  sudo yum list installed firewalld 2>&1 >/dev/null; ST=$? 
  if [ "$ST" -ne "0" ] ; then
    install_fw="not_set"
    while [ "1" = "1" ] ; do
      echo "The centos firewall is not installed.  Would you like to install it? [yes/no]"; read -r install_fw
      if [ "install_fw" == "no" ] ; then
        return 0 
      elif [ "install_fw" == "yes" ] ; then
        sudo yum -y install firewalld
        sudo systemctl enable firewalld
        break
      fi
    done
  fi
  sudo systemctl status firewalld 2>&1 >/dev/null; ST=$?
  if [ "$ST" -ne "0" ] ; then
    echo "The firewall daemon is not running.  Starting and enabling the daemon."
    sudo systemctl enable firewalld
    sudo systemctl start firewalld  
  fi
  echo "* Configuring your firewall for external access"
  printf "Running sudo firewall-cmd --zone=public --add-port=6273/tcp --permanent\n"
  sudo firewall-cmd --zone=public --add-port=6273/tcp --permanent
  echo "Running sudo firewall-cmd --reload"
  sudo firewall-cmd --reload
}

query_installation_type()
{
    # Is this an Enterprise or Open Source Installation? User can cancel at this point.
    local user_response="x"
    while [ "$user_response" != "1" ] && [ "$user_response" != "2" ] && [ "$user_response" != "c" ]; do
        local default="1"
        printf " * Is this an 1) Enterprise / Free or 2) Open Source installation? c) Cancel Install [$default] : " >&3
        read -r user_response
        user_response=${user_response:=$default}
        [ "$user_response" = "c" ]  && exit
    done

    if [ "$user_response" = "1" ]; then
        INSTALL_TYPE="Enterprise"
        touch "$TMPDIR"/ENT_INSTALL
    elif [ "$user_response" = "2" ]; then
        INSTALL_TYPE="OpenSource"
        touch "$TMPDIR"/OS_INSTALL
    fi
}

query_installation_method()
{
   local os_type=$1
    # Do we use apt, yum or a tar to install?
    local install_method="not_set"
    [[ $os_type = "Ubuntu" ]] && install_method="apt"
    [[ $os_type = "CentOS" ]] && install_method="yum"
    local user_response='x'
    while [ "$user_response" != "1" ] && [ "$user_response" != "2" ]; do
        local default="1"
        printf " * Installation method - 1) $install_method or 2) Tarball? [$default] : " >&3
        read -r user_response
        user_response=${user_response:=$default}
    done

    if [[ $user_response = "2" ]]; then
        install_method="tar"
    fi
    touch "$TMPDIR"/${install_method^^}_INSTALL
    INSTALL_METHOD=$install_method
}
 # Centos || Ubuntu
query_cpu_or_gpu(){
    local install_type=$1
    local os_type=$2 # Centos || Ubuntu
    # Will the user use a GPU in this installation?
    user_response="x"
    [[ $install_type = "OpenSource" ]] && user_response="n"
    while [ "$user_response" != "Y" ] && [ "$user_response" != "n" ]; do
        default="Y"
        printf " * Are you using a GPU with HEAVY.AI? [Y/n] : " >&3
        read -r user_response
        user_response=${user_response:=$default}
    done

    if [ "$user_response" = "Y" ]; then
       local supported_versions="SUPPORTED_VERSIONS_${os_type}"
       OPERATING_SYSTEM_VERSION=$(cat "$TMPDIR"/$os_type)
       if [[ ${!supported_versions} =~ $OPERATING_SYSTEM_VERSION ]]; then
         test_secure_boot; ST=$?
         [[ $ST -eq 1 ]] && return
         touch "$TMPDIR"/GPU_YES
        else
           not_Supported_OperatingSystem_Version_with_GPU
        fi
    else
      touch "$TMPDIR"/GPU_NO
    fi
}

query_set_firewall_options()
{
    local install_type=$1
    [[ $install_type = "OpenSource" ]] && return

    user_response="x"
    while [ "$user_response" != "Y" ] && [ "$user_response" != "n" ]; do
        default="Y"
    	printf "
 * To access HEAVYAI's web component (available with the EE version) it will be necessary to open the appropriate port(s)
 * in your firewall.  This script can attempt to configure your firewall.  However, if you already have a running firewall
 * with a complex set of rules it is recommended that your firewall be manually configured\n\n" >&3
    	printf " * Do you want ufw/firewalld installed and configured? [Y/n] " >&3
        read -r user_response
        user_response=${user_response:=$default}
    done

    if [ "$user_response" = "Y" ]; then
       touch "$TMPDIR"/FIREWALL
    fi

}

set_environment_variables()
{
  # Sets environment variables in $1 - ~/.bashrc by default

  CFG=${1:-/home/$USER/.bashrc}
  if [[ !  -f $CFG ]] ; then
   echo "Creating target cfg file [$CFG]"
   touch $CFG
  fi
  shift 1
  vars="$@"
  # an array of vars to be set in .bashrc
  if ! grep -q "#HEAVY.AI user specific variables" $CFG ; then
   echo "#HEAVY.AI user specific variables [$(date)]" >> $CFG
  else
   sed -i.bck "/#HEAVY.AI user specific variables/ s/variables .*$/variables [$(date)]/" $CFG
  fi

  for label in ${vars} ; do
    # read the contents of the file into var
     value="$(cat $TMPDIR/$label)"
    # update the export or create it.
     if grep -q $label $CFG ; then
       echo " * Updating $label to $value in $CFG"
       sed -i.bck "/${label}/ s|=.*|=${value}|" $CFG
     else
       echo " * Appending $label to $value in $CFG"
       echo "export $label=$value" >> $CFG
     fi
  done
}

initialize_systemd()
{
  if [[ ! -f $TMPDIR/env.sh ]] ; then
    echo "set_environment_variables must be run before initialization" >&3
   return 1
  fi
  # patch for a possible issue in install_heavy_systemd.
  [[ -f /usr/share/vulkan/icd.d/nvidia_icd.x86_64.json ]] &&  sudo ln -s /usr/share/vulkan/icd.d/nvidia_icd.x86_64.json /usr/share/vulkan/icd.d/nvidia_icd.json
  echo " * Initializing systemd for HEAVY.AI" >&3
  source $TMPDIR/env.sh

  pushd $HEAVYAI_PATH/systemd || exit_on_error $? | :0
  ./install_heavy_systemd.sh --non-interactive $HEAVYAI_PATH  $HEAVYAI_STORAGE $HEAVYAI_GROUP $HEAVYAI_USER || exit_on_error $? | :0
  popd
}

verify_installation()
{
    echo "
###############################################################################
#                                                                             #
#                           Verifying Installation                            #
#                                                                             #
############################################################################### " >&3
    if [ -f "$TMPDIR"/LICENSE_KEY ]; then

        echo " * Entering license key." >&3
        sudo cp  "$TMPDIR"/LICENSE_KEY $(cat "$TMPDIR"/HEAVYAI_STORAGE)/heavyai.license
        sudo chown $(cat "$TMPDIR"/HEAVYAI_USER):$(cat "$TMPDIR"/HEAVYAI_GROUP) $(cat "$TMPDIR"/HEAVYAI_STORAGE)/heavyai.license
	echo " * Restaring heavydb service" >&3
        sudo systemctl restart heavydb &
	spinner
    fi
    [ -f "$TMPDIR"/GPU_YES ] && verify_nvidia_install
}

import_sample_data_and_query()
{
    local user_response="x"
    while [ "$user_response" != "Y" ] && [ "$user_response" != "n" ]; do
      local default="Y"
    	printf " * Would you like to load the sample data ? [Y/n] : " >&3
      read -r user_response
      user_response=${user_response:=$default}
      [[ $user_response = "n" ]] && return
    done
    # only insert an allowed path entry of one isn't already present
    grep "allowed-import-path" $HEAVYAI_STORAGE/heavy.conf >/dev/null ; ST=$?
    if [[ $ST -ne 0 ]]; then
      printf "Updating $HEAVYAI_STORAGE/heavy.conf to add a white list path for import\n" >&3
      # make the import directory now or heavydb will not start.
      sudo mkdir -p $HEAVYAI_PATH/storage/import/sample_datasets
      sudo sed -i "/\[web\]/ i allowed-import-paths = [\"${HEAVYAI_PATH}/storage/import/sample_datasets/\"]" $HEAVYAI_STORAGE/heavy.conf
      sudo systemctl restart heavydb
      sudo systemctl status heavydb 
    else
      printf " *** \n   An entry for allowed-import-path already exists in $HEAVYAI_STORAGE/heavy.conf.\n   If this entry doesn't include ${HEAVYAI_PATH}/storage/import/ you might have to add it for the import to work\n ***\n\n" >&3
    fi

    echo " * Downloading and importing sample data. This may take some time." >&3
    pushd ${HEAVYAI_PATH}
    sudo ./insert_sample_data 2>&1 << -EOD-
2
-EOD-
    report_error $? "insert_sample_data" &&  ST=$? && popd && return $ST
    echo "
* Running query:
SELECT origin_city AS 'Origin', dest_city AS 'Destination', AVG(airtime) AS
'Average Airtime' FROM flights_2008_10k WHERE distance < 175 GROUP BY
origin_city, dest_city; " >&3

    ./bin/heavysql heavyai -u admin -p HyperInteractive >&3 << -EOD-
SELECT origin_city AS "Origin", dest_city AS "Destination", AVG(airtime) AS "Average Airtime"
FROM flights_2008_10k WHERE distance < 175 GROUP BY origin_city, dest_city;
-EOD-
   report_error $? "heavysql" && ST=$? && popd && return $ST
   popd

}

query_user_custom_details()
{
  local cust_config_confirmation="x"
  while [ "$cust_config_confirmation" != "Y" ]; do

    local heavyai_user
    read -p " *    Set the HEAVYAI user account. [${HEAVYAI_USER}] : " 2>&3 heavyai_user
    heavyai_user=${heavyai_user:=$HEAVYAI_USER}

    local heavyai_group
    read -p " *    Set the HEAVYAI user group. [${HEAVYAI_GROUP}] : " 2>&3 heavyai_group
    heavyai_group=${heavyai_group:=$HEAVYAI_GROUP}

    local  heavyai_storage
    read -p " *    Set the storage path. [${HEAVYAI_STORAGE}] : " 2>&3 heavyai_storage
    heavyai_storage=${heavyai_storage:=$HEAVYAI_STORAGE}

    local heavyai_install
    read -p " *    Set the install location. [${HEAVYAI_INSTALL}] : " 2>&3 -r heavyai_install
    heavyai_install=${heavyai_install:=$HEAVYAI_INSTALL}

    echo "
* The following settings will be used:
* HEAVYAI user account: ${heavyai_user}
* HEAVYAI user group: ${heavyai_group}
* Storage path: ${heavyai_storage}
* Install location: ${heavyai_install} " >&3

    while [ "$cust_config_confirmation" != "Y" ] && [ "$cust_config_confirmation" != "n" ]; do
      default="Y"
      read -p "Is this correct [Y/n] : " 2>&3 cust_config_confirmation
      cust_config_confirmation=${cust_config_confirmation:=$default}
    done
  done

  HEAVYAI_USER="$heavyai_user"
  HEAVYAI_GROUP="$heavyai_group"
  HEAVYAI_STORAGE="$heavyai_storage"
  HEAVYAI_INSTALL="$heavyai_install"

}

query_license()
{
    #If this is an enterprise install, store the license for verification step
    local  license_key=""
    if [ -f "$TMPDIR"/ENT_INSTALL ]; then
        local user_response="x"
        while [ "$user_response" != "1" ] && [ "$user_response" != "2" ]; do
          default="1"
          printf " * Enter License 1) by file 2) directly [$default]: " >&3;
          read -r user_response
          user_response=${user_response:=$default}
        done
        if [[ $user_response = 1 ]]; then
          local license_file=""
          while true; do
            printf " * Enter License file name : " >&3; read license_file
            [[ -f $license_file ]] && break
            printf "\n\tERROR - Can not read the file entered\n\n" >&3
          done
          license_key=$(cat $license_file)
        else
          printf " * Enter license key : " >&3 ; read -r license_key
        fi

        license_key=${license_key:-"NO KEY"}
        echo "$license_key" > "$TMPDIR"/LICENSE_KEY
    fi

}

get_user_install_information()
{
   local os_type=$1
   query_installation_type                  # creates state file and sets global INSTALL_TYPE
   query_license                            # creates state file
   query_installation_method $os_type       # creates state file and sets global INSTALL_METHOD
   query_cpu_or_gpu $INSTALL_TYPE $os_type  # create state file
   # Only EE comes with the web package - which may ports open in the firewall
   [ -f $TMPDIR/ENT_INSTALL ] && query_set_firewall_options $INSTALL_TYPE # create state file

    #Check if this is a custom or default install
    default="1"
    printf "
 * Is this a 1) default (recommended) or 2) custom (allows for specifying
   the HEAVYAI user, group, storage, and path) installation? [$default] : " >&3; read -r user_response
    user_response=${user_response:=$default}

    while [ "$user_response" != "1" ] && [ "$user_response" != "2" ]; do
        printf " ** Only 1 or 2 is a valid response : " >&3
        read -r user_response
        user_response=${user_response:=$default}
    done

    [[ $user_response = "2" ]] && query_user_custom_details

    echo "$HEAVYAI_USER" > "$TMPDIR"/HEAVYAI_USER
    echo "$HEAVYAI_GROUP" > "$TMPDIR"/HEAVYAI_GROUP
    echo "$HEAVYAI_STORAGE" > "$TMPDIR"/HEAVYAI_STORAGE
    echo "$HEAVYAI_PATH" > "$TMPDIR"/HEAVYAI_PATH
    echo "$HEAVYAI_LOG" > "$TMPDIR"/HEAVYAI_LOG
    echo "$HEAVYAI_INSTALL" > "$TMPDIR"/HEAVYAI_INSTALL

}

verify_nvidia_install()
{
echo "
###############################################################################
#                                                                             #
#                         Verifying CUDA Drivers                              #
#                                                                             #
###############################################################################" >&3

    echo " * Are you ready to verify your CUDA installation? [y/c]" >&3; read -r VERIFY_CUDA_RESPONSE

    while [ "$VERIFY_CUDA_RESPONSE" != "y" ] && [ "$VERIFY_CUDA_RESPONSE" != "c" ]; do
        echo " *   Press y when ready to verify CUDA installation or c to cancel installation." >&3
        read -r VERIFY_CUDA_RESPONSE
    done

    [ "$VERIFY_CUDA_RESPONSE" = "c" ] && echo "*   Cancelling installation." >&3 && exit 0

    nvidia-smi >&3

    echo " * Does the output show your GPU present? If so, press y to proceed.
* If not review Install CUDA drivers before proceeding. Press c to cancel installation" >&3; read -r PROCEED_RESPONSE

    while [ "$PROCEED_RESPONSE" != "y" ] && [ "$PROCEED_RESPONSE" != "c" ]; do
      read -r PROCEED_RESPONSE
    done

    [ "$PROCEED_RESPONSE" = "c" ] && echo " *   Cancelling installation." >&3 && exit 0
}


enable_firewall()
{
    echo " * Enabling the firewall" >&3
    if [ -f "$TMPDIR"/Ubuntu ]; then
        sudo ufw disable
        sudo ufw allow 6273/tcp
        sudo ufw allow ssh
        sudo ufw -f enable
    elif [ -f "$TMPDIR"/CentOS ]; then
        sudo yum -y install firewalld
        sudo systemctl start firewalld
        sudo systemctl enable firewalld
        sudo systemctl status firewalld

        echo " * Configuring your firewall for external access"
        sudo firewall-cmd --zone=public --add-port=6273/tcp --permanent
        sudo firewall-cmd --reload
    fi
    echo "DOne"
}


install_heavyai_apt()
{
echo "
###############################################################################
#                                                                             #
#                         Installing HEAVYAI                                  #
#                                                                             #
###############################################################################" >&3

    curl -s https://releases.heavy.ai/GPG-KEY-heavyai | sudo apt-key add -

    local processor="not_set"
    [ -f "$TMPDIR"/GPU_YES ] && processor="cuda"
    [ -f "$TMPDIR"/GPU_NO ] && processor="cpu"

    local install_type="not_set"
    [ -f "$TMPDIR"/OS_INSTALL ] && install_type="os"
    [ -f "$TMPDIR"/ENT_INSTALL ] && install_type="ee"
    echo "deb https://releases.heavy.ai/$install_type/apt/ stable $processor" | sudo tee /etc/apt/sources.list.d/heavyai.list

    sudo apt update &
    spinner
    sudo apt -y install heavyai &
    spinner
}

install_heavyai_yum()
{
echo "
###############################################################################
#                                                                             #
#                         Installing HEAVYAI                                  #
#                                                                             #
###############################################################################" >&3

    echo "[heavyai]" > $TMPDIR/heavyai.repo
    local processor="not_set"
    [ -f "$TMPDIR"/GPU_YES ] && processor="cuda"
    [ -f "$TMPDIR"/GPU_NO ] && processor="cpu"

    local install_type="not_set"
    [ -f "$TMPDIR"/OS_INSTALL ] && install_type="os"
    [ -f "$TMPDIR"/ENT_INSTALL ] && install_type="ee"
    echo "name='heavyai $install_type - $processor'" >> $TMPDIR/heavyai.repo
    echo "baseurl=https://releases.heavy.ai/$install_type/yum/stable/$processor" >> $TMPDIR/heavyai.repo

    echo "enabled=1
gpgcheck=1
repo_gpgcheck=0
gpgkey=https://releases.heavy.ai/GPG-KEY-heavyai" >> $TMPDIR/heavyai.repo

   sudo mv $TMPDIR/heavyai.repo /etc/yum.repos.d/heavyai.repo
   sudo chown root:root /etc/yum.repos.d/heavyai.repo
   sudo yum -y update
   sudo yum -y install heavyai &
   spinner

}

install_heavyai_tar()
{
    local heavyai_path=$(cat "$TMPDIR"/HEAVYAI_PATH)
    echo " * Installing HEAVY.AI using a Tarball" >&3
    #sudo mkdir -p "$heavyai_path"

    if [ -f "$TMPDIR"/GPU_NO -a -f "$TMPDIR"/ENT_INSTALL ]; then
            get_heavy_tarball $heavyai_path ee cpu &
            spinner
    elif [ -f "$TMPDIR"/GPU_YES -a  -f "$TMPDIR"/ENT_INSTALL ]; then
            get_heavy_tarball $heavyai_path ee render &
            spinner
    elif [ -f "$TMPDIR"/OS_INSTALL ]; then
            get_heavy_tarball $heavyai_path os cpu &
            spinner
    fi
}

activation()
{
    echo " * Activating OmniSci" >&3
    sudo systemctl enable heavydb
    exit_on_error $? !:0
    sudo systemctl start heavydb
    exit_on_error $? !:0

    if  [ -f "$TMPDIR"/ENT_INSTALL ]; then
        sudo systemctl enable heavy_web_server
        exit_on_error $? !:0
        sudo systemctl start heavy_web_server
        exit_on_error $? !:0
    fi

    echo " * Enabling HEAVYAI to start automatically when the system reboots."  >&3
}
set_installation_environment()
{
  echo "
###############################################################################
#                                                                             #
#                          Collecting system information                      #
#                                                                             #
###############################################################################

Preparing your Linux machine:
    Updating your system,
    creating the HEAVYAI users and groups,
    installing the required drivers,  HEAVYAI software and
    rebooting."  >&3

    get_linux_type || exit_on_error $? | :0 #get_linux_type sets return in DISTRIBUTION_ID either Ubuntu or CentOS
    get_linux_release $DISTRIBUTION_ID || exit_on_error $? | :0  # for linux 18.04 or 20.04 for centos 7 or 8

    # echo to files - will reread into env via env.sh
    echo $RELEASE_ID > $TMPDIR/$DISTRIBUTION_ID
    echo $DISTRIBUTION_ID > $TMPDIR/OS_TYPE
    get_user_install_information $DISTRIBUTION_ID
    local user_response="x"
    while [ "$user_response" != "y" ] && [ "$user_response" != "N" ]; do
      local default="N"
    	printf " * Configuration complete.  Proceed with installation? [y/N] : " >&3
      read -r user_response
      user_response=${user_response:=$default}
      [[ $user_response = "N" ]] && rm -rf $TMPDIR && exit
    done

    # base env for .bashrc
    vars="HEAVYAI_USER HEAVYAI_GROUP HEAVYAI_STORAGE HEAVYAI_PATH HEAVYAI_LOG"
    set_environment_variables  /home/$USER/.bashrc $vars
    # for the tmp installation env add OS_TYPE and PACKAGE_TYPE
    set_environment_variables $TMPDIR/env.sh $vars "OS_TYPE"

    source $TMPDIR/env.sh

    echo " * Creating the group  and user who will be the owner/run HEAVYAI daemons. " >&3
    grp_num=$(grep $HEAVYAI_GROUP /etc/group | wc -l)
    [[ $grp_num -eq 0 ]] && sudo groupadd $HEAVYAI_GROUP
    usr_num=$(grep $HEAVYAI_USER /etc/passwd | wc -l)
    [[ $usr_num -eq 0 ]] && sudo useradd --gid $HEAVYAI_GROUP --create-home  $HEAVYAI_USER
}
