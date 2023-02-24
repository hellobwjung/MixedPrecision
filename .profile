# ~/.profile: executed by the command interpreter for login shells.
# This file is not read by bash(1), if ~/.bash_profile or ~/.bash_login
# exists.
# see /usr/share/doc/bash/examples/startup-files for examples.
# the files are located in the bash-doc package.

# the default umask is set in /etc/profile; for setting the umask
# for ssh logins, install and configure the libpam-umask package.
#umask 022

# if running bash
PS1="\w \n\u@\h-$ "
if [ -n "$BASH_VERSION" ]; then
    # include .bashrc if it exists
    if [ -f "$HOME/.bashrc" ]; then
        . "$HOME/.bashrc"
    fi
fi
# path for bw
#export ANDROID_NDK_HOME=/home2/bw.jung/android/ndk/android-ndk-r16b
export CUDA_LIB_PATH="/usr/local/cuda/lib64"


export CURL_CA_BUNDLE="/usr/share/ca-certificates/McAfee_Certificate.crt"

# set PATH so it includes user's private bin directories

PATH="$HOME/bin:$HOME/.local/bin:$PATH"
export PATH=$ANDORID_NDK_HOME:$PATH
#export PATH=$CUDA_LIB_PATH$ANDORID_NDK_HOME:$PATH

#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64




export http_proxy="http://12.26.204.100:8080"
export https_proxy="https://12.26.204.100:8080"
export ftp_proxy="http://12.26.204.100:8080"
export no_proxy="12.*.*.*,*.samsung.net,127.0.0.1,pfs.nprotect.com,127.0.0.1:16105,127.0.0.1:16106,127.0.0.1:16108,127.0.0.1:16107,*.dsvdi.net,.samsungds.net,localhost,127.0.0.1:14440"

