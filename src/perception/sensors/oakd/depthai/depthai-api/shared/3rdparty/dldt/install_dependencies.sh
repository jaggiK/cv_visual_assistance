#!/bin/bash
# Copyright (C) 2018 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

params=$@

yes_or_no() {
    if [ "$params" == "-y" ]; then
        return 0
    fi

    while true; do
        read -p "Add third-party Nux Dextop repository and install FFmpeg package (y) / Skip this step (N)" yn
        case $yn in
            [Yy]*) return 0 ;;
            [Nn]*) return 1 ;;
        esac
    done
}

# install dependencies
if [ -f /etc/lsb-release ]; then
    # Ubuntu
    sudo -E apt update
    sudo -E apt-get install -y \
            build-essential \
            cmake \
            curl \
            wget \
            libssl-dev \
            ca-certificates \
            git \
            libboost-regex-dev \
            gcc-multilib \
            g++-multilib \
            libgtk2.0-dev \
            pkg-config \
            unzip \
            automake \
            libtool \
            autoconf \
            libcairo2-dev \
            libpango1.0-dev \
            libglib2.0-dev \
            libgtk2.0-dev \
            libswscale-dev \
            libavcodec-dev \
            libavformat-dev \
            libgstreamer1.0-0 \
            gstreamer1.0-plugins-base \
            libusb-1.0-0-dev \
            libopenblas-dev
    if apt-cache search --names-only '^libpng12-dev'| grep -q libpng12; then
        sudo -E apt-get install -y libpng12-dev
    else
        sudo -E apt-get install -y libpng-dev
    fi
elif [ -f /etc/redhat-release ]; then
    # CentOS 7.x
    sudo -E yum install -y centos-release-scl epel-release
    sudo -E yum install -y \
            wget \
            tar \
            xz \
            p7zip \
            unzip \
            yum-plugin-ovl \
            which \
            libssl-dev \
            ca-certificates \
            git \
            boost-devel \
            libtool \
            gcc \
            gcc-c++ \
            make \
            glibc-static \
            glibc-devel \
            libstdc++-static \
            libstdc++-devel \
            libstdc++ libgcc \
            glibc-static.i686 \
            glibc-devel.i686 \
            libstdc++-static.i686 \
            libstdc++.i686 \
            libgcc.i686 \
            libusbx-devel \
            openblas-devel \
            libusbx-devel \
            gstreamer1 \
            gstreamer1-plugins-base

    # Python 3.6 for Model Optimizer
    sudo -E yum install -y rh-python36
    source scl_source enable rh-python36

    wget https://cmake.org/files/v3.12/cmake-3.12.3.tar.gz
    tar xf cmake-3.12.3.tar.gz
    cd cmake-3.12.3
    ./configure
    make -j16
    sudo -E make install

    echo
    echo "FFmpeg is required for processing audio and video streams with OpenCV. Please select your preferred method for installing FFmpeg:"
    echo
    echo "Option 1: Allow installer script to add a third party repository, Nux Dextop (http://li.nux.ro/repos.html), which contains FFmpeg. FFmpeg rpm package will be installed from this repository. "
    echo "WARNING: This repository is NOT PROVIDED OR SUPPORTED by CentOS."
    echo "Once added, this repository will be enabled on your operating system and can thus receive updates to all packages installed from it. "
    echo
    echo "Consider the following ways to prevent unintended 'updates' from this third party repository from over-writing some core part of CentOS:"
    echo "a) Only enable these archives from time to time, and generally leave them disabled. See: man yum"
    echo "b) Use the exclude= and includepkgs= options on a per sub-archive basis, in the matching .conf file found in /etc/yum.repos.d/ See: man yum.conf"
    echo "c) The yum Priorities plug-in can prevent a 3rd party repository from replacing base packages, or prevent base/updates from replacing a 3rd party package."
    echo
    echo "Option 2: Skip FFmpeg installation."
    echo

    if yes_or_no; then
        sudo -E rpm -Uvh http://li.nux.ro/download/nux/dextop/el7/x86_64/nux-dextop-release-0-1.el7.nux.noarch.rpm
        sudo -E yum install -y ffmpeg
    else
        echo "FFmpeg installation skipped. You may build FFmpeg from sources as described here: https://trac.ffmpeg.org/wiki/CompilationGuide/Centos"
        echo
    fi
elif [ -f /etc/os-release ] && grep -q "raspbian" /etc/os-release; then
    # Raspbian
    sudo -E apt update
    sudo -E apt-get install -y \
            build-essential \
            cmake \
            curl \
            wget \
            libssl-dev \
            ca-certificates \
            git \
            libboost-regex-dev \
            libgtk2.0-dev \
            pkg-config \
            unzip \
            automake \
            libtool \
            autoconf \
            libcairo2-dev \
            libpango1.0-dev \
            libglib2.0-dev \
            libgtk2.0-dev \
            libswscale-dev \
            libavcodec-dev \
            libavformat-dev \
            libgstreamer1.0-0 \
            gstreamer1.0-plugins-base \
            libusb-1.0-0-dev \
            libopenblas-dev
    if apt-cache search --names-only '^libpng12-dev'| grep -q libpng12; then
        sudo -E apt-get install -y libpng12-dev
    else
        sudo -E apt-get install -y libpng-dev
    fi
else
    echo "Unknown OS, please install build dependencies manually"
fi