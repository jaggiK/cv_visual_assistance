# Copyright (C) 2018-2020 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

cmake_policy(SET CMP0054 NEW)

#we have number of dependencies stored on ftp
include(dependency_solver)

set_temp_directory(TEMP "${IE_MAIN_SOURCE_DIR}")
if (CMAKE_CROSSCOMPILING)
    set(CMAKE_STAGING_PREFIX "${TEMP}")
endif()

include(ExternalProject)

include(linux_name)
if(COMMAND get_linux_name)
    get_linux_name(LINUX_OS_NAME)
endif()

include(CMakeParseArguments)

if (ENABLE_MYRIAD)
    include(vpu_dependencies)
endif()

## enable cblas_gemm from OpenBLAS package
if (GEMM STREQUAL "OPENBLAS")
    if(NOT BLAS_LIBRARIES OR NOT BLAS_INCLUDE_DIRS)
        find_package(BLAS REQUIRED)
        if(BLAS_FOUND)
            find_path(BLAS_INCLUDE_DIRS cblas.h)
        else()
            message(ERROR "OpenBLAS not found: install OpenBLAS or set -DBLAS_INCLUDE_DIRS=<path to dir with cblas.h> and -DBLAS_LIBRARIES=<path to libopenblas.so or openblas.lib>")
        endif()
    endif()
    debug_message(STATUS "openblas=" ${BLAS_LIBRARIES})
endif ()

#MKL-ml package
if (GEMM STREQUAL "MKL")
if(NOT MKLROOT)
    message(FATAL_ERROR "MKLROOT not found: install MKL and set -DMKLROOT=<path_to_MKL>")
endif()
set(MKL ${MKLROOT})
debug_message(STATUS "mkl_ml=" ${MKLROOT})
endif ()

## Intel OMP package
if (THREADING STREQUAL "OMP")
    if (WIN32)
        RESOLVE_DEPENDENCY(OMP
                ARCHIVE_WIN "iomp.zip"
                TARGET_PATH "${TEMP}/omp"
                ENVIRONMENT "OMP"
                VERSION_REGEX ".*_([a-z]*_([a-z0-9]+\\.)*[0-9]+).*")
    elseif(LINUX)
        RESOLVE_DEPENDENCY(OMP
                ARCHIVE_LIN "iomp.tgz"
                TARGET_PATH "${TEMP}/omp"
                ENVIRONMENT "OMP"
                VERSION_REGEX ".*_([a-z]*_([a-z0-9]+\\.)*[0-9]+).*")
    else(APPLE)
        RESOLVE_DEPENDENCY(OMP
                ARCHIVE_MAC "iomp_20190130_mac.tgz"
                TARGET_PATH "${TEMP}/omp"
                ENVIRONMENT "OMP"
                VERSION_REGEX ".*_([a-z]*_([a-z0-9]+\\.)*[0-9]+).*")
    endif()
    log_rpath_from_dir(OMP "${OMP}/lib")
    debug_message(STATUS "intel_omp=" ${OMP})
endif ()

## TBB package
if (THREADING STREQUAL "TBB" OR THREADING STREQUAL "TBB_AUTO")
    reset_deps_cache(TBBROOT TBB_DIR)

    if(NOT DEFINED TBB_DIR AND NOT DEFINED ENV{TBB_DIR})
        if (WIN32)
            #TODO: add target_path to be platform specific as well, to avoid following if
            RESOLVE_DEPENDENCY(TBB
                    ARCHIVE_WIN "tbb2020_20191023_win_tbbbind_patched.zip"
                    TARGET_PATH "${TEMP}/tbb"
                    ENVIRONMENT "TBBROOT"
                    VERSION_REGEX ".*_([a-z]*_([a-z0-9]+\\.)*[0-9]+).*")
        elseif(ANDROID)  # Should be before LINUX due LINUX is detected as well
            RESOLVE_DEPENDENCY(TBB
                    ARCHIVE_ANDROID "tbb2020_20191023_android.tgz"
                    TARGET_PATH "${TEMP}/tbb"
                    ENVIRONMENT "TBBROOT"
                    VERSION_REGEX ".*_([a-z]*_([a-z0-9]+\\.)*[0-9]+).*")
        elseif(LINUX)
            RESOLVE_DEPENDENCY(TBB
                    ARCHIVE_LIN "tbb2020_20191023_lin_tbbbind_patched.tgz"
                    TARGET_PATH "${TEMP}/tbb"
                    ENVIRONMENT "TBBROOT")
        else(APPLE)
            RESOLVE_DEPENDENCY(TBB
                    ARCHIVE_MAC "tbb2020_20191023_mac.tgz"
                    TARGET_PATH "${TEMP}/tbb"
                    ENVIRONMENT "TBBROOT"
                    VERSION_REGEX ".*_([a-z]*_([a-z0-9]+\\.)*[0-9]+).*")
        endif()
    else()
        if(DEFINED TBB_DIR)
            get_filename_component(TBB ${TBB_DIR} DIRECTORY)
        else()
            get_filename_component(TBB $ENV{TBB_DIR} DIRECTORY)
        endif()
    endif()

    update_deps_cache(TBBROOT "${TBB}" "Path to TBB root folder")
    update_deps_cache(TBB_DIR "${TBBROOT}/cmake" "Path to TBB package folder")

    if (WIN32)
        log_rpath_from_dir(TBB "${TBB_DIR}/../bin")
    else ()
        log_rpath_from_dir(TBB "${TBB_DIR}/../lib")
    endif ()
    debug_message(STATUS "tbb=" ${TBB})
endif ()

if (ENABLE_OPENCV)
    reset_deps_cache(OpenCV_DIR)

    set(OPENCV_VERSION "4.2.0")
    set(OPENCV_BUILD "082")
    if (WIN32)
        RESOLVE_DEPENDENCY(OPENCV
                ARCHIVE_WIN "opencv_${OPENCV_VERSION}-${OPENCV_BUILD}.txz"
                TARGET_PATH "${TEMP}/opencv_${OPENCV_VERSION}/opencv"
                ENVIRONMENT "OpenCV_DIR"
                VERSION_REGEX ".*_([0-9]+.[0-9]+.[0-9]+).*")
    elseif(APPLE)
        RESOLVE_DEPENDENCY(OPENCV
                ARCHIVE_MAC "opencv_${OPENCV_VERSION}-${OPENCV_BUILD}_osx.txz"
                TARGET_PATH "${TEMP}/opencv_${OPENCV_VERSION}_osx/opencv"
                ENVIRONMENT "OpenCV_DIR"
                VERSION_REGEX ".*_([0-9]+.[0-9]+.[0-9]+).*")
    elseif(LINUX)
        if (${CMAKE_SYSTEM_PROCESSOR} STREQUAL "armv7l")
            set(OPENCV_SUFFIX "debian9arm")
        elseif (${LINUX_OS_NAME} STREQUAL "CentOS 7" OR CMAKE_CXX_COMPILER_VERSION VERSION_LESS "4.9")
            set(OPENCV_SUFFIX "centos7")
        elseif (${LINUX_OS_NAME} STREQUAL "Ubuntu 16.04")
            set(OPENCV_SUFFIX "ubuntu16")
        elseif (${LINUX_OS_NAME} STREQUAL "Ubuntu 18.04")
            set(OPENCV_SUFFIX "ubuntu18")
        endif()
        RESOLVE_DEPENDENCY(OPENCV
                ARCHIVE_LIN "opencv_${OPENCV_VERSION}-${OPENCV_BUILD}_${OPENCV_SUFFIX}.txz"
                TARGET_PATH "${TEMP}/opencv_${OPENCV_VERSION}_${OPENCV_SUFFIX}/opencv"
                ENVIRONMENT "OpenCV_DIR"
                VERSION_REGEX ".*_([0-9]+.[0-9]+.[0-9]+).*")
    endif()

    if(ANDROID)
        set(ocv_cmake_path "${OPENCV}/sdk/native/jni/")
    else()
        set(ocv_cmake_path "${OPENCV}/cmake")
    endif()

    update_deps_cache(OpenCV_DIR "${ocv_cmake_path}" "Path to OpenCV package folder")

    if(WIN32)
        log_rpath_from_dir(OPENCV "${OpenCV_DIR}/../bin")
    elseif(ANDROID)
        log_rpath_from_dir(OPENCV "${OpenCV_DIR}/../../../lib")
    else()
        log_rpath_from_dir(OPENCV "${OpenCV_DIR}/../lib")
    endif()
    debug_message(STATUS "opencv=" ${OPENCV})
endif()

include(ie_parallel)

if (ENABLE_GNA)
    reset_deps_cache(
            GNA_PLATFORM_DIR
            GNA_KERNEL_LIB_NAME
            GNA_LIBS_LIST
            GNA_LIB_DIR
            libGNA_INCLUDE_DIRS
            libGNA_LIBRARIES_BASE_PATH)
    if (GNA_LIBRARY_VERSION STREQUAL "GNA1")
        RESOLVE_DEPENDENCY(GNA
                ARCHIVE_UNIFIED "gna_20181120.zip"
                TARGET_PATH "${TEMP}/gna")
    else()
        if(GNA_LIBRARY_VERSION STREQUAL "GNA1_1401")
            set(GNA_VERSION "01.00.00.1401")
        endif()
        if(GNA_LIBRARY_VERSION STREQUAL "GNA2")
            set(GNA_VERSION "02.00.00.0587")
        endif()
        RESOLVE_DEPENDENCY(GNA
                ARCHIVE_UNIFIED "GNA_${GNA_VERSION}.zip"
                TARGET_PATH "${TEMP}/gna_${GNA_VERSION}"
                VERSION_REGEX ".*_([0-9]+.[0-9]+.[0-9]+.[0-9]+).*")
    endif()
    debug_message(STATUS "gna=" ${GNA})
endif()

configure_file(
        "${IE_MAIN_SOURCE_DIR}/cmake/share/InferenceEngineConfig.cmake.in"
        "${CMAKE_BINARY_DIR}/share/InferenceEngineConfig.cmake"
        @ONLY)

configure_file(
        "${IE_MAIN_SOURCE_DIR}/cmake/share/InferenceEngineConfig-version.cmake.in"
        "${CMAKE_BINARY_DIR}/share/InferenceEngineConfig-version.cmake"
        COPYONLY)

configure_file(
        "${IE_MAIN_SOURCE_DIR}/cmake/ie_parallel.cmake"
        "${CMAKE_BINARY_DIR}/share/ie_parallel.cmake"
        COPYONLY)
