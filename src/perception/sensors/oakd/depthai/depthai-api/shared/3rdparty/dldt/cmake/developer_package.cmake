# Copyright (C) 2018 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

include(CPackComponent)
unset(IE_CPACK_COMPONENTS_ALL CACHE)

set(IE_CPACK_IE_DIR       deployment_tools/inference_engine)

function(ie_cpack_set_library_dir)
    string(TOLOWER ${CMAKE_SYSTEM_PROCESSOR} ARCH)
    if(ARCH STREQUAL "x86_64" OR ARCH STREQUAL "amd64") # Windows detects Intel's 64-bit CPU as AMD64
        set(ARCH intel64)
    elseif(ARCH STREQUAL "i386")
        set(ARCH ia32)
    endif()

    if(WIN32)
        set(IE_CPACK_LIBRARY_PATH ${IE_CPACK_IE_DIR}/lib/$<CONFIG>/${ARCH} PARENT_SCOPE)
    else()
        set(IE_CPACK_LIBRARY_PATH ${IE_CPACK_IE_DIR}/lib/${ARCH} PARENT_SCOPE)
    endif()
endfunction()

ie_cpack_set_library_dir()

#
# ie_cpack_add_component(NAME ...)
#
# Wraps original `cpack_add_component` and adds component to internal IE list
#
macro(ie_cpack_add_component NAME)
    list(APPEND IE_CPACK_COMPONENTS_ALL ${NAME})
    set(IE_CPACK_COMPONENTS_ALL "${IE_CPACK_COMPONENTS_ALL}" CACHE STRING "" FORCE)
    cpack_add_component(${NAME} ${ARGN})
endmacro()

macro(ie_cpack)
    set(CPACK_GENERATOR "TGZ")
    if(WIN32)
        set(CPACK_PACKAGE_NAME inference-engine_$<CONFIG>)
    else()
        set(CPACK_PACKAGE_NAME inference-engine)
    endif()
    set(CPACK_INCLUDE_TOPLEVEL_DIRECTORY OFF)
    set(CPACK_ARCHIVE_COMPONENT_INSTALL ON)
    set(CPACK_PACKAGE_VENDOR "Intel")
    set(CPACK_COMPONENTS_ALL ${ARGN})

    if(OS_FOLDER)
        set(CPACK_SYSTEM_NAME "${OS_FOLDER}")
    endif()

    include(CPack)
endmacro()

# External dependencies
find_package(Threads)

# Detect target
include(target_flags)

# printing debug messages
include(debug)

if(UNIX AND NOT APPLE)
    set(LINUX ON)
endif()

string(TOLOWER ${CMAKE_SYSTEM_PROCESSOR} ARCH_FOLDER)
if(ARCH_FOLDER STREQUAL "x86_64" OR ARCH_FOLDER STREQUAL "amd64") # Windows detects Intel's 64-bit CPU as AMD64
    set(ARCH_FOLDER intel64)
elseif(ARCH_FOLDER STREQUAL "i386")
    set(ARCH_FOLDER ia32)
endif()

if(OS_FOLDER)
	message ("**** OS FOLDER IS: [${OS_FOLDER}]")
	if("${OS_FOLDER}" STREQUAL "ON")
		message ("**** USING OS FOLDER: [${CMAKE_SYSTEM_NAME}]")
		set(BIN_FOLDER "bin/${CMAKE_SYSTEM_NAME}/${ARCH_FOLDER}")
	else()
		set(BIN_FOLDER "bin/${OS_FOLDER}/${ARCH_FOLDER}")
	endif()
else()
    set(BIN_FOLDER "bin/${ARCH_FOLDER}")
endif()

if("${CMAKE_BUILD_TYPE}" STREQUAL "")
    debug_message(STATUS "CMAKE_BUILD_TYPE not defined, 'Release' will be used")
    set(CMAKE_BUILD_TYPE "Release")
endif()

if(COVERAGE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fprofile-arcs -ftest-coverage -O0")
endif()

if(UNIX)
    SET(LIB_DL ${CMAKE_DL_LIBS})
endif()

set(OUTPUT_ROOT ${OpenVINO_MAIN_SOURCE_DIR})

# Enable postfixes for Debug/Release builds
set(IE_DEBUG_POSTFIX_WIN "d")
set(IE_RELEASE_POSTFIX_WIN "")
set(IE_DEBUG_POSTFIX_LIN "")
set(IE_RELEASE_POSTFIX_LIN "")
set(IE_DEBUG_POSTFIX_MAC "d")
set(IE_RELEASE_POSTFIX_MAC "")

if(WIN32)
    set(IE_DEBUG_POSTFIX ${IE_DEBUG_POSTFIX_WIN})
    set(IE_RELEASE_POSTFIX ${IE_RELEASE_POSTFIX_WIN})
elseif(APPLE)
    set(IE_DEBUG_POSTFIX ${IE_DEBUG_POSTFIX_MAC})
    set(IE_RELEASE_POSTFIX ${IE_RELEASE_POSTFIX_MAC})
else()
    set(IE_DEBUG_POSTFIX ${IE_DEBUG_POSTFIX_LIN})
    set(IE_RELEASE_POSTFIX ${IE_RELEASE_POSTFIX_LIN})
endif()

set(CMAKE_DEBUG_POSTFIX ${IE_DEBUG_POSTFIX})
set(CMAKE_RELEASE_POSTFIX ${IE_RELEASE_POSTFIX})

if (WIN32)
    # Support CMake multiconfiguration for Visual Studio build
    set(IE_BUILD_POSTFIX $<$<CONFIG:Debug>:${IE_DEBUG_POSTFIX}>$<$<CONFIG:Release>:${IE_RELEASE_POSTFIX}>)
else ()
    if (${CMAKE_BUILD_TYPE} STREQUAL "Debug" )
        set(IE_BUILD_POSTFIX ${IE_DEBUG_POSTFIX})
    else()
        set(IE_BUILD_POSTFIX ${IE_RELEASE_POSTFIX})
    endif()
endif()
message(STATUS "CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")

add_definitions(-DIE_BUILD_POSTFIX=\"${IE_BUILD_POSTFIX}\")

if(NOT UNIX)
    if (WIN32)
        # set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MT")
        # set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MTd")
    endif()
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${OUTPUT_ROOT}/${BIN_FOLDER})
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${OUTPUT_ROOT}/${BIN_FOLDER})
    set(CMAKE_COMPILE_PDB_OUTPUT_DIRECTORY ${OUTPUT_ROOT}/${BIN_FOLDER})
    set(CMAKE_PDB_OUTPUT_DIRECTORY ${OUTPUT_ROOT}/${BIN_FOLDER})
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${OUTPUT_ROOT}/${BIN_FOLDER})
else()
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${OUTPUT_ROOT}/${BIN_FOLDER}/${CMAKE_BUILD_TYPE}/lib)
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${OUTPUT_ROOT}/${BIN_FOLDER}/${CMAKE_BUILD_TYPE}/lib)
    set(CMAKE_COMPILE_PDB_OUTPUT_DIRECTORY ${OUTPUT_ROOT}/${BIN_FOLDER}/${CMAKE_BUILD_TYPE})
    set(CMAKE_PDB_OUTPUT_DIRECTORY ${OUTPUT_ROOT}/${BIN_FOLDER}/${CMAKE_BUILD_TYPE})
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${OUTPUT_ROOT}/${BIN_FOLDER}/${CMAKE_BUILD_TYPE})
endif()

if(APPLE)
	set(CMAKE_MACOSX_RPATH 1)
endif(APPLE)

# Use solution folders
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

include(sdl)
include(os_flags)
include(sanitizer)

function(set_ci_build_number)
    set(OpenVINO_MAIN_SOURCE_DIR "${CMAKE_SOURCE_DIR}")
    include(version)
    set(CI_BUILD_NUMBER "${CI_BUILD_NUMBER}" PARENT_SCOPE)
endfunction()
set_ci_build_number()
