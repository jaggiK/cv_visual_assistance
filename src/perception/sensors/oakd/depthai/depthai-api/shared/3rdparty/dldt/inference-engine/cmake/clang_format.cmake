# Copyright (C) 2018-2020 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
if (ENABLE_CLANG_FORMAT)
    set(CLANG_FORMAT_FILENAME clang-format-9 clang-format)
    find_program(CLANG_FORMAT NAMES ${CLANG_FORMAT_FILENAME} PATHS ENV PATH)
    if (CLANG_FORMAT)
        execute_process(COMMAND ${CLANG_FORMAT} ${CMAKE_CURRENT_SOURCE_DIR} ARGS --version OUTPUT_VARIABLE CLANG_VERSION)
        if (NOT CLANG_VERSION OR CLANG_VERSION STREQUAL "")
            message(WARNING "Supported clang-format version is 9!")
            set(ENABLE_CLANG_FORMAT OFF)
        else()
            string(REGEX REPLACE ".*([0-9]+)\\.[0-9]+\\.[0-9]+.*" "\\1" CLANG_FORMAT_MAJOR_VERSION ${CLANG_VERSION})
            if (NOT ${CLANG_FORMAT_MAJOR_VERSION} EQUAL "9")
                message(WARNING "Supported clang-format version is 9!")
                set(ENABLE_CLANG_FORMAT OFF)
            endif()
        endif()
    endif()
endif()

if(ENABLE_CLANG_FORMAT)
    add_custom_target(clang_format_check_all)
    add_custom_target(clang_format_fix_all)
    set(CLANG_FORMAT_ALL_OUTPUT_FILES "" CACHE INTERNAL "All clang-format output files")
endif()

function(add_clang_format_target TARGET_NAME)
    if(NOT ENABLE_CLANG_FORMAT)
        return()
    endif()

    set(options ALL)
    set(oneValueArgs "")
    set(multiValueArgs "FOR_TARGETS" "FOR_SOURCES" "EXCLUDE_PATTERNS")
    cmake_parse_arguments(CLANG_FORMAT "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    if(CLANG_FORMAT_ALL)
        set(all ALL)
    endif()

    foreach(target IN LISTS CLANG_FORMAT_FOR_TARGETS)
        get_target_property(target_sources "${target}" SOURCES)
        list(APPEND CLANG_FORMAT_FOR_SOURCES ${target_sources})
    endforeach()
    list(REMOVE_DUPLICATES CLANG_FORMAT_FOR_SOURCES)

    set(all_output_files "")
    foreach(source_file IN LISTS CLANG_FORMAT_FOR_SOURCES)
        set(exclude FALSE)
        foreach(pattern IN LISTS CLANG_FORMAT_EXCLUDE_PATTERNS)
            if(source_file MATCHES "${pattern}")
                set(exclude ON)
                break()
            endif()
        endforeach()

        if(exclude)
            continue()
        endif()

        # ignore object libraries
        if(NOT EXISTS "${source_file}")
            continue()
        endif()

        file(RELATIVE_PATH source_file_relative "${CMAKE_CURRENT_SOURCE_DIR}" "${source_file}")
        set(output_file "${CMAKE_CURRENT_BINARY_DIR}/clang_format/${source_file_relative}.clang")
        string(REPLACE ".." "__" output_file "${output_file}")
        get_filename_component(output_dir "${output_file}" DIRECTORY)
        file(MAKE_DIRECTORY "${output_dir}")

        add_custom_command(
            OUTPUT
            "${output_file}"
            COMMAND
            "${CMAKE_COMMAND}"
            -D "CLANG_FORMAT=${CLANG_FORMAT}"
            -D "INPUT_FILE=${source_file}"
            -D "OUTPUT_FILE=${output_file}"
            -P "${IE_MAIN_SOURCE_DIR}/cmake/clang_format_check.cmake"
            DEPENDS
            "${source_file}"
            "${IE_MAIN_SOURCE_DIR}/cmake/clang_format_check.cmake"
            COMMENT
            "[clang-format] ${source_file}"
            VERBATIM)

        list(APPEND all_output_files "${output_file}")
    endforeach()

    set(CLANG_FORMAT_ALL_OUTPUT_FILES
        ${CLANG_FORMAT_ALL_OUTPUT_FILES} ${all_output_files}
        CACHE INTERNAL
        "All clang-format output files")

    add_custom_target(${TARGET_NAME}
        ${all}
        DEPENDS ${all_output_files}
        COMMENT "[clang-format] ${TARGET_NAME}")

    add_custom_target(${TARGET_NAME}_fix
        COMMAND
        "${CMAKE_COMMAND}"
        -D "CLANG_FORMAT=${CLANG_FORMAT}"
        -D "INPUT_FILES=${CLANG_FORMAT_FOR_SOURCES}"
        -D "EXCLUDE_PATTERNS=${CLANG_FORMAT_EXCLUDE_PATTERNS}"
        -P "${IE_MAIN_SOURCE_DIR}/cmake/clang_format_fix.cmake"
        DEPENDS
        "${CLANG_FORMAT_FOR_SOURCES}"
        "${IE_MAIN_SOURCE_DIR}/cmake/clang_format_fix.cmake"
        COMMENT
        "[clang-format] ${TARGET_NAME}_fix"
        VERBATIM)

    # if(CLANG_FORMAT_FOR_TARGETS)
    #     foreach(target IN LISTS CLANG_FORMAT_FOR_TARGETS)
    #         add_dependencies(${target} ${TARGET_NAME})
    #     endforeach()
    # endif()

    add_dependencies(clang_format_check_all ${TARGET_NAME})
    add_dependencies(clang_format_fix_all ${TARGET_NAME}_fix)
endfunction()
