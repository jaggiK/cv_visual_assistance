// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <functional>
#include <vector>
#include <algorithm>

namespace GNAPluginNS {
namespace memory {

enum rType : uint8_t {
    REQUEST_STORE = 0x1,
    REQUEST_ALLOCATE = 0x2,
    REQUEST_BIND = 0x4,
    REQUEST_INITIALIZER = 0x8,
};
/**
 * @brief region of firmware data
 */
enum rRegion {
    REGION_RO,
    REGION_RW,
    REGION_AUTO,
};

struct MemRequest {
    rRegion  _region;
    uint8_t   _type;
    void *_ptr_out;
    const void *_ptr_in = nullptr;
    std::function<void(void * data, size_t size)> _initializer;
    // holds arbitrary value
    std::vector<uint8_t> _data;
    uint8_t _element_size;
    size_t _num_elements;
    size_t _alignment;
    size_t _offset = 0;
    // expansion in bytes due to large depended layers
    size_t _padding = 0;
    MemRequest(rRegion region,
                rType req,
                void *ptr_out,
                const void *ptr_in,
                uint8_t element_size = 0,
                size_t num_elements = 0,
                size_t alignment = 1,
                size_t offset = 0) : _region(region),
                                     _type(req),
                                     _ptr_out(ptr_out),
                                     _ptr_in(ptr_in),
                                     _element_size(element_size),
                                     _num_elements(num_elements),
                                     _alignment(alignment),
                                     _offset(offset) {}

    /**
     * Store value only request
     * @tparam T
     * @param req
     * @param ptr_out
     * @param element
     * @param num_elements
     * @param alignment
     */
    template<class T>
    MemRequest(rRegion region,
                void *ptr_out,
                T element,
                size_t num_elements,
                size_t alignment = 1) : _region(region),
                                        _type(REQUEST_STORE),
                                        _ptr_out(ptr_out),
                                        _element_size(sizeof(T)),
                                        _num_elements(num_elements),
                                        _alignment(alignment) {
        _data.resize(sizeof(T));
        std::copy(reinterpret_cast<uint8_t *>(&element), reinterpret_cast<uint8_t *>(&element) + sizeof(T), _data.begin());
    }
/**
     * Store initializer request
     * @param req
     * @param ptr_out
     * @param element
     * @param num_elements
     * @param alignment
     */
    MemRequest(rRegion region,
               void   *ptr_out,
               size_t  regionSize,
               std::function<void(void * data, size_t size)> initializer,
               rType req = REQUEST_INITIALIZER,
               size_t  alignment = 1) : _region(region),
                                        _type(REQUEST_INITIALIZER | req),
                                        _ptr_in(ptr_out),
                                        _ptr_out(ptr_out),
                                        _element_size(1),
                                        _num_elements(regionSize),
                                        _alignment(alignment),
                                        _initializer(initializer) {
    }
};
}  // namespace memory
}  // namespace GNAPluginNS