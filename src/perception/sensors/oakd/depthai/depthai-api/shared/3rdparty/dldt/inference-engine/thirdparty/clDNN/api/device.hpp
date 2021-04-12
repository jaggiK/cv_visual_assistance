/*
// Copyright (c) 2019 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
*/

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once
#include "cldnn.hpp"
#include <string>
#include <stdexcept>
#include <vector>
#include <map>

namespace cldnn {
/// @addtogroup cpp_api C++ API
/// @{

/// @defgroup cpp_device GPU Device
/// @{

/// @brief Information about the device properties and capabilities.
struct device_info {
    uint32_t cores_count;     ///< Number of available HW cores.
    uint32_t core_frequency;  ///< Clock frequency in MHz.

    uint64_t max_work_group_size;  ///< Maximum number of work-items in a work-group executing a kernel using the data parallel execution model.
    uint64_t max_local_mem_size;   ///< Maximum size of local memory arena in bytes.
    uint64_t max_global_mem_size;  ///< Maximum size of global device memory in bytes.
    uint64_t max_alloc_mem_size;   ///< Maximum size of memory object allocation in bytes.

    uint64_t max_image2d_width;   ///< Maximum image 2d width supported by the device.
    uint64_t max_image2d_height;  ///< Maximum image 2d height supported by the device.

    // Flags (for layout compatibility fixed size types are used).
    uint8_t supports_fp16;             ///< Does engine support FP16.
    uint8_t supports_fp16_denorms;     ///< Does engine support denormalized FP16.
    uint8_t supports_subgroups_short;  ///< Does engine support cl_intel_subgroups_short.
    uint8_t supports_image;            ///< Does engine support images (CL_DEVICE_IMAGE_SUPPORT cap).

    uint8_t supports_imad;   ///< Does engine support int8 mad.
    uint8_t supports_immad;  ///< Does engine support int8 multi mad.

    std::string dev_name;     ///< Device ID string
    std::string driver_version;  ///< Version of OpenCL driver
};

struct device_impl;

/// @brief Represents clDNN detected device object. Use device_query to get list of available objects.
struct device {
    static device create_default();

    explicit device(device_impl* data)
        : _impl(data) {
        if (_impl == nullptr)
            throw std::invalid_argument("implementation pointer should not be null");
    }

    /// @brief Returns information about properties and capabilities of the device.
    device_info get_info() const;

    // TODO add move construction/assignment
    device(const device& other) : _impl(other._impl) {
        retain();
    }

    device& operator=(const device& other) {
        if (_impl == other._impl) return *this;
        release();
        _impl = other._impl;
        retain();
        return *this;
    }

    ~device() {
        release();
    }

    friend bool operator==(const device& lhs, const device& rhs) { return lhs._impl == rhs._impl; }
    friend bool operator!=(const device& lhs, const device& rhs) { return !(lhs == rhs); }

    device_impl* get() const { return _impl; }

private:
    device_impl* _impl;

    void retain();
    void release();
};

struct device_query_impl;

/// @brief Represents clDNN object, which allows to query for list of devices.
struct device_query {
    /// @brief Constructs engine configuration with specified options.
    /// @param Query only for devices, which supports out of order execution (default in cldnn).
    /// @param Query for devices in user provided opencl context.
    explicit device_query(void* clcontext = nullptr, void* user_device = nullptr);
    // TODO add move construction/assignment
    device_query(const device_query& other) : _impl(other._impl) {
        retain();
    }

    /// Returns map of {device_id, device object} of available devices on system.
    /// Device_id is string. First device will have id: "0", second "1" etc.
    std::map<std::string, device> get_available_devices() const;

    device_query& operator=(const device_query& other) {
        if (_impl == other._impl) return *this;
        release();
        _impl = other._impl;
        retain();
        return *this;
    }

    ~device_query() {
        release();
    }

    friend bool operator==(const device_query& lhs, const device_query& rhs) { return lhs._impl == rhs._impl; }
    friend bool operator!=(const device_query& lhs, const device_query& rhs) { return !(lhs == rhs); }

    device_query_impl* get() const { return _impl; }

private:
    device_query_impl* _impl;

    void retain();
    void release();
};
CLDNN_API_CLASS(device_query)

/// @}

/// @}

}  // namespace cldnn
