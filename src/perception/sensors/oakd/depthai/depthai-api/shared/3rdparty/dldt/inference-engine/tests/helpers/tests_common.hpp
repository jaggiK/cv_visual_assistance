// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cmath>
#include <cctype>
#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <ie_layers.h>
#include <ie_blob.h>
#include <ie_input_info.hpp>
#include <ie_icnn_network.hpp>

#include "test_model_path.hpp"
#include <tests_file_utils.hpp>
#include <chrono>

#include "stdlib.h"
#include "stdio.h"
#include "string.h"

template <class T>
inline std::string to_string_c_locale(T value) {
    std::stringstream val_stream;
    val_stream.imbue(std::locale("C"));
    val_stream << value;
    return val_stream.str();
}

namespace testing {
const std::string DEVICE_CPU = "CPU";
const std::string DEVICE_GNA = "GNA";
const std::string DEVICE_GPU = "GPU";
const std::string DEVICE_HDDL = "HDDL";
const std::string DEVICE_FPGA = "FPGA";
const std::string DEVICE_MYRIAD = "MYRIAD";
const std::string DEVICE_MULTI = "MULTI";
const std::string DEVICE_HETERO = "HETERO";
}

class TestsCommon : public ::testing::Test {
public:
    static InferenceEngine::CNNLayer::Ptr createLayer(const std::string& type);

protected:
    void SetUp() override;
    void TearDown() override;

public:
    inline std::string get_mock_engine_name() {
        return make_plugin_name("mock_engine");
    }

    static std::string get_data_path(){
        const char* data_path = std::getenv("DATA_PATH");

        if (data_path == NULL){
            if(DATA_PATH != NULL){
                data_path = DATA_PATH;
            } else{
                ::testing::AssertionFailure()<<"DATA_PATH not defined";
            }
        }
        return std::string(data_path);
    }

    static std::string make_so_name(const std::string & input) {
#ifdef _WIN32
    #ifdef __MINGW32__
        std::string pre = "lib";
        std::string ext = ".dll";
    #else
        std::string pre = "";
        std::string ext = ".dll";
    #endif
#elif defined __APPLE__
        std::string pre = "lib";
        std::string ext = ".dylib";
#else
        std::string pre = "lib";
        std::string ext = ".so";
#endif
        return pre + input + IE_BUILD_POSTFIX + ext;
    }

    static std::string make_plugin_name(const std::string & input) {
        return make_so_name(input);
    }

    static void fill_data(InferenceEngine::Blob::Ptr& blob) {
        fill_data(blob->buffer().as<float*>(), blob->byteSize() / sizeof(float));
    }

    static void fill_data_const(InferenceEngine::Blob::Ptr& blob, float val) {
        fill_data_const(blob->buffer().as<float*>(), blob->size(), val);
    }

    static void fill_data(float *data, size_t size, size_t duty_ratio = 10) {
        for (size_t i = 0; i < size; i++) {
            if ( ( i / duty_ratio)%2 == 1) {
                data[i] = 0.0;
            } else {
                data[i] = sin((float)i);
            }
        }
    }

    static void fill_data_non_zero(int32_t *data, size_t size, int n) {
        for (size_t i = 0; i < size; i++) {
            data[i] = n*i%254+1;
        }
    }

    static void fill_data_bin(float *data, size_t size) {
        for (size_t i = 0; i < size; i++) {
            data[i] = sinf((float)i) > 0.f ? 1.f : -1.f;
        }
    }

    static void fill_data_bin_packed(int8_t *data, size_t size) {
        int nbits = 8;
        for (size_t i = 0; i < div_up(size, nbits); i++) {
            data[i] = static_cast<int8_t>(i % 255);
        }
    }

    static void fill_data_sine(float *data, size_t size, float center, float ampl, float omega) {
        for (size_t i = 0; i < size; i++) {
            data[i] = center + ampl * sin((float)i * omega);
        }
    }

    static void fill_data_const(float *data, size_t size, float value) {
        for (size_t i = 0; i < size; i++) {
            data[i] = value;
        }
    }

    static void fill_data_dbgval(float *data, size_t size, float alpha = 1.0f) {
        for (size_t i = 0; i < size; i++) {
            data[i] = i * alpha;
        }
    }

    static void compare(
            InferenceEngine::Blob &res,
            InferenceEngine::Blob &ref,
            float max_diff = 0.01,
            const std::string assertDetails = "") {
        float *res_ptr = res.buffer().as<float*>();
        size_t res_size = res.size();

        float *ref_ptr = ref.buffer().as<float*>();
        size_t ref_size = ref.size();

        ASSERT_EQ(res_size, ref_size) << assertDetails;

        for (size_t i = 0; i < ref_size; i++) {
            ASSERT_NEAR(res_ptr[i], ref_ptr[i], max_diff) << assertDetails;
        }
    }

    static void compare(
            const float* res,
            const float* ref,
            size_t size,
            float max_diff = 0.01f,
            const std::string assertDetails = "") {
        for (size_t i = 0lu; i < size; i++) {
            ASSERT_NEAR(res[i], ref[i], max_diff) << assertDetails << ", index=" << i;
        }
    }

    template <class T>
    static void compare(std::vector<T> & a, std::vector<T> & b) {
        ASSERT_EQ(a.size(), b.size());
        for (size_t i = 0; i < a.size(); ++i) {
            ASSERT_EQ(a[i], b[i]);
        }
    }

    static void relative_compare(
            const float* res,
            const float* ref,
            size_t size,
            float max_diff = 0.01f,
            const std::string assertDetails = "") {
        for (size_t i = 0lu; i < size; i++) {
            if (ref[i] == 0.f) {
                const float diff = fabs(res[i] - ref[i]);
                ASSERT_TRUE(diff < 1e-7) << "\nRelative comparison of values ref: " << ref[i] << " and res: " << res[i] << "\n" << assertDetails;
            } else {
                const float diff = fabs((res[i] - ref[i]) / ref[i]);
                ASSERT_LT(diff, max_diff) << "\nRelative comparison of values ref: " << ref[i] << " and res: " << res[i] << "\n" << assertDetails;
            }
        }
    }

    void replace(std::string& str, const std::string& from, const std::string& to) {
        std::string::size_type pos = 0;

        while((pos = str.find(from, pos)) != std::string::npos) {
            str.replace(pos, from.length(), to);
            pos += to.length();
        }
    }

    std::string replace(std::string& str, const std::string& from, const int& to) {
        replace(str, from, to_string_c_locale(to));
        return str;
    }

    std::string replace(std::string& str, const std::string& from, const size_t& to) {
        replace(str, from, to_string_c_locale(to));
        return str;
    }

    std::string replace(std::string& str, const std::string& from, const float& to) {
        replace(str, from, to_string_c_locale(to));
        return str;
    }
    // trim from both ends (in place)
    static inline std::string &trim(std::string &s) {
        s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](int c){
            return !std::isspace(c);}));
        s.erase(std::find_if(s.rbegin(), s.rend(), [](int c){
            return !std::isspace(c);}).base(), s.end());
        return s;
    }

    template <typename T, typename U>
    static inline T div_up(const T a, const U b) {
        assert(b);
        return (a + b - 1) / b;
    }
};

// Check bitness
#include <stdint.h>
#if UINTPTR_MAX == 0xffffffff
    /* 32-bit */
    #define ENVIRONMENT32
#elif UINTPTR_MAX == 0xffffffffffffffff
    /* 64-bit */
    #define ENVIRONMENT64
#else
    # error Unsupported architecture
#endif


template <typename T,typename S>
std::shared_ptr<InferenceEngine::TBlob<T>> to_tblob(const std::shared_ptr<S> &obj)
{
    return std::dynamic_pointer_cast<InferenceEngine::TBlob<T>>(obj);
}

inline InferenceEngine::InputInfo::Ptr getFirstInput(InferenceEngine::ICNNNetwork *pNet)
{
    InferenceEngine::InputsDataMap inputs;
    pNet->getInputsInfo(inputs);
    //ASSERT_GT(inputs.size(), 0);
    return inputs.begin()->second;
}
