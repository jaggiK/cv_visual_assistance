/*
// Copyright (c) 2016-2018 Intel Corporation
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
#include "error_handler.h"
#include <string>
#include <vector>

namespace cldnn {

void err_details::cldnn_print_error_message(const std::string& file,
                                            int line,
                                            const std::string& instance_id,
                                            std::stringstream& msg,
                                            const std::string& add_msg) {
    {
        std::stringstream source_of_error;
        source_of_error << file << " at line: " << line << std::endl;
        source_of_error << "Error has occured for: " << instance_id << std::endl;

        std::stringstream addidtional_message;
        if (!add_msg.empty()) {
            addidtional_message << add_msg << std::endl;
        }

        throw std::invalid_argument(source_of_error.str() + msg.str() + addidtional_message.str());
    }
}

void error_message(const std::string& file, int line, const std::string& instance_id, const std::string& message) {
    std::stringstream error_msg;
    error_msg << message << std::endl;
    err_details::cldnn_print_error_message(file, line, instance_id, error_msg);
}

void error_on_not_supported_fp16(const std::string& file,
                                 int line,
                                 const std::string& instance_id,
                                 uint8_t supp_fp16,
                                 bool fp16_used) {
    if (!supp_fp16 && fp16_used) {
        std::stringstream error_msg;
        error_msg << "GPU device does not support half precision floating-point formats (cl_khr_fp16 extension)"
                  << std::endl;
        err_details::cldnn_print_error_message(file, line, instance_id, error_msg);
    }
}

void error_on_bool(const std::string& file,
                   int line,
                   const std::string& instance_id,
                   const std::string& condition_id,
                   bool condition,
                   const std::string& additional_message) {
    if (condition) {
        std::stringstream error_msg;
        auto condition_to_string = [](const bool& condi) -> std::string { return condi ? "true" : "false"; };
        error_msg << condition_id << "(" << condition_to_string(condition) << ") should be "
                  << condition_to_string(!condition) << std::endl;
        err_details::cldnn_print_error_message(file, line, instance_id, error_msg, additional_message);
    }
}

void error_on_mismatching_data_types(const std::string& file,
                                     int line,
                                     const std::string& instance_id,
                                     const std::string& data_format_1_id,
                                     data_types data_format_1,
                                     const std::string& data_format_2_id,
                                     data_types data_format_2,
                                     const std::string& additional_message,
                                     bool ignore_sign) {
    if (data_format_1 != data_format_2 && !ignore_sign &&
        ((data_format_1 == data_types::i8 && data_format_2 == data_types::u8) ||
         (data_format_1 == data_types::u8 && data_format_2 == data_types::i8))) {
        std::stringstream error_msg;
        error_msg << "Data formats are incompatible." << std::endl;
        error_msg << data_format_1_id << " format is: " << data_type_traits::name(data_format_1) << ", "
                  << data_format_2_id << " is: " << data_type_traits::name(data_format_2) << std::endl;
        error_msg << "Data formats should be the same!" << std::endl;
        err_details::cldnn_print_error_message(file, line, instance_id, error_msg, additional_message);
    }
}

void error_on_tensor_dims_less_than_other_tensor_dims(const std::string& file,
                                                      int line,
                                                      const std::string& instance_id,
                                                      const std::string& tensor_id,
                                                      const tensor& tens,
                                                      const std::string& tensor_to_compare_to_id,
                                                      const tensor& tens_to_compre,
                                                      const std::string& additional_message) {
    std::vector<std::string> errors;
    if (tens.batch[0] < tens_to_compre.batch[0]) {
        errors.push_back("Batch");
    }
    if (tens.feature[0] < tens_to_compre.feature[0]) {
        errors.push_back("Feature");
    }
    if (tens.spatial[0] < tens_to_compre.spatial[0]) {
        errors.push_back("Spatial x");
    }
    if (tens.spatial[1] < tens_to_compre.spatial[1]) {
        errors.push_back("Spatial y");
    }

    if (!errors.empty()) {
        std::stringstream error_msg;
        error_msg << tensor_id << " sizes: " << tens << std::endl;
        error_msg << tensor_to_compare_to_id << " sizes: " << tens_to_compre << std::endl;
        error_msg << "All " << tensor_id << " dimensions should not be less than " << tensor_to_compare_to_id
                  << " dimensions." << std::endl;
        error_msg << "Mismatching dimensions: ";
        for (size_t i = 0; i < errors.size(); i++) {
            error_msg << errors.at(i) << std::endl;
        }
        err_details::cldnn_print_error_message(file, line, instance_id, error_msg, additional_message);
    }
}

void error_on_tensor_dims_greater_than_other_tensor_dims(const std::string& file,
                                                         int line,
                                                         const std::string& instance_id,
                                                         const std::string& tensor_id,
                                                         const tensor& tens,
                                                         const std::string& tensor_to_compare_to_id,
                                                         const tensor& tens_to_compre,
                                                         const std::string& additional_message) {
    std::vector<std::string> errors;
    if (tens.batch[0] > tens_to_compre.batch[0]) {
        errors.push_back("Batch");
    }
    if (tens.feature[0] > tens_to_compre.feature[0]) {
        errors.push_back("Feature");
    }
    if (tens.spatial[0] > tens_to_compre.spatial[0]) {
        errors.push_back("Spatial x");
    }
    if (tens.spatial[1] > tens_to_compre.spatial[1]) {
        errors.push_back("Spatial y");
    }

    if (!errors.empty()) {
        std::stringstream error_msg;
        error_msg << tensor_id << " sizes: " << tens << std::endl;
        error_msg << tensor_to_compare_to_id << " sizes: " << tens_to_compre << std::endl;
        error_msg << "All " << tensor_id << " dimensions should not be greater than " << tensor_to_compare_to_id
                  << std::endl;
        error_msg << "Mismatching dimensions: ";
        for (size_t i = 0; i < errors.size(); i++) {
            error_msg << errors.at(i) << std::endl;
        }
        err_details::cldnn_print_error_message(file, line, instance_id, error_msg, additional_message);
    }
}

void error_on_tensor_dims_not_dividable_by_other_tensor_dims(const std::string& file,
                                                             int line,
                                                             const std::string& instance_id,
                                                             const std::string& tensor_id,
                                                             const tensor& tens,
                                                             const std::string& tensor_to_compare_to_id,
                                                             const tensor& tens_to_compre,
                                                             const std::string& additional_message) {
    std::vector<std::string> errors;
    if (tens.batch[0] % tens_to_compre.batch[0] != 0) {
        errors.push_back("Batch");
    }
    if (tens.feature[0] % tens_to_compre.feature[0] != 0) {
        errors.push_back("Feature");
    }
    if (tens.spatial[0] % tens_to_compre.spatial[0] != 0) {
        errors.push_back("Spatial x");
    }
    if (tens.spatial[1] % tens_to_compre.spatial[1] != 0) {
        errors.push_back("Spatial y");
    }

    if (!errors.empty()) {
        std::stringstream error_msg;
        error_msg << tensor_id << " sizes: " << tens << std::endl;
        error_msg << tensor_to_compare_to_id << " sizes: " << tens_to_compre << std::endl;
        error_msg << "All " << tensor_id << " dimensions must be dividable by corresponding dimensions from "
                  << tensor_to_compare_to_id << std::endl;
        error_msg << "Mismatching dimensions: ";
        for (size_t i = 0; i < errors.size(); i++) {
            error_msg << errors.at(i) << std::endl;
        }
        err_details::cldnn_print_error_message(file, line, instance_id, error_msg, additional_message);
    }
}

void error_on_mismatch_layout(const std::string& file,
                              int line,
                              const std::string& instance_id,
                              const std::string& layout_1_id,
                              const layout& layout_1,
                              const std::string& layout_2_id,
                              const layout& layout_2,
                              const std::string& additional_message) {
    if (layout_1 != layout_2) {
        std::stringstream error_msg;
        error_msg << "Layouts mismatch." << std::endl;

        if (layout_1.data_padding != layout_2.data_padding) {
            error_msg << layout_1_id << " data padding mismatch: " << layout_2_id << " data padding." << std::endl;
            error_msg << layout_1_id << " upper data padding: " << layout_1.data_padding.upper_size() << ", "
                      << layout_2_id << " upper data padding: " << layout_2.data_padding.upper_size() << std::endl;
            error_msg << layout_1_id << " lower data padding: " << layout_1.data_padding.lower_size() << ", "
                      << layout_2_id << " lower data padding: " << layout_2.data_padding.lower_size() << std::endl;
        }
        if (layout_1.data_type != layout_2.data_type) {
            error_msg << layout_1_id << " data type mismatch: " << layout_2_id << " data type." << std::endl;
            error_msg << layout_1_id << " data type: " << data_type_traits::name(layout_1.data_type) << ", "
                      << layout_2_id << " data type: " << data_type_traits::name(layout_2.data_type) << std::endl;
        }
        if (layout_1.format != layout_2.format) {
            error_msg << layout_1_id << " format mismatch: " << layout_2_id << " format." << std::endl;
            error_msg << layout_1_id << " format: " << format::traits(layout_1.format).order << ", " << layout_2_id
                      << " format: " << format::traits(layout_2.format).order << std::endl;
        }
        if (layout_1.size != layout_2.size) {
            error_msg << layout_1_id << " size mismatch : " << layout_2_id << " size." << std::endl;
            error_msg << layout_1_id << " size: " << layout_1.size << ", " << layout_2_id << " size: " << layout_2.size
                      << std::endl;
        }
        err_details::cldnn_print_error_message(file, line, instance_id, error_msg, additional_message);
    }
}

}  // namespace cldnn
