// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cstdlib>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <unordered_map>
#include <map>
#include <vector>
#include <string>

#include <gflags/gflags.h>

#include "inference_engine.hpp"
#include <vpu/private_plugin_config.hpp>
#include <vpu/utils/string.hpp>
#include "samples/common.hpp"
#include <dlia/dlia_config.hpp>

static constexpr char help_message[] = "Optional. Print the usage message.";
static constexpr char model_message[] = "Required. Path to the XML model.";
static constexpr char targetDeviceMessage[] = "Required. Specify a target device for which executable network will be compiled."
                                                "Use \"-d HETERO:<comma-separated_devices_list>\" format to specify HETERO plugin. "
                                                "Use \"-d MULTI:<comma-separated_devices_list>\" format to specify MULTI plugin. "
                                                "The application looks for a suitable plugin for the specified device.";

static constexpr char output_message[] = "Optional. Path to the output file. Default value: \"<model_xml_file>.blob\".";
static constexpr char config_message[] = "Optional. Path to the configuration file. Default value: \"config\".";
static constexpr char platform_message[] = "Optional. Specifies Movidius platform."
                                           " Supported values: VPU_MYRIAD_2450, VPU_MYRIAD_2480."
                                           " Overwrites value from config.\n"
"                                             This option must be used in order to compile blob"
                                           " without a connected Myriad device.";
static constexpr char number_of_shaves_message[] = "Optional. Specifies number of shaves."
                                                   " Should be set with \"VPU_NUMBER_OF_CMX_SLICES\"."
                                                   " Overwrites value from config.";
static constexpr char number_of_cmx_slices_message[] = "Optional. Specifies number of CMX slices."
                                                       " Should be set with \"VPU_NUMBER_OF_SHAVES\"."
                                                       " Overwrites value from config.";
static constexpr char inputs_precision_message[] = "Optional. Specifies precision for all input layers of the network."
                                                   " Supported values: FP32, FP16, U8. Default value: FP16.";
static constexpr char outputs_precision_message[] = "Optional. Specifies precision for all output layers of the network."
                                                    " Supported values: FP32, FP16, U8. Default value: FP16.";
static constexpr char iop_message[] = "Optional. Specifies precision for input and output layers by name.\n"
"                                             By default, all inputs and outputs have the FP16 precision.\n"
"                                             Available precisions: FP32, FP16, U8.\n"
"                                             Example: -iop \"input:FP16, output:FP16\".\n"
"                                             Notice that quotes are required.\n"
"                                             Overwrites precision from ip and op options for specified layers.";

static constexpr char dla_arch_name[] = "Optional. Specify architecture name used to compile executable network for FPGA device.";

DEFINE_bool(h, false, help_message);
DEFINE_string(m, "", model_message);
DEFINE_string(d, "", targetDeviceMessage);
DEFINE_string(o, "", output_message);
DEFINE_string(c, "config", config_message);
DEFINE_string(ip, "", inputs_precision_message);
DEFINE_string(op, "", outputs_precision_message);
DEFINE_string(iop, "", iop_message);
DEFINE_string(VPU_MYRIAD_PLATFORM, "", platform_message);
DEFINE_string(VPU_NUMBER_OF_SHAVES, "", number_of_shaves_message);
DEFINE_string(VPU_NUMBER_OF_CMX_SLICES, "", number_of_cmx_slices_message);
DEFINE_string(DLA_ARCH_NAME, "", dla_arch_name);

static void showUsage() {
    std::cout << std::endl;
    std::cout << "compile_tool [OPTIONS]" << std::endl;
    std::cout << "[OPTIONS]:" << std::endl;
    std::cout << "    -h                                       "   << help_message                 << std::endl;
    std::cout << "    -m                           <value>     "   << model_message                << std::endl;
    std::cout << "    -d                           <value>     "   << targetDeviceMessage          << std::endl;
    std::cout << "    -o                           <value>     "   << output_message               << std::endl;
    std::cout << "    -c                           <value>     "   << config_message               << std::endl;
    std::cout << "    -ip                          <value>     "   << inputs_precision_message     << std::endl;
    std::cout << "    -op                          <value>     "   << outputs_precision_message    << std::endl;
    std::cout << "    -iop                        \"<value>\"    " << iop_message                  << std::endl;
    std::cout << "                                             "                                   << std::endl;
    std::cout << "    VPU options:                             "                                   << std::endl;
    std::cout << "      -VPU_MYRIAD_PLATFORM       <value>     "   << platform_message             << std::endl;
    std::cout << "      -VPU_NUMBER_OF_SHAVES      <value>     "   << number_of_shaves_message     << std::endl;
    std::cout << "      -VPU_NUMBER_OF_CMX_SLICES  <value>     "   << number_of_cmx_slices_message << std::endl;
    std::cout << "    DLA options:                             "                                   << std::endl;
    std::cout << "      -DLA_ARCH_NAME             <value>     "   << dla_arch_name                << std::endl;
    std::cout << std::endl;
}

static bool parseCommandLine(int *argc, char ***argv, InferenceEngine::Core& ie) {
    gflags::ParseCommandLineNonHelpFlags(argc, argv, true);

    if (FLAGS_h) {
        showUsage();
        return false;
    }

    if (FLAGS_m.empty()) {
        throw std::invalid_argument("Path to model xml file is required");
    }

    if (FLAGS_d.empty()) {
        throw std::invalid_argument("Target device name is required");
    }

    if (std::string::npos != FLAGS_d.find("MYRIAD") && FLAGS_VPU_MYRIAD_PLATFORM.empty()) {
        std::vector<std::string> myriadDeviceIds = ie.GetMetric("MYRIAD", METRIC_KEY(AVAILABLE_DEVICES));
        if (myriadDeviceIds.empty()) {
            throw std::runtime_error{"No available MYRIAD devices. Please specify -VPU_MYRIAD_PLATFORM option explicitly"};
        }
    }

    if (1 < *argc) {
        std::stringstream message;
        message << "Unknown arguments: ";
        for (auto arg = 1; arg < *argc; arg++) {
            message << argv[arg];
            if (arg < *argc) {
                message << " ";
            }
        }
        throw std::invalid_argument(message.str());
    }

    return true;
}

static std::map<std::string, std::string> parseConfig(const std::string& configName, char comment = '#') {
    std::map<std::string, std::string> config;
    std::ifstream file{configName};
    if (file.is_open()) {
        std::string key, value;
        while (file >> key >> value) {
            if (key.empty() || key[0] == comment) {
                continue;
            }
            config[key] = value;
        }
    }
    return config;
}

static std::map<std::string, std::string> configure(const std::string &configFile, const std::string &xmlFileName) {
    auto config = parseConfig(configFile);

    if (!FLAGS_VPU_MYRIAD_PLATFORM.empty()) {
        config[VPU_MYRIAD_CONFIG_KEY(PLATFORM)] = FLAGS_VPU_MYRIAD_PLATFORM;
    }

    if (!FLAGS_VPU_NUMBER_OF_SHAVES.empty()) {
        config[VPU_CONFIG_KEY(NUMBER_OF_SHAVES)] = FLAGS_VPU_NUMBER_OF_SHAVES;
    }

    if (!FLAGS_VPU_NUMBER_OF_CMX_SLICES.empty()) {
        config[VPU_CONFIG_KEY(NUMBER_OF_CMX_SLICES)] = FLAGS_VPU_NUMBER_OF_CMX_SLICES;
    }

    if (!FLAGS_DLA_ARCH_NAME.empty()) {
        config[DLIA_CONFIG_KEY(ARCH_NAME)] = FLAGS_DLA_ARCH_NAME;
    }

    return config;
}

static std::map<std::string, std::string> parsePrecisions(const std::string &iop) {
    std::string user_input = iop;
    user_input.erase(std::remove_if(user_input.begin(), user_input.end(), ::isspace), user_input.end());

    std::vector<std::string> inputs;
    vpu::splitStringList(user_input, inputs, ',');

    std::map<std::string, std::string> precisions;
    for (auto &&input : inputs) {
        std::vector<std::string> precision;
        vpu::splitStringList(input, precision, ':');
        if (precision.size() != 2) {
            throw std::invalid_argument("Invalid precision " + input + ". Expected layer_name : precision_value");
        }

        precisions[precision[0]] = precision[1];
    }

    return precisions;
}

using supported_precisions_t = std::unordered_map<std::string, InferenceEngine::Precision>;

static InferenceEngine::Precision getPrecision(const std::string &value,
                                               const supported_precisions_t &supported_precisions,
                                               const std::string& error_report = std::string()) {
    std::string upper_value = value;
    std::transform(value.begin(), value.end(), upper_value.begin(), ::toupper);
    auto precision = supported_precisions.find(upper_value);
    if (precision == supported_precisions.end()) {
        std::string report = error_report.empty() ? ("") : (" " + error_report);
        throw std::logic_error("\"" + value + "\"" + " is not a valid precision" + report);
    }

    return precision->second;
}

static InferenceEngine::Precision getInputPrecision(const std::string &value) {
    static const supported_precisions_t supported_precisions = {
         { "FP32", InferenceEngine::Precision::FP32 },
         { "FP16", InferenceEngine::Precision::FP16 },
         { "U8", InferenceEngine::Precision::U8 }
    };
    return getPrecision(value, supported_precisions, "for input layer");
}

static InferenceEngine::Precision getOutputPrecision(const std::string &value) {
    static const supported_precisions_t supported_precisions = {
         { "FP32", InferenceEngine::Precision::FP32 },
         { "FP16", InferenceEngine::Precision::FP16 }
    };
    return getPrecision(value, supported_precisions, "for output layer");
}

bool isFP16(InferenceEngine::Precision precision) {
    return precision == InferenceEngine::Precision::FP16;
}

bool isFP32(InferenceEngine::Precision precision) {
    return precision == InferenceEngine::Precision::FP32;
}

bool isU8(InferenceEngine::Precision precision) {
    return precision == InferenceEngine::Precision::U8;
}

bool isFloat(InferenceEngine::Precision precision) {
    return isFP16(precision) || isFP32(precision);
}

static void setPrecisions(const InferenceEngine::CNNNetwork &network, const std::string &iop) {
    auto user_precisions_map = parsePrecisions(iop);
    auto inputs = network.getInputsInfo();
    auto outputs = network.getOutputsInfo();

    for (auto &&item : user_precisions_map) {
        std::string layer_name = item.first;
        std::string user_precision = item.second;

        auto input = inputs.find(layer_name);
        auto output = outputs.find(layer_name);

        if (input != inputs.end()) {
            const auto input_precision = input->second->getPrecision();
            if ((isFloat(input_precision) && isFloat(getInputPrecision(user_precision))) ||
                (isFP16(input_precision) && isU8(getInputPrecision(user_precision)))) {
                input->second->setPrecision(getInputPrecision(user_precision));
            }
        } else if (output != outputs.end()) {
            const auto output_precision = output->second->getPrecision();
            if (isFloat(output_precision) && isFloat(getOutputPrecision(user_precision))) {
                output->second->setPrecision(getOutputPrecision(user_precision));
            }
        } else {
            throw std::logic_error(layer_name + " is not an input neither output");
        }
    }
}

static void setDefaultIOPrecisions(InferenceEngine::CNNNetwork &network, const std::string & device) {
    bool isMyriad = FLAGS_d.find("MYRIAD") != std::string::npos;

    if (isMyriad) {
        const InferenceEngine::Precision fp16 = InferenceEngine::Precision::FP16;

        for (auto &&layer : network.getInputsInfo()) {
            if (isFP32(layer.second->getPrecision())) {
                layer.second->setPrecision(fp16);
            }
        }

        for (auto &&layer : network.getOutputsInfo()) {
            if (isFP32(layer.second->getPrecision())) {
                layer.second->setPrecision(fp16);
            }
        }
    }
}

static void processPrecisions(InferenceEngine::CNNNetwork &network,
                              const std::string &inputs_precision, const std::string &outputs_precision,
                              const std::string &iop) {
    if (!inputs_precision.empty()) {
        auto precision = getInputPrecision(inputs_precision);
        for (auto &&layer : network.getInputsInfo()) {
            const auto layerPrecision = layer.second->getPrecision();
            if ((isFloat(layerPrecision) && isFloat(precision)) ||
                (isFP16(layerPrecision) && isU8(precision))) {
                layer.second->setPrecision(precision);
            }
        }
    }

    if (!outputs_precision.empty()) {
        auto precision = getOutputPrecision(outputs_precision);
        for (auto &&layer : network.getOutputsInfo()) {
            const auto layerPrecision = layer.second->getPrecision();
            if (isFloat(layerPrecision) && isFloat(precision)) {
                layer.second->setPrecision(precision);
            }
        }
    }

    if (!iop.empty()) {
        setPrecisions(network, iop);
    }
}

std::string getFileNameFromPath(const std::string& path,
#if defined(_WIN32)
                                const std::string sep = "\\") {
#else
                                const std::string sep = "/") {
#endif
    auto pos = path.rfind(sep);
    if (std::string::npos == pos) {
        return path;
    } else {
        return path.substr(pos + 1);
    }
}

int main(int argc, char *argv[]) {
    try {
        std::cout << "Inference Engine: " << InferenceEngine::GetInferenceEngineVersion() << std::endl;

        InferenceEngine::Core ie;

        if (!parseCommandLine(&argc, &argv, ie)) {
            return EXIT_SUCCESS;
        }

        auto network = ie.ReadNetwork(FLAGS_m);

        setDefaultIOPrecisions(network, FLAGS_d);
        processPrecisions(network, FLAGS_ip, FLAGS_op, FLAGS_iop);

        auto executableNetwork = ie.LoadNetwork(network, FLAGS_d, configure(FLAGS_c, FLAGS_m));

        std::string outputName = FLAGS_o;
        if (outputName.empty()) {
            outputName = getFileNameFromPath(fileNameNoExt(FLAGS_m)) + ".blob";
        }
        std::ofstream outputFile{outputName};
        executableNetwork.Export(outputFile);
    } catch (const std::exception &error) {
        std::cerr << error.what() << std::endl;
        return EXIT_FAILURE;
    } catch (...) {
        std::cerr << "Unknown/internal exception happened." << std::endl;
        return EXIT_FAILURE;
    }

    std::cout << "Done" << std::endl;
    return EXIT_SUCCESS;
}
