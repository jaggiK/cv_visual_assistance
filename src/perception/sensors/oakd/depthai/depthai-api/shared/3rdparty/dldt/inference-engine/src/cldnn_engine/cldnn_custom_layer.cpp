// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cldnn_custom_layer.h"
#include "xml_parse_utils.h"
#include <description_buffer.hpp>
#include <map>
#include <fstream>
#include <streambuf>
#include <climits>

#ifdef _WIN32
# include <windows.h>
#endif

#include "simple_math.h"

using namespace InferenceEngine;
using namespace XMLParseUtils;

#define CheckAndReturnError(cond, errorMsg) \
    if (cond) { std::stringstream ss; ss << errorMsg; m_ErrorMessage = ss.str(); return; }
#define CheckNodeTypeAndReturnError(node, type) \
    CheckAndReturnError((std::string(node.name()).compare(type)), "Wrong node! expected: " << #type << " found: " << node.name())
#define CheckStrAttrAndReturnError(node, attr, value) \
    CheckAndReturnError(GetStrAttr(node, attr, "").compare(value), "Wrong attribute value! expected: " << value << " found: " << GetStrAttr(node, attr, ""))
#define CheckIntAttrAndReturnError(node, attr, value) \
    CheckAndReturnError(GetIntAttr(node, attr, -1) != (value), "Wrong attribute value! expected: " << value << " found: " << GetIntAttr(node, attr, -1))

namespace CLDNNPlugin {

void CLDNNCustomLayer::LoadSingleLayer(const pugi::xml_node & node) {
    // Root checks
    CheckNodeTypeAndReturnError(node, "CustomLayer");
    CheckStrAttrAndReturnError(node, "type", "SimpleGPU");
    CheckIntAttrAndReturnError(node, "version", 1);
    m_layerName = GetStrAttr(node, "name", "");
    CheckAndReturnError(m_layerName.length() == 0, "Missing Layer name in CustomLayer");

    // Process child nodes
    ProcessKernelNode(node.child("Kernel"));
    ProcessBuffersNode(node.child("Buffers"));
    ProcessCompilerOptionsNode(node.child("CompilerOptions"));
    ProcessWorkSizesNode(node.child("WorkSizes"));
}

void CLDNNCustomLayer::ProcessKernelNode(const pugi::xml_node & node) {
    CheckNodeTypeAndReturnError(node, "Kernel");
    CheckAndReturnError(m_kernelSource.length() > 0, "Multiple definition of Kernel");
    m_kernelEntry = GetStrAttr(node, "entry", "");
    CheckAndReturnError(m_kernelEntry.length() == 0, "No Kernel entry in layer: " << GetStrAttr(node.parent(), "name"));

    // Handle Source nodes
    for (auto sourceNode = node.child("Source"); !sourceNode.empty(); sourceNode = sourceNode.next_sibling("Source")) {
        // open file
        std::string filename = m_configDir + "/" + GetStrAttr(sourceNode, "filename", "");
        std::ifstream inputFile(filename);
        CheckAndReturnError(!inputFile.is_open(), "Couldn't open kernel file: " << filename);

        // read to string
        std::string fileContent;
        inputFile.seekg(0, std::ios::end);
        fileContent.reserve(inputFile.tellg());
        inputFile.seekg(0, std::ios::beg);

        fileContent.assign((std::istreambuf_iterator<char>(inputFile)),
            std::istreambuf_iterator<char>());

        // append to source string
        m_kernelSource.append("\n// Custom Layer Kernel " + filename + "\n\n");
        m_kernelSource.append(fileContent);
    }

    // Handle Define nodes
    for (auto defineNode = node.child("Define"); !defineNode.empty(); defineNode = defineNode.next_sibling("Define")) {
        KernelDefine kd;
        kd.name = GetStrAttr(defineNode, "name", "");
        CheckAndReturnError((kd.name.length() == 0), "Missing name for define node");
        kd.param = GetStrAttr(defineNode, "param", "");
        kd.default_value = GetStrAttr(defineNode, "default", "");
        std::string type = GetStrAttr(defineNode, "type", "");
        if (type.compare("int[]") == 0 || type.compare("float[]") == 0) {
            kd.prefix = "(" + type + ") {";
            kd.postfix = "}";
        }
        m_defines.push_back(kd);
    }
}

void CLDNNCustomLayer::ProcessBuffersNode(const pugi::xml_node & node) {
    CheckNodeTypeAndReturnError(node, "Buffers");
    for (auto tensorNode = node.child("Tensor"); !tensorNode.empty(); tensorNode = tensorNode.next_sibling("Tensor")) {
        KerenlParam kp;
        kp.format = FormatFromString(GetStrAttr(tensorNode, "format", "BFYX"));
        CheckAndReturnError(kp.format == cldnn::format::format_num, "Tensor node has an invalid format: " << GetStrAttr(tensorNode, "format"));
        kp.paramIndex = GetIntAttr(tensorNode, "arg-index", -1);
        CheckAndReturnError(kp.paramIndex == -1, "Tensor node has no arg-index");
        kp.portIndex = GetIntAttr(tensorNode, "port-index", -1);
        CheckAndReturnError(kp.portIndex == -1, "Tensor node has no port-index");
        std::string typeStr = GetStrAttr(tensorNode, "type");
        if (typeStr.compare("input") == 0) {
            kp.type = ParamType::Input;
        } else if (typeStr.compare("output") == 0) {
            kp.type = ParamType::Output;
        } else {
            CheckAndReturnError(true, "Tensor node has an invalid type: " << typeStr);
        }
        m_kernelParams.push_back(kp);
    }
    for (auto dataNode = node.child("Data"); !dataNode.empty(); dataNode = dataNode.next_sibling("Data")) {
        KerenlParam kp;
        kp.type = ParamType::Data;
        kp.paramIndex = GetIntAttr(dataNode, "arg-index", -1);
        CheckAndReturnError(kp.paramIndex == -1, "Data node has no arg-index");
        kp.blobName = GetStrAttr(dataNode, "name", "");
        CheckAndReturnError(kp.blobName.empty(), "Data node has no name");
        m_kernelParams.push_back(kp);
    }
}

void CLDNNCustomLayer::ProcessCompilerOptionsNode(const pugi::xml_node & node) {
    if (node.empty()) {
        return;  // Optional node doesn't exist
    }
    CheckNodeTypeAndReturnError(node, "CompilerOptions");
    CheckAndReturnError(m_compilerOptions.length() > 0, "Multiple definition of CompilerOptions");
    m_compilerOptions = GetStrAttr(node, "options", "");
}

void CLDNNCustomLayer::ProcessWorkSizesNode(const pugi::xml_node & node) {
    if (node.empty()) {
        return;  // Optional node doesn't exist
    }
    CheckNodeTypeAndReturnError(node, "WorkSizes");

    m_wgDimInputIdx = -1;
    std::string dim_src_string = node.attribute("dim").as_string("");
    if (!dim_src_string.empty() && "output" != dim_src_string) {
        // try to locate index separator
        auto pos = dim_src_string.find_first_of(',');
        auto flag = dim_src_string.substr(0, pos);
        CheckAndReturnError(("input" != flag), "Invalid WG dim source: " << flag);

        int input_idx = 0;
        if (pos != std::string::npos) {
            // user explicitly set input index in config
            auto input_idx_string = dim_src_string.substr(pos + 1, std::string::npos);
            input_idx = std::stoi(input_idx_string);
        }
        CheckAndReturnError((input_idx < 0), "Invalid input tensor index: " << input_idx);
        m_wgDimInputIdx = input_idx;
    }

    std::string gws = node.attribute("global").as_string("");
    while (!gws.empty()) {
        auto pos = gws.find_first_of(',');
        auto rule = gws.substr(0, pos);
        CheckAndReturnError(!IsLegalSizeRule(rule), "Invalid WorkSize: " << rule);
        m_globalSizeRules.push_back(rule);
        if (pos == std::string::npos) {
            gws.clear();
        } else {
            gws = gws.substr(pos + 1, std::string::npos);
        }
    }

    std::string lws = node.attribute("local").as_string("");
    while (!lws.empty()) {
        auto pos = lws.find_first_of(',');
        auto rule = lws.substr(0, pos);
        CheckAndReturnError(!IsLegalSizeRule(rule), "Invalid WorkSize: " << rule);
        m_localSizeRules.push_back(rule);
        if (pos == std::string::npos) {
            lws.clear();
        } else {
            lws = lws.substr(pos + 1, std::string::npos);
        }
    }
}

bool CLDNNCustomLayer::IsLegalSizeRule(const std::string & rule) {
    SimpleMathExpression expr;
    expr.SetVariables({
        { 'b', 1 }, { 'B', 1 },
        { 'f', 1 }, { 'F', 1 },
        { 'y', 1 }, { 'Y', 1 },
        { 'x', 1 }, { 'X', 1 },
    });
    if (!expr.SetExpression(rule)) {
        return false;
    }

    try {
        expr.Evaluate();
    } catch (...) {
        return false;
    }
    return true;
}

cldnn::format CLDNNCustomLayer::FormatFromString(const std::string & str) {
    static const std::map<std::string, cldnn::format> FormatNameToType = {
        { "BFYX" , cldnn::format::bfyx },
        { "bfyx" , cldnn::format::bfyx },

        { "BYXF" , cldnn::format::byxf },
        { "byxf" , cldnn::format::byxf },

        { "FYXB" , cldnn::format::fyxb },
        { "fyxb" , cldnn::format::fyxb },

        { "YXFB" , cldnn::format::yxfb },
        { "yxfb" , cldnn::format::yxfb },

        { "ANY" , cldnn::format::any },
        { "any" , cldnn::format::any },
    };
    auto it = FormatNameToType.find(str);
    if (it != FormatNameToType.end())
        return it->second;
    else
        return cldnn::format::format_num;
}

void CLDNNCustomLayer::LoadFromFile(const std::string configFile, CLDNNCustomLayerMap& customLayers, bool can_be_missed) {
    pugi::xml_document xmlDoc;
    pugi::xml_parse_result res = xmlDoc.load_file(configFile.c_str());
    if (res.status != pugi::status_ok) {
        if (can_be_missed) {
            // config file might not exist - like global config, for example
            return;
        } else {
            THROW_IE_EXCEPTION << "Error loading custom layer configuration file: " << configFile << ", " << res.description()
                << " at offset " << res.offset;
        }
    }

#ifdef _WIN32
    char path[MAX_PATH];
    char* abs_path_ptr = _fullpath(path, configFile.c_str(), MAX_PATH);
#elif __linux__
    char path[PATH_MAX];
    char* abs_path_ptr = realpath(configFile.c_str(), path);
#endif
    if (abs_path_ptr == nullptr) {
        THROW_IE_EXCEPTION << "Error loading custom layer configuration file: " << configFile << ", "
                           << "Can't get canonicalized absolute pathname.";
    }

    std::string abs_file_name(path);
    // try extracting directory from config path
    std::string dir_path;
    std::size_t dir_split_pos = abs_file_name.find_last_of("/\\");
    std::size_t colon_pos = abs_file_name.find_first_of(":");
    std::size_t first_slash_pos = abs_file_name.find_first_of("/");

    if (dir_split_pos != std::string::npos &&
       (colon_pos != std::string::npos || first_slash_pos == 0)) {
        // path is absolute
        dir_path = abs_file_name.substr(0, dir_split_pos);
    } else {
        THROW_IE_EXCEPTION << "Error loading custom layer configuration file: " << configFile << ", "
                           << "Path is not valid";
    }

    for (auto r = xmlDoc.document_element(); r; r = r.next_sibling()) {
        CLDNNCustomLayerPtr layer = std::make_shared<CLDNNCustomLayer>(CLDNNCustomLayer(dir_path));
        layer->LoadSingleLayer(r);
        if (layer->Error()) {
            customLayers.clear();
            THROW_IE_EXCEPTION << layer->m_ErrorMessage;
        } else {
            customLayers[layer->Name()] = layer;
        }
    }
}

};  // namespace CLDNNPlugin
