// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ie_util_internal.hpp"

#include <ie_layers.h>

#include <cassert>
#include <deque>
#include <iomanip>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "cpp/ie_plugin_cpp.hpp"
#include "details/caseless.hpp"
#include "details/ie_cnn_network_tools.h"
#include "details/os/os_filesystem.hpp"
#include "file_utils.h"
#include "graph_tools.hpp"
#include "ie_icnn_network_stats.hpp"
#include "net_pass.h"
#include "precision_utils.h"

using std::string;

namespace InferenceEngine {

std::exception_ptr& CurrentException() {
    static thread_local std::exception_ptr currentException = nullptr;
    return currentException;
}

using namespace details;

DataPtr cloneData(const InferenceEngine::Data& source) {
    auto cloned = std::make_shared<InferenceEngine::Data>(source);
    if (cloned != nullptr) {
        cloned->getCreatorLayer().reset();
        cloned->getInputTo().clear();
    }
    return cloned;
}

namespace {
template <typename T>
CNNLayerPtr layerCloneImpl(const CNNLayer* source) {
    auto layer = dynamic_cast<const T*>(source);
    if (nullptr != layer) {
        auto newLayer = std::make_shared<T>(*layer);
        newLayer->_fusedWith = nullptr;
        newLayer->outData.clear();
        newLayer->insData.clear();
        return std::static_pointer_cast<CNNLayer>(newLayer);
    }
    return nullptr;
}

/* Make this function explicit for TensorIterator layer
 * because of specific handling of the body field */
template <>
CNNLayerPtr layerCloneImpl<TensorIterator>(const CNNLayer* source) {
    auto layer = dynamic_cast<const TensorIterator*>(source);
    if (nullptr != layer) {
        auto newLayer = std::make_shared<TensorIterator>(*layer);
        newLayer->_fusedWith = nullptr;
        newLayer->outData.clear();
        newLayer->insData.clear();

        newLayer->body = NetPass::CopyTIBody(newLayer->body);

        return std::static_pointer_cast<CNNLayer>(newLayer);
    }
    return nullptr;
}

}  // namespace

CNNLayerPtr clonelayer(const CNNLayer& source) {
    using fptr = CNNLayerPtr (*)(const CNNLayer*);
    // Most derived layers must go first in this list
    static const fptr cloners[] = {&layerCloneImpl<ScatterLayer>,
                                   &layerCloneImpl<NonMaxSuppressionLayer>,
                                   &layerCloneImpl<SelectLayer>,
                                   &layerCloneImpl<BatchNormalizationLayer>,
                                   &layerCloneImpl<TopKLayer>,
                                   &layerCloneImpl<PowerLayer>,
                                   &layerCloneImpl<ScaleShiftLayer>,
                                   &layerCloneImpl<PReLULayer>,
                                   &layerCloneImpl<TileLayer>,
                                   &layerCloneImpl<ReshapeLayer>,
                                   &layerCloneImpl<CropLayer>,
                                   &layerCloneImpl<EltwiseLayer>,
                                   &layerCloneImpl<GemmLayer>,
                                   &layerCloneImpl<PadLayer>,
                                   &layerCloneImpl<GatherLayer>,
                                   &layerCloneImpl<StridedSliceLayer>,
                                   &layerCloneImpl<ShuffleChannelsLayer>,
                                   &layerCloneImpl<DepthToSpaceLayer>,
                                   &layerCloneImpl<SpaceToDepthLayer>,
                                   &layerCloneImpl<SparseFillEmptyRowsLayer>,
                                   &layerCloneImpl<SparseSegmentReduceLayer>,
                                   &layerCloneImpl<ExperimentalSparseWeightedReduceLayer>,
                                   &layerCloneImpl<SparseToDenseLayer>,
                                   &layerCloneImpl<BucketizeLayer>,
                                   &layerCloneImpl<ReverseSequenceLayer>,
                                   &layerCloneImpl<RangeLayer>,
                                   &layerCloneImpl<FillLayer>,
                                   &layerCloneImpl<BroadcastLayer>,
                                   &layerCloneImpl<MathLayer>,
                                   &layerCloneImpl<ReduceLayer>,
                                   &layerCloneImpl<ClampLayer>,
                                   &layerCloneImpl<ReLULayer>,
                                   &layerCloneImpl<SoftMaxLayer>,
                                   &layerCloneImpl<GRNLayer>,
                                   &layerCloneImpl<MVNLayer>,
                                   &layerCloneImpl<NormLayer>,
                                   &layerCloneImpl<SplitLayer>,
                                   &layerCloneImpl<ConcatLayer>,
                                   &layerCloneImpl<FullyConnectedLayer>,
                                   &layerCloneImpl<PoolingLayer>,
                                   &layerCloneImpl<DeconvolutionLayer>,
                                   &layerCloneImpl<DeformableConvolutionLayer>,
                                   &layerCloneImpl<ConvolutionLayer>,
                                   &layerCloneImpl<TensorIterator>,
                                   &layerCloneImpl<RNNSequenceLayer>,
                                   &layerCloneImpl<LSTMCell>,
                                   &layerCloneImpl<GRUCell>,
                                   &layerCloneImpl<RNNCell>,
                                   &layerCloneImpl<QuantizeLayer>,
                                   &layerCloneImpl<BinaryConvolutionLayer>,
                                   &layerCloneImpl<WeightableLayer>,
                                   &layerCloneImpl<OneHotLayer>,
                                   &layerCloneImpl<CNNLayer>,
                                   &layerCloneImpl<UniqueLayer>};
    for (auto cloner : cloners) {
        auto cloned = cloner(&source);
        if (nullptr != cloned) {
            return cloned;
        }
    }
    assert(!"All layers derived from CNNLayer so we must never get here");
    return nullptr;  // Silence "control may reach end of non-void function" warning
}

details::CNNNetworkImplPtr cloneNet(const ICNNNetwork& network) {
    std::vector<CNNLayerPtr> layers;
    details::CNNNetworkIterator i(const_cast<ICNNNetwork*>(&network));
    while (i != details::CNNNetworkIterator()) {
        layers.push_back(*i);
        i++;
    }

    InferenceEngine::ICNNNetworkStats* pstatsSrc = nullptr;
    if (StatusCode::OK != network.getStats(&pstatsSrc, nullptr)) {
        pstatsSrc = nullptr;
    }
    // copy of the network
    details::CNNNetworkImplPtr net = cloneNet(layers, pstatsSrc);
    // going over output layers and aligning output ports and outputs
    OutputsDataMap outputs;
    network.getOutputsInfo(outputs);
    OutputsDataMap outputInfo;
    net->getOutputsInfo(outputInfo);
    for (auto o : outputs) {
        auto it = outputInfo.find(o.first);
        if (it != outputInfo.end()) {
            outputInfo.erase(it);
        } else {
            net->addOutput(o.first);
        }
    }
    // remove output ports which unconnected with outputs
    for (auto o : outputInfo) {
        net->removeOutput(o.first);
    }
    net->setPrecision(network.getPrecision());
    net->setName(network.getName());

    InputsDataMap externalInputsData;
    network.getInputsInfo(externalInputsData);

    InputsDataMap clonedInputs;
    net->getInputsInfo(clonedInputs);
    for (auto&& it : externalInputsData) {
        auto inp = clonedInputs.find(it.first);
        if (inp != clonedInputs.end() && nullptr != inp->second) {
            inp->second->setPrecision(it.second->getPrecision());
            inp->second->getPreProcess() = it.second->getPreProcess();
        }
    }

    return net;
}

details::CNNNetworkImplPtr cloneNet(const std::vector<CNNLayerPtr>& layers, const ICNNNetworkStats* networkStats) {
    auto net = std::make_shared<InferenceEngine::details::CNNNetworkImpl>();

    // Src to cloned data map
    std::unordered_map<InferenceEngine::DataPtr, InferenceEngine::DataPtr> dataMap;
    // Cloned to src data map
    std::unordered_map<InferenceEngine::DataPtr, InferenceEngine::DataPtr> clonedDataMap;
    std::vector<InferenceEngine::DataPtr> clonedDatas;

    auto createDataImpl = [&](const InferenceEngine::DataPtr& data) {
        assert(nullptr != data);
        if (!contains(dataMap, data)) {
            auto clonedData = cloneData(*data);
            dataMap[data] = clonedData;
            clonedDataMap[clonedData] = data;
            clonedDatas.push_back(clonedData);
            net->getData(clonedData->getName()) = clonedData;
            return clonedData;
        }
        return dataMap[data];
    };

    auto cloneLayerImpl = [&](const CNNLayer& srcLayer) {
        CNNLayerPtr clonedLayer = clonelayer(srcLayer);
        clonedLayer->_fusedWith = nullptr;
        // We will need to reconstruct all connections in new graph
        clonedLayer->outData.clear();
        clonedLayer->insData.clear();
        net->addLayer(clonedLayer);
        return clonedLayer;
    };

    for (auto&& srcLayer : layers) {
        CNNLayerPtr clonedLayer = cloneLayerImpl(*srcLayer);
        for (auto&& src : srcLayer->insData) {
            auto data = src.lock();
            auto clonedData = createDataImpl(data);

            string inputName;
            // Find input name
            for (auto&& inp : data->getInputTo()) {
                if (srcLayer == inp.second) {
                    inputName = inp.first;
                    break;
                }
            }
            assert(!inputName.empty());
            clonedData->getInputTo().insert({inputName, clonedLayer});
            clonedLayer->insData.push_back(clonedData);
        }

        for (auto&& data : srcLayer->outData) {
            auto clonedData = createDataImpl(data);
            clonedData->getCreatorLayer() = clonedLayer;
            clonedLayer->outData.push_back(clonedData);
            for (auto&& inp : data->getInputTo()) {
                auto layer = inp.second;
                // TODO(amalyshe) is it the best place to check priorbox and remove
                // such edge from outputs?
                if (std::find(layers.begin(), layers.end(), layer) == layers.end() &&
                    !(CaselessEq<string>()(layer->type, "priorbox") ||
                      CaselessEq<string>()(layer->type, "PriorBoxClustered"))) {
                    net->addOutput(data->getName());
                    break;
                }
            }
        }
    }

    for (auto&& data : clonedDatas) {
        auto layer = data->getCreatorLayer().lock();
        // create an artificial input layer because logic in some algorithms rely
        // on existence of these layers in the network
        if (nullptr == layer) {
            assert(contains(clonedDataMap, data));
            auto originalData = clonedDataMap[data];
            assert(nullptr != originalData);

            if (auto originalLayer = originalData->getCreatorLayer().lock()) {
                if (CaselessEq<string>()(originalLayer->type, "input") ||
                    CaselessEq<string>()(originalLayer->type, "const") ||
                    CaselessEq<string>()(originalLayer->type, "memory")) {
                    layer = cloneLayerImpl(*originalLayer);
                    layer->outData.push_back(data);
                    data->getCreatorLayer() = layer;
                }
            }

            if (nullptr == layer) {
                LayerParams params;
                params.name = data->getName();
                params.precision = data->getPrecision();
                params.type = "Input";
                layer = std::make_shared<CNNLayer>(params);
                // this place should be transactional
                layer->outData.push_back(data);
                data->getCreatorLayer() = layer;
                net->addLayer(layer);
            }
        }
        if (CaselessEq<string>()(layer->type, "input")) {
            auto input = std::make_shared<InferenceEngine::InputInfo>();
            input->setInputData(data);
            net->setInputInfo(input);
        }
    }

    net->resolveOutput();

    // cloning of statistics
    InferenceEngine::ICNNNetworkStats* pstatsTarget = nullptr;
    if (networkStats != nullptr && !networkStats->isEmpty()) {
        StatusCode st = net->getStats(&pstatsTarget, nullptr);
        if (st == StatusCode::OK && pstatsTarget) {
            pstatsTarget->setNodesStats(networkStats->getNodesStats());
        }
    }

    return net;
}

struct NodePrinter {
    enum FILL_COLOR { DATA, SUPPORTED_LAYER, UNSOPPORTED_LAYER };

    std::unordered_set<InferenceEngine::Data*> printed_data;
    std::unordered_set<InferenceEngine::CNNLayer*> printed_layers;
    std::ostream& out;

    printer_callback layer_cb;

    explicit NodePrinter(std::ostream& os, printer_callback cb): out(os), layer_cb(std::move(cb)) {}

    bool isPrinted(const CNNLayerPtr& layer) {
        return static_cast<bool>(printed_layers.count(layer.get()));
    }

    bool isPrinted(const DataPtr& datum) {
        return static_cast<bool>(printed_data.count(datum.get()));
    }

    string colorToStr(FILL_COLOR color) {
        switch (color) {
        case DATA:
            return "#FCF6E3";
        case SUPPORTED_LAYER:
            return "#D9EAD3";
        case UNSOPPORTED_LAYER:
            return "#F4CCCC";
        default:
            return "#FFFFFF";
        }
    }

    string formatSize_(const std::vector<unsigned int>& spatialDims) {
        string result;
        if (spatialDims.empty()) return result;
        result = std::to_string(spatialDims[0]);
        for (auto dim : spatialDims) {
            result += "x" + std::to_string(dim);
        }
        return result;
    }

    string cleanNodeName_(string node_name) const {
        // remove dot and dash symbols from node name. It is incorrectly displayed in xdot
        node_name.erase(remove(node_name.begin(), node_name.end(), '.'), node_name.end());
        std::replace(node_name.begin(), node_name.end(), '-', '_');
        std::replace(node_name.begin(), node_name.end(), ':', '_');
        return node_name;
    }

    void printLayerNode(const CNNLayerPtr& layer) {
        auto node_name = "layer_" + cleanNodeName_(layer->name);
        printed_layers.insert(layer.get());

        ordered_properties printed_properties;

        ordered_properties node_properties = {{"shape", "box"},
                                              {"style", "filled"},
                                              {"fillcolor", colorToStr(SUPPORTED_LAYER)}};

        auto type = layer->type;
        printed_properties.emplace_back("type", type);

        if (type == "Convolution") {
            auto* conv = dynamic_cast<ConvolutionLayer*>(layer.get());

            if (conv != nullptr) {
                unsigned int depth = conv->_out_depth, group = conv->_group;

                printed_properties.emplace_back(
                    "kernel size", formatSize_({&(conv->_kernel[0]), &(conv->_kernel[conv->_kernel.size() - 1])}));
                printed_properties.emplace_back("output depth", std::to_string(depth));
                printed_properties.emplace_back("group", std::to_string(group));
                printed_properties.emplace_back(
                    "padding begin", formatSize_({&(conv->_padding[0]), &(conv->_padding[conv->_padding.size() - 1])}));
                printed_properties.emplace_back(
                    "padding end",
                    formatSize_({&(conv->_pads_end[0]), &(conv->_pads_end[conv->_pads_end.size() - 1])}));
                printed_properties.emplace_back(
                    "strides", formatSize_({&(conv->_stride[0]), &(conv->_stride[conv->_stride.size() - 1])}));
                printed_properties.emplace_back(
                    "dilations", formatSize_({&(conv->_dilation[0]), &(conv->_dilation[conv->_dilation.size() - 1])}));
            }
        } else if (type == "Pooling") {
            auto* pool = dynamic_cast<PoolingLayer*>(layer.get());

            if (pool != nullptr) {
                printed_properties.emplace_back(
                    "window size", formatSize_({&(pool->_kernel[0]), &(pool->_kernel[pool->_kernel.size() - 1])}));
                printed_properties.emplace_back(
                    "padding begin", formatSize_({&(pool->_padding[0]), &(pool->_padding[pool->_padding.size() - 1])}));
                printed_properties.emplace_back(
                    "padding end",
                    formatSize_({&(pool->_pads_end[0]), &(pool->_pads_end[pool->_pads_end.size() - 1])}));
                printed_properties.emplace_back(
                    "strides", formatSize_({&(pool->_stride[0]), &(pool->_stride[pool->_stride.size() - 1])}));
            }
        } else if (type == "ReLU") {
            auto* relu = dynamic_cast<ReLULayer*>(layer.get());

            if (relu != nullptr) {
                float negative_slope = relu->negative_slope;

                if (negative_slope != 0.0f)
                    printed_properties.emplace_back("negative_slope", std::to_string(negative_slope));
            }
        } else if (type == "Eltwise") {
            auto* eltwise = dynamic_cast<EltwiseLayer*>(layer.get());

            if (eltwise != nullptr) {
                std::string operation;

                if (eltwise->_operation == EltwiseLayer::Sum)
                    operation = "Sum";
                else if (eltwise->_operation == EltwiseLayer::Prod)
                    operation = "Prod";
                else if (eltwise->_operation == EltwiseLayer::Max)
                    operation = "Max";
                else if (eltwise->_operation == EltwiseLayer::Sub)
                    operation = "Sub";
                else if (eltwise->_operation == EltwiseLayer::Min)
                    operation = "Min";
                else if (eltwise->_operation == EltwiseLayer::Div)
                    operation = "Div";
                else if (eltwise->_operation == EltwiseLayer::Squared_diff)
                    operation = "Squared_diff";
                else if (eltwise->_operation == EltwiseLayer::Equal)
                    operation = "Equal";
                else if (eltwise->_operation == EltwiseLayer::Not_equal)
                    operation = "Not_equal";
                else if (eltwise->_operation == EltwiseLayer::Less)
                    operation = "Less";
                else if (eltwise->_operation == EltwiseLayer::Less_equal)
                    operation = "Less_equal";
                else if (eltwise->_operation == EltwiseLayer::Greater)
                    operation = "Greater";
                else if (eltwise->_operation == EltwiseLayer::Greater_equal)
                    operation = "Greater_equal";
                else if (eltwise->_operation == EltwiseLayer::Logical_NOT)
                    operation = "Logical_NOT";
                else if (eltwise->_operation == EltwiseLayer::Logical_AND)
                    operation = "Logical_AND";
                else if (eltwise->_operation == EltwiseLayer::Logical_OR)
                    operation = "Logical_OR";
                else if (eltwise->_operation == EltwiseLayer::Logical_XOR)
                    operation = "Logical_XOR";
                else if (eltwise->_operation == EltwiseLayer::Floor_mod)
                    operation = "Floor_mod";
                else if (eltwise->_operation == EltwiseLayer::Pow)
                    operation = "Pow";
                else if (eltwise->_operation == EltwiseLayer::Mean)
                    operation = "Mean";

                printed_properties.emplace_back("operation", operation);
            }
        }

        if (layer_cb != nullptr) {
            layer_cb(layer, printed_properties, node_properties);
        }

        printNode(node_name, layer->name, node_properties, printed_properties);
    }

    void printDataNode(const std::shared_ptr<Data>& data) {
        auto node_name = "data_" + cleanNodeName_(data->getName());
        printed_data.insert(data.get());

        ordered_properties printed_properties;
        ordered_properties node_properties = {{"shape", "ellipse"},
                                              {"style", "filled"},
                                              {"fillcolor", colorToStr(DATA)}};

        std::stringstream dims_ss;
        size_t idx = data->getTensorDesc().getDims().size();
        dims_ss << '[';
        for (auto& dim : data->getTensorDesc().getDims()) {
            dims_ss << dim << ((--idx) != 0u ? ", " : "");
        }
        dims_ss << ']';

        printed_properties.emplace_back("dims", dims_ss.str());
        printed_properties.emplace_back("precision", data->getPrecision().name());

        std::stringstream ss;
        ss << data->getTensorDesc().getLayout();
        printed_properties.emplace_back("layout", ss.str());
        printed_properties.emplace_back("name", data->getName());
        if (data->getCreatorLayer().lock() != nullptr)
            printed_properties.emplace_back("creator layer", data->getCreatorLayer().lock()->name);
        printNode(node_name, data->getName(), node_properties, printed_properties);
    }

    void printNode(string const& node_name, const string& node_title, ordered_properties const& node_properties,
                   ordered_properties const& printed_properties) {
        // normalization of names, removing all prohibited symbols like "/"
        string nodeNameN = node_name;
        std::replace(nodeNameN.begin(), nodeNameN.end(), '/', '_');
        string dataNameN = node_title;
        std::replace(dataNameN.begin(), dataNameN.end(), '/', '_');

        out << '\t' << nodeNameN << " [";
        for (auto& node_property : node_properties) {
            out << node_property.first << "=\"" << node_property.second << "\", ";
        }

        out << "label=\"" << node_title;
        for (auto& printed_property : printed_properties) {
            out << "\\n" << printed_property.first << ": " << printed_property.second;
        }
        out << "\"];\n";
    }

    void printEdge(const CNNLayerPtr& from_, const DataPtr& to_, bool reverse) {
        auto from_name = "layer_" + cleanNodeName_(from_->name);
        auto to_name = "data_" + cleanNodeName_(to_->getName());
        std::replace(from_name.begin(), from_name.end(), '/', '_');
        std::replace(to_name.begin(), to_name.end(), '/', '_');
        if (reverse) std::swap(from_name, to_name);
        out << '\t' << from_name << " -> " << to_name << ";\n";
    }
};

void saveGraphToDot(InferenceEngine::ICNNNetwork& network, std::ostream& out, printer_callback layer_cb) {
    NodePrinter printer(out, std::move(layer_cb));

    out << "digraph Network {\n";
    // Traverse graph and print nodes
    for (const auto& layer : details::CNNNetSortTopologically(network)) {
        printer.printLayerNode(layer);

        // Print output Data Object
        for (auto& dataptr : layer->outData) {
            if (!printer.isPrinted(dataptr)) {
                printer.printDataNode(dataptr);
            }
            printer.printEdge(layer, dataptr, false);
        }

        // Print input Data objects
        for (auto& datum : layer->insData) {
            auto dataptr = datum.lock();
            if (!printer.isPrinted(dataptr)) {
                printer.printDataNode(dataptr);
            }
            printer.printEdge(layer, dataptr, true);
        }
    }
    out << "}" << std::endl;
}

std::unordered_set<DataPtr> getRootDataObjects(ICNNNetwork& network) {
    std::unordered_set<DataPtr> ret;
    details::CNNNetworkIterator i(&network);
    while (i != details::CNNNetworkIterator()) {
        CNNLayer::Ptr layer = *i;

        // TODO: Data without creatorLayer
        if (CaselessEq<string>()(layer->type, "input") || CaselessEq<string>()(layer->type, "const") ||
            CaselessEq<string>()(layer->type, "memory")) {
            ret.insert(layer->outData.begin(), layer->outData.end());
        }
        i++;
    }
    return ret;
}

namespace {

template <typename C, typename = InferenceEngine::details::enableIfSupportedChar<C> >
std::basic_string<C> getPathName(const std::basic_string<C>& s) {
    size_t i = s.rfind(FileUtils::FileTraits<C>::FileSeparator, s.length());
    if (i != std::string::npos) {
        return (s.substr(0, i));
    }

    return {};
}

}  // namespace

#ifndef _WIN32

static std::string getIELibraryPathUnix() {
    Dl_info info;
    dladdr(reinterpret_cast<void*>(getIELibraryPath), &info);
    return getPathName(std::string(info.dli_fname)).c_str();
}

#endif  // _WIN32

#ifdef ENABLE_UNICODE_PATH_SUPPORT

std::wstring getIELibraryPathW() {
#if defined(_WIN32) || defined(_WIN64)
    wchar_t ie_library_path[4096];
    HMODULE hm = NULL;
    if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                            (LPCWSTR)getIELibraryPath, &hm)) {
        THROW_IE_EXCEPTION << "GetModuleHandle returned " << GetLastError();
    }
    GetModuleFileNameW(hm, (LPWSTR)ie_library_path, sizeof(ie_library_path));
    return getPathName(std::wstring(ie_library_path));
#else
    Dl_info info;
    dladdr(reinterpret_cast<void*>(getIELibraryPath), &info);
    return details::multiByteCharToWString(getIELibraryPathUnix().c_str());
#endif
}

#endif

std::string getIELibraryPath() {
#ifdef ENABLE_UNICODE_PATH_SUPPORT
    return details::wStringtoMBCSstringChar(getIELibraryPathW());
#else
    return getIELibraryPathUnix();
#endif
}

}  // namespace InferenceEngine
