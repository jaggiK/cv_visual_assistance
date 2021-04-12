// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ie_api_impl.hpp"
#include "hetero/hetero_plugin_config.hpp"
#include "ie_iinfer_request.hpp"
#include "details/ie_cnn_network_tools.h"

const std::string EXPORTED_NETWORK_NAME = "undefined";
std::map <std::string, InferenceEngine::Precision> precision_map = {{"FP32", InferenceEngine::Precision::FP32},
                                                                    {"FP16", InferenceEngine::Precision::FP16},
                                                                    {"I8",   InferenceEngine::Precision::I8},
                                                                    {"I16",  InferenceEngine::Precision::I16},
                                                                    {"I32",  InferenceEngine::Precision::I32},
                                                                    {"I64",  InferenceEngine::Precision::I64},
                                                                    {"U16",  InferenceEngine::Precision::U16},
                                                                    {"U8",   InferenceEngine::Precision::U8}};

std::map <std::string, InferenceEngine::Layout> layout_map = {{"ANY",     InferenceEngine::Layout::ANY},
                                                              {"NCHW",    InferenceEngine::Layout::NCHW},
                                                              {"NHWC",    InferenceEngine::Layout::NHWC},
                                                              {"OIHW",    InferenceEngine::Layout::OIHW},
                                                              {"C",       InferenceEngine::Layout::C},
                                                              {"CHW",     InferenceEngine::Layout::CHW},
                                                              {"HW",      InferenceEngine::Layout::HW},
                                                              {"NC",      InferenceEngine::Layout::NC},
                                                              {"CN",      InferenceEngine::Layout::CN},
                                                              {"NCDHW",   InferenceEngine::Layout::NCDHW},
                                                              {"BLOCKED", InferenceEngine::Layout::BLOCKED}};
#define stringify(name) # name
#define IE_CHECK_CALL(expr) {                       \
    auto ret = (expr);                              \
    if (ret != InferenceEngine::StatusCode::OK) {   \
        THROW_IE_EXCEPTION << response.msg;         \
    }                                               \
}                                                   \


uint32_t getOptimalNumberOfRequests(const InferenceEngine::IExecutableNetwork::Ptr actual) {
    try {
        InferenceEngine::ResponseDesc response;
        InferenceEngine::Parameter parameter_value;
        IE_CHECK_CALL(actual->GetMetric(METRIC_KEY(SUPPORTED_METRICS), parameter_value, &response));
        auto supported_metrics = parameter_value.as < std::vector < std::string >> ();
        std::string key = METRIC_KEY(OPTIMAL_NUMBER_OF_INFER_REQUESTS);
        if (std::find(supported_metrics.begin(), supported_metrics.end(), key) != supported_metrics.end()) {
            IE_CHECK_CALL(actual->GetMetric(key, parameter_value, &response));
            if (parameter_value.is<unsigned int>())
                return parameter_value.as<unsigned int>();
            else
                THROW_IE_EXCEPTION << "Unsupported format for " << key << "!"
                                   << " Please specify number of infer requests directly!";
        } else {
            THROW_IE_EXCEPTION << "Can't load network: " << key << " is not supported!"
                               << " Please specify number of infer requests directly!";
        }
    } catch (const std::exception &ex) {
        THROW_IE_EXCEPTION << "Can't load network: " << ex.what()
                           << " Please specify number of infer requests directly!";
    }
}

PyObject *parse_parameter(const InferenceEngine::Parameter &param) {
    // Check for std::string
    if (param.is<std::string>()) {
        return PyUnicode_FromString(param.as<std::string>().c_str());
    }
        // Check for int
    else if (param.is<int>()) {
        auto val = param.as<int>();
        return PyLong_FromLong((long)val);
    }
        // Check for unsinged int
    else if (param.is<unsigned int>()) {
        auto val = param.as<unsigned int>();
        return PyLong_FromLong((unsigned long)val);
    }
        // Check for float
    else if (param.is<float>()) {
        auto val = param.as<float>();
        return PyFloat_FromDouble((double)val);
    }
        // Check for bool
    else if (param.is<bool>()) {
        auto val = param.as<bool>();
        return val ? Py_True : Py_False;
    }
        // Check for std::vector<std::string>
    else if (param.is<std::vector<std::string>>()) {
        auto val = param.as<std::vector<std::string>>();
        PyObject *list = PyList_New(0);
        for (const auto & it : val){
            PyObject *str_val = PyUnicode_FromString(it.c_str());
            PyList_Append(list, str_val);
        }
        return list;
    }
        // Check for std::vector<int>
    else if (param.is<std::vector<int>>()){
        auto val = param.as<std::vector<int>>();
        PyObject *list = PyList_New(0);
        for (const auto & it : val){
            PyList_Append(list, PyLong_FromLong(it));
        }
        return list;
    }
        // Check for std::vector<unsigned int>
    else if (param.is<std::vector<unsigned int>>()){
        auto val = param.as<std::vector<unsigned int>>();
        PyObject *list = PyList_New(0);
        for (const auto &it : val) {
            PyList_Append(list, PyLong_FromLong(it));
        }
        return list;
    }
        // Check for std::vector<float>
    else if (param.is<std::vector<float>>()){
        auto val = param.as<std::vector<float>>();
        PyObject *list = PyList_New(0);
        for (const auto &it : val) {
            PyList_Append(list, PyFloat_FromDouble((double) it));
        }
        return list;
    }
        // Check for std::tuple<unsigned int, unsigned int>
    else if (param.is<std::tuple<unsigned int, unsigned int >>()) {
        auto val = param.as<std::tuple<unsigned int, unsigned int >>();
        PyObject *tuple = PyTuple_New(2);
        PyTuple_SetItem(tuple, 0, PyLong_FromUnsignedLong((unsigned long)std::get<0>(val)));
        PyTuple_SetItem(tuple, 1, PyLong_FromUnsignedLong((unsigned long)std::get<1>(val)));
        return tuple;
    }
        // Check for std::tuple<unsigned int, unsigned int, unsigned int>
    else if (param.is<std::tuple<unsigned int, unsigned int, unsigned int >>()) {
        auto val = param.as<std::tuple<unsigned int, unsigned int, unsigned int >>();
        PyObject *tuple = PyTuple_New(3);
        PyTuple_SetItem(tuple, 0, PyLong_FromUnsignedLong((unsigned long)std::get<0>(val)));
        PyTuple_SetItem(tuple, 1, PyLong_FromUnsignedLong((unsigned long)std::get<1>(val)));
        PyTuple_SetItem(tuple, 2, PyLong_FromUnsignedLong((unsigned long)std::get<2>(val)));
        return tuple;
    }
        // Check for std::map<std::string, std::string>
    else if (param.is<std::map<std::string, std::string>>()) {
        auto val = param.as<std::map<std::string, std::string>>();
        PyObject *dict = PyDict_New();
        for (const auto &it : val){
            PyDict_SetItemString(dict, it.first.c_str(), PyUnicode_FromString(it.second.c_str()));
        }
        return dict;
    }
        // Check for std::map<std::string, int>
    else if (param.is<std::map<std::string, int>>()) {
        auto val = param.as<std::map<std::string, int>>();
        PyObject *dict = PyDict_New();
        for (const auto &it : val){
            PyDict_SetItemString(dict, it.first.c_str(), PyLong_FromLong((long)it.second));
        }
        return dict;
    }
    else {
        PyErr_SetString(PyExc_TypeError, "Failed to convert parameter to Python representation!");
        return (PyObject *) NULL;
    }
}

InferenceEnginePython::IENetwork::IENetwork(const std::string &model, const std::string &weights) {
    IE_SUPPRESS_DEPRECATED_START
    InferenceEngine::CNNNetReader net_reader;
    net_reader.ReadNetwork(model);
    net_reader.ReadWeights(weights);
    auto net = net_reader.getNetwork();
    IE_SUPPRESS_DEPRECATED_END
    actual = std::make_shared<InferenceEngine::CNNNetwork>(net);
    name = actual->getName();
    batch_size = actual->getBatchSize();
    precision = actual->getPrecision().name();
}

InferenceEnginePython::IENetwork::IENetwork(const std::shared_ptr<InferenceEngine::CNNNetwork> &cnn_network)
        : actual(cnn_network) {
    name = actual->getName();
    batch_size = actual->getBatchSize();
    precision = actual->getPrecision().name();
}

InferenceEnginePython::IENetwork::IENetwork(PyObject* network) {
#if defined(ENABLE_NGRAPH)
    auto* capsule_ptr = PyCapsule_GetPointer(network, "ngraph_function");
    auto* function_sp = static_cast<std::shared_ptr<ngraph::Function>*>(capsule_ptr);
    if (function_sp == nullptr)
        THROW_IE_EXCEPTION << "Cannot create CNNNetwork from capsule! Capsule doesn't contain nGraph function!";

    InferenceEngine::CNNNetwork cnnNetwork(*function_sp);
    actual = std::make_shared<InferenceEngine::CNNNetwork>(cnnNetwork);
    name = actual->getName();
    batch_size = actual->getBatchSize();
    precision = actual->getPrecision().name();
#else
    THROW_IE_EXCEPTION << "InferenceEngine was built without nGraph support!";
#endif
}

void
InferenceEnginePython::IENetwork::load_from_buffer(const char *xml, size_t xml_size, uint8_t *bin, size_t bin_size) {
    IE_SUPPRESS_DEPRECATED_START
    InferenceEngine::CNNNetReader net_reader;
    net_reader.ReadNetwork(xml, xml_size);
    InferenceEngine::TensorDesc tensorDesc(InferenceEngine::Precision::U8, { bin_size }, InferenceEngine::Layout::C);
    auto weights_blob = InferenceEngine::make_shared_blob<uint8_t>(tensorDesc, bin, bin_size);
    net_reader.SetWeights(weights_blob);
    name = net_reader.getName();
    auto net = net_reader.getNetwork();
    IE_SUPPRESS_DEPRECATED_END
    actual = std::make_shared<InferenceEngine::CNNNetwork>(net);
    batch_size = actual->getBatchSize();
    precision = actual->getPrecision().name();
}

void InferenceEnginePython::IENetwork::serialize(const std::string &path_to_xml, const std::string &path_to_bin) {
    actual->serialize(path_to_xml, path_to_bin);
}

const std::vector <InferenceEngine::CNNLayerPtr>
InferenceEnginePython::IENetwork::getLayers() {
    std::vector<InferenceEngine::CNNLayerPtr> result;
    std::vector<InferenceEngine::CNNLayerPtr> sorted_layers = InferenceEngine::details::CNNNetSortTopologically(*actual);
    for (const auto &layer : sorted_layers) {
        result.emplace_back(layer);
    }
    return result;
}

PyObject* InferenceEnginePython::IENetwork::getFunction() {
#if defined(ENABLE_NGRAPH)
    const char * py_capsule_name = "ngraph_function";
    auto ngraph_func_ptr = actual->getFunction();
    // create a shared pointer on the heap before putting it in the capsule
    // this secures the lifetime of the object transferred by the capsule
    auto* sp_copy = new std::shared_ptr<const ngraph::Function>(ngraph_func_ptr);

    // a destructor callback that will delete the heap allocated shared_ptr
    // when the capsule is destructed
    auto sp_deleter = [](PyObject* capsule) {
        auto* capsule_ptr = PyCapsule_GetPointer(capsule, "ngraph_function");
        auto* function_sp = static_cast<std::shared_ptr<ngraph::Function>*>(capsule_ptr);
        if (function_sp) {
            delete function_sp;
        }
    };
    if (ngraph_func_ptr) {
        //return PyCapsule_New(&ngraph_func_ptr, py_capsule_name, NULL);
        return PyCapsule_New(sp_copy, py_capsule_name, sp_deleter);
    } else {
        return nullptr;
    }
#else
    return nullptr;
#endif
}

const std::map <std::string, InferenceEngine::DataPtr> InferenceEnginePython::IENetwork::getInputs() {
    std::map <std::string, InferenceEngine::DataPtr> inputs;
    const InferenceEngine::InputsDataMap &inputsInfo = actual->getInputsInfo();
    for (auto &in : inputsInfo) {
        inputs[in.first] = in.second->getInputData();
    }
    return inputs;
}

const std::map <std::string, InferenceEngine::DataPtr> InferenceEnginePython::IENetwork::getOutputs() {
    std::map <std::string, InferenceEngine::DataPtr> outputs;
    const InferenceEngine::OutputsDataMap &outputsInfo = actual->getOutputsInfo();
    for (auto &out : outputsInfo) {
        outputs[out.first] = out.second;
    }
    return outputs;
}

void
InferenceEnginePython::IENetwork::addOutput(const std::string &out_layer, size_t port_id) {
    actual->addOutput(out_layer, port_id);
}

void InferenceEnginePython::IENetwork::setBatch(const size_t size) {
    actual->setBatchSize(size);
}

void InferenceEnginePython::IENetwork::reshape(const std::map <std::string, std::vector<size_t>> &input_shapes) {
    actual->reshape(input_shapes);
}

const std::map <std::string, std::map<std::string, std::vector < float>>>

InferenceEnginePython::IENetwork::getStats() {
    InferenceEngine::ICNNNetworkStats *pstats = nullptr;
    InferenceEngine::ResponseDesc response;
    IE_CHECK_CALL(((InferenceEngine::ICNNNetwork &) *actual).getStats(&pstats, &response));
    auto statsMap = pstats->getNodesStats();
    std::map < std::string, std::map < std::string, std::vector < float >> > map;
    for (const auto &it : statsMap) {
        std::map <std::string, std::vector<float>> stats;
        stats.emplace("min", it.second->_minOutputs);
        stats.emplace("max", it.second->_maxOutputs);
        map.emplace(it.first, stats);
    }
    return map;
}

void InferenceEnginePython::IENetwork::setStats(const std::map<std::string, std::map<std::string,
        std::vector<float>>> &stats) {
    InferenceEngine::ICNNNetworkStats *pstats = nullptr;
    InferenceEngine::ResponseDesc response;
    IE_CHECK_CALL(((InferenceEngine::ICNNNetwork &) *actual).getStats(&pstats, &response));
    std::map<std::string, InferenceEngine::NetworkNodeStatsPtr> newNetNodesStats;
    for (const auto &it : stats) {
        InferenceEngine::NetworkNodeStatsPtr nodeStats = InferenceEngine::NetworkNodeStatsPtr(
                new InferenceEngine::NetworkNodeStats());
        newNetNodesStats.emplace(it.first, nodeStats);
        nodeStats->_minOutputs = it.second.at("min");
        nodeStats->_maxOutputs = it.second.at("max");
    }
    pstats->setNodesStats(newNetNodesStats);
}


IE_SUPPRESS_DEPRECATED_START
InferenceEnginePython::IEPlugin::IEPlugin(const std::string &device, const std::vector <std::string> &plugin_dirs) {

    InferenceEngine::PluginDispatcher dispatcher{plugin_dirs};
    actual = dispatcher.getPluginByDevice(device);
    auto pluginVersion = actual.GetVersion();
    version = std::to_string(pluginVersion->apiVersion.major) + ".";
    version += std::to_string(pluginVersion->apiVersion.minor) + ".";
    version += pluginVersion->buildNumber;
    device_name = device;
}
IE_SUPPRESS_DEPRECATED_END

void InferenceEnginePython::IEPlugin::setInitialAffinity(const InferenceEnginePython::IENetwork &net) {
    IE_SUPPRESS_DEPRECATED_START
    InferenceEngine::InferenceEnginePluginPtr hetero_plugin(actual);
    InferenceEngine::QueryNetworkResult queryRes;
    auto &network = net.actual;

    hetero_plugin->QueryNetwork(*network, {}, queryRes);
    IE_SUPPRESS_DEPRECATED_END

    if (queryRes.rc != InferenceEngine::StatusCode::OK) {
        THROW_IE_EXCEPTION << queryRes.resp.msg;
    }
    for (auto &&layer : queryRes.supportedLayersMap) {
        network->getLayerByName(layer.first.c_str())->affinity = layer.second;
    }
}

std::set <std::string> InferenceEnginePython::IEPlugin::queryNetwork(const InferenceEnginePython::IENetwork &net) {
    const std::shared_ptr<InferenceEngine::CNNNetwork> &network = net.actual;
    InferenceEngine::QueryNetworkResult queryRes;
    IE_SUPPRESS_DEPRECATED_START
    actual.QueryNetwork(*network, {}, queryRes);
    IE_SUPPRESS_DEPRECATED_END

    std::set <std::string> supportedLayers;
    for (auto &&layer : queryRes.supportedLayersMap) {
        supportedLayers.insert(layer.first);
    }

    return supportedLayers;
}


void InferenceEnginePython::IEPlugin::addCpuExtension(const std::string &extension_path) {
    auto extension_ptr = InferenceEngine::make_so_pointer<InferenceEngine::IExtension>(extension_path);
    auto extension = std::dynamic_pointer_cast<InferenceEngine::IExtension>(extension_ptr);
    IE_SUPPRESS_DEPRECATED_START
    actual.AddExtension(extension);
    IE_SUPPRESS_DEPRECATED_END
}

std::unique_ptr <InferenceEnginePython::IEExecNetwork>
InferenceEnginePython::IEPlugin::load(const InferenceEnginePython::IENetwork &net,
                                      int num_requests,
                                      const std::map <std::string, std::string> &config) {
    InferenceEngine::ResponseDesc response;
    auto exec_network = InferenceEnginePython::make_unique<InferenceEnginePython::IEExecNetwork>(net.name,
                                                                                                 num_requests);
    IE_SUPPRESS_DEPRECATED_START
    exec_network->actual = actual.LoadNetwork(*net.actual, config);
    IE_SUPPRESS_DEPRECATED_END

    if (0 == num_requests) {
        num_requests = getOptimalNumberOfRequests(exec_network->actual);
        exec_network->infer_requests.resize(num_requests);
    }

    for (size_t i = 0; i < num_requests; ++i) {
        InferRequestWrap &infer_request = exec_network->infer_requests[i];
        IE_CHECK_CALL(exec_network->actual->CreateInferRequest(infer_request.request_ptr, &response))
    }

    return exec_network;
}

void InferenceEnginePython::IEPlugin::setConfig(const std::map<std::string, std::string> &config) {
    IE_SUPPRESS_DEPRECATED_START
    actual.SetConfig(config);
    IE_SUPPRESS_DEPRECATED_END
}

InferenceEnginePython::IEExecNetwork::IEExecNetwork(const std::string &name, size_t num_requests) :
        infer_requests(num_requests), name(name) {
}

void InferenceEnginePython::IEExecNetwork::infer() {
    InferRequestWrap &request = infer_requests[0];
    request.infer();
}

InferenceEnginePython::IENetwork InferenceEnginePython::IEExecNetwork::GetExecGraphInfo() {
    InferenceEngine::ResponseDesc response;
    InferenceEngine::ICNNNetwork::Ptr graph;
    IE_CHECK_CALL(actual->GetExecGraphInfo(graph, &response));
    return IENetwork(std::make_shared<InferenceEngine::CNNNetwork>(graph));
}

PyObject *InferenceEnginePython::IEExecNetwork::getMetric(const std::string &metric_name) {
    InferenceEngine::Parameter parameter;
    InferenceEngine::ResponseDesc response;
    IE_CHECK_CALL(actual->GetMetric(metric_name, parameter, &response));
    return parse_parameter(parameter);
}

PyObject *InferenceEnginePython::IEExecNetwork::getConfig(const std::string &metric_name) {
    InferenceEngine::Parameter parameter;
    InferenceEngine::ResponseDesc response;
    IE_CHECK_CALL(actual->GetMetric(metric_name, parameter, &response));
    return parse_parameter(parameter);
}

void InferenceEnginePython::IEExecNetwork::exportNetwork(const std::string &model_file) {
    InferenceEngine::ResponseDesc response;
    IE_CHECK_CALL(actual->Export(model_file, &response));
}

std::map <std::string, InferenceEngine::DataPtr> InferenceEnginePython::IEExecNetwork::getInputs() {
    InferenceEngine::ConstInputsDataMap inputsDataMap;
    InferenceEngine::ResponseDesc response;
    IE_CHECK_CALL(actual->GetInputsInfo(inputsDataMap, &response));
    std::map <std::string, InferenceEngine::DataPtr> pyInputs;
    for (const auto &item : inputsDataMap) {
        pyInputs[item.first] = item.second->getInputData();
    }
    return pyInputs;
}

std::map <std::string, InferenceEngine::CDataPtr> InferenceEnginePython::IEExecNetwork::getOutputs() {
    InferenceEngine::ConstOutputsDataMap outputsDataMap;
    InferenceEngine::ResponseDesc response;
    IE_CHECK_CALL(actual->GetOutputsInfo(outputsDataMap, &response));
    std::map <std::string, InferenceEngine::CDataPtr> pyInputs;
    for (const auto &item : outputsDataMap) {
        pyInputs[item.first] = item.second;
    }
    return pyInputs;
}

void InferenceEnginePython::InferRequestWrap::getBlobPtr(const std::string &blob_name,
                                                         InferenceEngine::Blob::Ptr &blob_ptr) {
    InferenceEngine::ResponseDesc response;
    IE_CHECK_CALL(request_ptr->GetBlob(blob_name.c_str(), blob_ptr, &response));
}


void InferenceEnginePython::InferRequestWrap::setBatch(int size) {
    InferenceEngine::ResponseDesc response;
    IE_CHECK_CALL(request_ptr->SetBatch(size, &response));
}

void latency_callback(InferenceEngine::IInferRequest::Ptr request, InferenceEngine::StatusCode code) {
    if (code != InferenceEngine::StatusCode::OK) {
        THROW_IE_EXCEPTION << "Async Infer Request failed with status code " << code;
    }
    InferenceEnginePython::InferRequestWrap *requestWrap;
    InferenceEngine::ResponseDesc dsc;
    request->GetUserData(reinterpret_cast<void **>(&requestWrap), &dsc);
    auto end_time = Time::now();
    auto execTime = std::chrono::duration_cast<ns>(end_time - requestWrap->start_time);
    requestWrap->exec_time = static_cast<double>(execTime.count()) * 0.000001;
    if (requestWrap->user_callback) {
        requestWrap->user_callback(requestWrap->user_data, code);
    }
}

void InferenceEnginePython::InferRequestWrap::setCyCallback(cy_callback callback, void *data) {
    user_callback = callback;
    user_data = data;
}

void InferenceEnginePython::InferRequestWrap::infer() {
    InferenceEngine::ResponseDesc response;
    start_time = Time::now();
    IE_CHECK_CALL(request_ptr->Infer(&response));
    auto end_time = Time::now();
    auto execTime = std::chrono::duration_cast<ns>(end_time - start_time);
    exec_time = static_cast<double>(execTime.count()) * 0.000001;
}


void InferenceEnginePython::InferRequestWrap::infer_async() {
    InferenceEngine::ResponseDesc response;
    start_time = Time::now();
    IE_CHECK_CALL(request_ptr->SetUserData(this, &response));
    request_ptr->SetCompletionCallback(latency_callback);
    IE_CHECK_CALL(request_ptr->StartAsync(&response));
}

int InferenceEnginePython::InferRequestWrap::wait(int64_t timeout) {
    InferenceEngine::ResponseDesc responseDesc;
    InferenceEngine::StatusCode code = request_ptr->Wait(timeout, &responseDesc);
    return static_cast<int >(code);
}

std::map <std::string, InferenceEnginePython::ProfileInfo>
InferenceEnginePython::InferRequestWrap::getPerformanceCounts() {
    std::map <std::string, InferenceEngine::InferenceEngineProfileInfo> perf_counts;
    InferenceEngine::ResponseDesc response;
    request_ptr->GetPerformanceCounts(perf_counts, &response);
    std::map <std::string, InferenceEnginePython::ProfileInfo> perf_map;

    for (auto it : perf_counts) {
        InferenceEnginePython::ProfileInfo profile_info;
        switch (it.second.status) {
            case InferenceEngine::InferenceEngineProfileInfo::EXECUTED:
                profile_info.status = "EXECUTED";
                break;
            case InferenceEngine::InferenceEngineProfileInfo::NOT_RUN:
                profile_info.status = "NOT_RUN";
                break;
            case InferenceEngine::InferenceEngineProfileInfo::OPTIMIZED_OUT:
                profile_info.status = "OPTIMIZED_OUT";
                break;
            default:
                profile_info.status = "UNKNOWN";
        }
        profile_info.exec_type = it.second.exec_type;
        profile_info.layer_type = it.second.layer_type;
        profile_info.cpu_time = it.second.cpu_uSec;
        profile_info.real_time = it.second.realTime_uSec;
        profile_info.execution_index = it.second.execution_index;
        perf_map[it.first] = profile_info;
    }
    return perf_map;
}

std::string InferenceEnginePython::get_version() {
    auto version = InferenceEngine::GetInferenceEngineVersion();
    std::string version_str = std::to_string(version->apiVersion.major) + ".";
    version_str += std::to_string(version->apiVersion.minor) + ".";
    version_str += version->buildNumber;
    return version_str;
}


InferenceEnginePython::IECore::IECore(const std::string &xmlConfigFile) {
    actual = InferenceEngine::Core(xmlConfigFile);
}

std::map <std::string, InferenceEngine::Version>
InferenceEnginePython::IECore::getVersions(const std::string &deviceName) {
    return actual.GetVersions(deviceName);
}

std::unique_ptr <InferenceEnginePython::IEExecNetwork> InferenceEnginePython::IECore::loadNetwork(IENetwork network,
                                                                                                  const std::string &deviceName,
                                                                                                  const std::map <std::string, std::string> &config,
                                                                                                  int num_requests) {

    InferenceEngine::ResponseDesc response;
    auto exec_network = InferenceEnginePython::make_unique<InferenceEnginePython::IEExecNetwork>(network.name,
                                                                                                 num_requests);
    exec_network->actual = actual.LoadNetwork(*network.actual, deviceName, config);

    if (0 == num_requests) {
        num_requests = getOptimalNumberOfRequests(exec_network->actual);
        exec_network->infer_requests.resize(num_requests);
    }

    for (size_t i = 0; i < num_requests; ++i) {
        InferRequestWrap &infer_request = exec_network->infer_requests[i];
        IE_CHECK_CALL(exec_network->actual->CreateInferRequest(infer_request.request_ptr, &response))
    }

    return exec_network;
}

std::unique_ptr <InferenceEnginePython::IEExecNetwork> InferenceEnginePython::IECore::importNetwork(
        const std::string &modelFIle, const std::string &deviceName, const std::map <std::string, std::string> &config,
        int num_requests) {
    InferenceEngine::ResponseDesc response;
    auto exec_network = InferenceEnginePython::make_unique<InferenceEnginePython::IEExecNetwork>(EXPORTED_NETWORK_NAME,
                                                                                                 num_requests);
    exec_network->actual = actual.ImportNetwork(modelFIle, deviceName, config);

    if (0 == num_requests) {
        num_requests = getOptimalNumberOfRequests(exec_network->actual);
        exec_network->infer_requests.resize(num_requests);
    }

    for (size_t i = 0; i < num_requests; ++i) {
        InferRequestWrap &infer_request = exec_network->infer_requests[i];
        IE_CHECK_CALL(exec_network->actual->CreateInferRequest(infer_request.request_ptr, &response))
    }

    return exec_network;

}

std::map <std::string, std::string>
InferenceEnginePython::IECore::queryNetwork(InferenceEnginePython::IENetwork network,
                                            const std::string &deviceName,
                                            const std::map <std::string, std::string> &config) {
    auto res = actual.QueryNetwork(*network.actual, deviceName, config);
    return res.supportedLayersMap;
}

void InferenceEnginePython::IECore::setConfig(const std::map <std::string, std::string> &config,
                                              const std::string &deviceName) {
    actual.SetConfig(config, deviceName);
}

void InferenceEnginePython::IECore::registerPlugin(const std::string &pluginName, const std::string &deviceName) {
    actual.RegisterPlugin(pluginName, deviceName);
}

void InferenceEnginePython::IECore::unregisterPlugin(const std::string &deviceName) {
    actual.UnregisterPlugin(deviceName);
}

void InferenceEnginePython::IECore::registerPlugins(const std::string &xmlConfigFile) {
    actual.RegisterPlugins(xmlConfigFile);
}

void InferenceEnginePython::IECore::addExtension(const std::string &ext_lib_path, const std::string &deviceName) {
    auto extension_ptr = InferenceEngine::make_so_pointer<InferenceEngine::IExtension>(ext_lib_path);
    auto extension = std::dynamic_pointer_cast<InferenceEngine::IExtension>(extension_ptr);
    actual.AddExtension(extension, deviceName);
}

std::vector <std::string> InferenceEnginePython::IECore::getAvailableDevices() {
    return actual.GetAvailableDevices();
}

PyObject *InferenceEnginePython::IECore::getMetric(const std::string &deviceName, const std::string &name) {
    InferenceEngine::Parameter param = actual.GetMetric(deviceName, name);
    return parse_parameter(param);
}

PyObject *InferenceEnginePython::IECore::getConfig(const std::string &deviceName, const std::string &name) {
    InferenceEngine::Parameter param = actual.GetConfig(deviceName, name);
    return parse_parameter(param);
}
