// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include <map>
#include <vector>
#include <limits>
#include <chrono>
#include <climits>
#include <memory>
#include <utility>
#include <future>

#include "mkldnn_graph.h"
#include "ie_parallel.hpp"
#include "mkldnn_streams.h"
#include "ie_compound_blob.h"

using namespace mkldnn;
using namespace MKLDNNPlugin;
using namespace InferenceEngine;
using namespace InferenceEngine::details;

namespace MKLDNNPlugin {

thread_local MultiWorkerTaskContext MultiWorkerTaskExecutor::ptrContext;

bool check_env_variables() {
#if IE_THREAD == IE_THREAD_OMP
    return MKLDNNPlugin::cpu::checkOpenMpEnvVars(false);
#else
    return false;
#endif
}

#if !(defined(__APPLE__) || defined(_WIN32))
/* Get the cores affinity mask for the current process */
bool get_process_mask(int& ncpus, cpu_set_t*& mask) {
    for (ncpus = sizeof(cpu_set_t) / CHAR_BIT; ncpus < 32768 /* reasonable limit of #cores*/; ncpus <<= 1) {
        mask = CPU_ALLOC(ncpus);
        if (!mask) return false;

        const size_t size = CPU_ALLOC_SIZE(ncpus);
        CPU_ZERO_S(size, mask);
        const int err = sched_getaffinity(getpid(), size, mask);
        // the result fits the mask
        if (!err) break;
        // mask size is not enough
        CPU_FREE(mask);
        mask = NULL;
        // other error
        if (errno != EINVAL) break;
    }
    if (!mask) {
        return false;
    }
    return true;
}
/* Pin current thread to a set of cores determined by the mask. */
bool pin_current_thread_by_mask(int ncores, const cpu_set_t* proc_mask) {
    return 0 == sched_setaffinity(0, ncores, proc_mask);
}
/* Pin thread to a spare core in the round-robin scheme, while respecting the given process mask.
 * The function can also handle the hyper-threading (by populating the physical cores first) */
bool pin_thread_to_vacant_core(int thr_idx, int hyperthreads, int ncores, const cpu_set_t* proc_mask) {
    if (proc_mask == nullptr)
        return false;
    const size_t size = CPU_ALLOC_SIZE(ncores);
    const int num_cpus = CPU_COUNT_S(size, proc_mask);
    thr_idx %= num_cpus;  // To limit unique number in [; num_cpus-1] range
    // Place threads with specified step
    int cpu_idx = 0;
    for (int i = 0, offset = 0; i < thr_idx; ++i) {
        cpu_idx += hyperthreads;
        if (cpu_idx >= num_cpus)
            cpu_idx = ++offset;
    }

    // Find index of 'cpu_idx'-th bit that equals to 1
    int mapped_idx = -1;
    while (cpu_idx >= 0) {
        if (CPU_ISSET_S(++mapped_idx, size, proc_mask))
            --cpu_idx;
    }

    cpu_set_t *target_mask = CPU_ALLOC(ncores);
    CPU_ZERO_S(size, target_mask);
    CPU_SET_S(mapped_idx, size, target_mask);
    bool res = pin_current_thread_by_mask(size, target_mask);
    CPU_FREE(target_mask);
    return res;
}
bool pin_current_thread_to_socket(int socket) {
    const int numa_nodes_num = MKLDNNPlugin::cpu::getAvailableNUMANodes().size();
    const int cores = MKLDNNPlugin::cpu::getNumberOfCPUCores();
    const int cores_per_socket = cores/numa_nodes_num;

    int ncpus;
    cpu_set_t *mask;
    if (!get_process_mask(ncpus, mask))
        return false;
    cpu_set_t *target_mask = CPU_ALLOC(ncpus);
    const size_t size = CPU_ALLOC_SIZE(ncpus);
    CPU_ZERO_S(size, target_mask);

    for (int core = socket*cores_per_socket; core < (socket+1)*cores_per_socket; core++) {
        CPU_SET_S(core, size, target_mask);
    }
    // respect the user-defined mask for the entire process
    CPU_AND_S(size, target_mask, target_mask, mask);
    CPU_FREE(mask);
    bool res = false;
    if (CPU_COUNT_S(size, target_mask))  // if we have non-zero mask to set
        res = pin_current_thread_by_mask(size, target_mask);
    CPU_FREE(target_mask);
    return res;
}
#else   // no threads pinning/binding on Win/MacOS
bool get_process_mask(int& ncpus, cpu_set_t*& mask) {
    ncpus = 0;
    mask =  nullptr;
    return false;
}
bool pin_thread_to_vacant_core(int thr_idx, int hyperthreads, int ncores, const cpu_set_t* proc_mask) {
    return false;
}
bool pin_current_thread_by_mask(int ncores, const cpu_set_t* proc_mask) {
    return false;
}
bool pin_current_thread_to_socket(int socket) {
    return false;
}
#endif  // !(defined(__APPLE__) || defined(_WIN32))

MultiWorkerTaskExecutor::MultiWorkerTaskExecutor(const std::vector<Task>& init_tasks, std::string name) :
        _isStopped(false), _name(name) {
    std::vector<std::packaged_task<void()>> initTasks;
    std::vector<std::future<void>> futures;
    for (int t = 0; t < init_tasks.size(); t++) {
        initTasks.emplace_back([&init_tasks, t] {init_tasks[t]();});
        futures.emplace_back(initTasks.back().get_future());
    }
    for (int t = 0; t < init_tasks.size(); t++) {
        _threads.emplace_back(std::thread([&, t] {
            // initialization (no contention, every worker thread is doing it's own task)
            initTasks[t]();

            while (!_isStopped) {
                Task currentTask = nullptr;
                {  // waiting for the new task or for stop signal
                    std::unique_lock<std::mutex> lock(_queueMutex);
                    _queueCondVar.wait(lock, [&]() { return !_taskQueue.empty() || _isStopped; });
                    if (!_taskQueue.empty()) {
                        currentTask = std::move(_taskQueue.front());
                        _taskQueue.pop();
                    }
                }
                if (currentTask)
                    currentTask();
            }
        }));
    }
    for (auto&& f : futures)
        f.wait();
    for (auto&& f : futures) {
        try {
            f.get();
        } catch(...) {
            stop();
            throw;
        }
    }
}

void MultiWorkerTaskExecutor::stop() {
    _isStopped = true;
    _queueCondVar.notify_all();
    for (auto& thread : _threads) {
        if (thread.joinable()) {
            thread.join();
        }
    }
}

MultiWorkerTaskExecutor::~MultiWorkerTaskExecutor() {
    stop();
}

void MultiWorkerTaskExecutor::run(Task task) {
    {
        std::lock_guard<std::mutex> lock(_queueMutex);
        _taskQueue.push(std::move(task));
    }
    _queueCondVar.notify_one();
}

MKLDNNPlugin::MKLDNNGraphlessInferRequest::MKLDNNGraphlessInferRequest(InferenceEngine::InputsDataMap networkInputs,
                                                                       InferenceEngine::OutputsDataMap networkOutputs)
        : InferRequestInternal(networkInputs, networkOutputs), m_curBatch(-1) {
    // Allocate all input blobs
    for (const auto& it : networkInputs) {
        InferenceEngine::Blob::Ptr blob;
        GetBlob(it.first.c_str(), blob);
    }
    // Allocate all output blobs
    for (const auto& it : networkOutputs) {
        InferenceEngine::Blob::Ptr blob;
        GetBlob(it.first.c_str(), blob);
    }
}


void MKLDNNPlugin::MKLDNNGraphlessInferRequest::InferImpl() {
    IE_PROFILING_AUTO_SCOPE(MKLDNN_INFER)

    auto infer = [this] {
        IE_ASSERT(MKLDNNPlugin::MultiWorkerTaskExecutor::ptrContext.ptrGraph != nullptr);
        MKLDNNGraph::Ptr graph = MKLDNNPlugin::MultiWorkerTaskExecutor::ptrContext.ptrGraph;
        if (!graph->IsReady())
            THROW_IE_EXCEPTION << "Network not loaded.";
        if (m_curBatch > 0 && !graph->getProperty().enableDynamicBatch)
            THROW_IE_EXCEPTION << "Dynamic batch is not enabled.";

        if (m_curBatch > graph->getProperty().batchLimit)
            THROW_IE_EXCEPTION << "Invalid dynamic batch size " << m_curBatch <<
                               " for this request.";

        // execute input pre-processing.
        execDataPreprocessing(_inputs);

        // need to retain converted blobs until infer finish
        std::vector<InferenceEngine::Blob::Ptr> convertedInputs;
        for (auto input : _inputs) {
            if (!_networkInputs[input.first]) {
                THROW_IE_EXCEPTION <<
                                   "input blobs map contains not registered during IInferencePlugin::LoadNetwork blob with name "
                                   << input.first;
            }
            InferenceEngine::Blob::Ptr iconv;
            InferenceEngine::TBlob<float> *in_f = nullptr;
            switch (input.second->getTensorDesc().getPrecision()) {
                case InferenceEngine::Precision::FP32:
                case InferenceEngine::Precision::I32:
                case InferenceEngine::Precision::I8:
                    graph->PushInputData(input.first, input.second);
                    break;
                case InferenceEngine::Precision::U16:
                    // U16 is unsupported by mkldnn, so here we convert the blob and send FP32
                    iconv = InferenceEngine::make_shared_blob<float>({InferenceEngine::Precision::FP32,
                                                                      input.second->getTensorDesc().getDims(),
                                                                      input.second->getTensorDesc().getLayout()});
                    convertedInputs.push_back(iconv);
                    iconv->allocate();
                    in_f = dynamic_cast<InferenceEngine::TBlob<float> *>(iconv.get());
                    if (in_f == nullptr)
                        THROW_IE_EXCEPTION << "Cannot get TBlob";
                    IE_SUPPRESS_DEPRECATED_START
                    InferenceEngine::copyToFloat<uint16_t>(in_f->data(), input.second.get());
                    IE_SUPPRESS_DEPRECATED_END
                    graph->PushInputData(input.first, iconv);
                    break;
                case InferenceEngine::Precision::I16:
                    if (graph->hasMeanImageFor(input.first)) {
                        // If a mean image exists, we convert the blob and send FP32
                        iconv = InferenceEngine::make_shared_blob<float>({InferenceEngine::Precision::FP32,
                                                                          input.second->getTensorDesc().getDims(),
                                                                          input.second->getTensorDesc().getLayout()});
                        convertedInputs.push_back(iconv);
                        iconv->allocate();
                        in_f = dynamic_cast<InferenceEngine::TBlob<float> *>(iconv.get());
                        if (in_f == nullptr)
                            THROW_IE_EXCEPTION << "Cannot get TBlob";
                        IE_SUPPRESS_DEPRECATED_START
                        InferenceEngine::copyToFloat<int16_t>(in_f->data(), input.second.get());
                        IE_SUPPRESS_DEPRECATED_END
                        graph->PushInputData(input.first, iconv);
                    } else {
                        // Instead we can send I16 directly
                        graph->PushInputData(input.first, input.second);
                    }
                    break;
                case InferenceEngine::Precision::U8:
                    if (graph->hasMeanImageFor(input.first)) {
                        // If a mean image exists, we convert the blob and send FP32
                        iconv = InferenceEngine::make_shared_blob<float>({InferenceEngine::Precision::FP32,
                                                                          input.second->getTensorDesc().getDims(),
                                                                          input.second->getTensorDesc().getLayout()});
                        convertedInputs.push_back(iconv);
                        iconv->allocate();
                        in_f = dynamic_cast<InferenceEngine::TBlob<float> *>(iconv.get());
                        if (in_f == nullptr)
                            THROW_IE_EXCEPTION << "Cannot get TBlob";
                        IE_SUPPRESS_DEPRECATED_START
                        InferenceEngine::copyToFloat<uint8_t>(in_f->data(), input.second.get());
                        IE_SUPPRESS_DEPRECATED_END
                        graph->PushInputData(input.first, iconv);
                    } else {
                        // Instead we can send I8 directly
                        graph->PushInputData(input.first, input.second);
                    }
                    break;
                default:
                    THROW_IE_EXCEPTION << "Unsupported input precision " << input.second->getTensorDesc().getPrecision();
            }
        }
        graph->Infer(m_curBatch);
        graph->PullOutputData(_outputs);
        if (graph->getProperty().collectPerfCounters) {
            m_perfMap.clear();
            graph->GetPerfData(m_perfMap);
        }
    };
#if (IE_THREAD == IE_THREAD_TBB || IE_THREAD == IE_THREAD_TBB_AUTO)
    auto_scope_observing observer(MKLDNNPlugin::MultiWorkerTaskExecutor::ptrContext.ptrGraph->ptrObserver);
    // a TBB arena is made "this" for Infer call via executing lambda for the arena
    MKLDNNPlugin::MultiWorkerTaskExecutor::ptrContext.ptrGraph->ptrArena->execute([&] { infer(); });
#else
    infer();
#endif
}

void MKLDNNPlugin::MKLDNNGraphlessInferRequest::GetPerformanceCounts(
        std::map<std::string, InferenceEngine::InferenceEngineProfileInfo> &perfMap) const {
    perfMap = m_perfMap;
}

void MKLDNNPlugin::MKLDNNGraphlessInferRequest::GetBlob(const char *name, InferenceEngine::Blob::Ptr &data) {
    // ROI blob is returned only if it was set previously.
    auto it = _preProcData.find(name);
    if (it != _preProcData.end()) {
        data = it->second->getRoiBlob();
        return;
    }

    if (_inputs.find(name) != _inputs.end()) {
        data = _inputs[name];
        checkBlob(data, name, true);
        return;
    } else if (_networkInputs.find(name) != _networkInputs.end()) {
        InferenceEngine::Layout l = _networkInputs[name]->getLayout();
        InferenceEngine::Precision p = _networkInputs[name]->getPrecision();
        InferenceEngine::SizeVector dims = _networkInputs[name]->getTensorDesc().getDims();

        InferenceEngine::TensorDesc desc = InferenceEngine::TensorDesc(p, dims, l);
        _inputs[name] = data = make_blob_with_precision(desc);
        _inputs[name]->allocate();
        checkBlob(data, name, true);
        return;
    }

    if (_outputs.find(name) != _outputs.end()) {
        data = _outputs[name];
        checkBlob(data, name, false);
        return;
    } else if (_networkOutputs.find(name) != _networkOutputs.end()) {
        InferenceEngine::Layout l = _networkOutputs[name]->getLayout();
        InferenceEngine::Precision p = _networkOutputs[name]->getPrecision();
        InferenceEngine::SizeVector dims = _networkOutputs[name]->getTensorDesc().getDims();

        InferenceEngine::TensorDesc desc = InferenceEngine::TensorDesc(p, dims, l);
        _outputs[name] = data = make_blob_with_precision(desc);
        _outputs[name]->allocate();
        checkBlob(data, name, false);
        return;
    }

    THROW_IE_EXCEPTION << "Cannot find blob with name: " << name;
}

void MKLDNNPlugin::MKLDNNGraphlessInferRequest::SetBlob(const char *name, const InferenceEngine::Blob::Ptr &data) {
    if (name == nullptr) {
        THROW_IE_EXCEPTION << NOT_FOUND_str + "Failed to set blob with empty name";
    }
    if (!data)
        THROW_IE_EXCEPTION << NOT_ALLOCATED_str << "Failed to set empty blob with name: \'" << name << "\'";
    const bool compoundBlobPassed = data->is<CompoundBlob>();
    if (!compoundBlobPassed && data->buffer() == nullptr)
        THROW_IE_EXCEPTION << "Input data was not allocated. Input name: \'" << name << "\'";
    if (data->size() == 0) {
        THROW_IE_EXCEPTION << "Input data is empty. Input name: \'" << name << "\'";
    }

    InferenceEngine::InputInfo::Ptr foundInput;
    InferenceEngine::DataPtr foundOutput;
    size_t dataSize = data->size();
    if (findInputAndOutputBlobByName(name, foundInput, foundOutput)) {
        if (foundInput->getPrecision() != data->getTensorDesc().getPrecision()) {
            THROW_IE_EXCEPTION << PARAMETER_MISMATCH_str << "Failed to set Blob with precision "
                               << data->getTensorDesc().getPrecision();
        }

        const bool preProcRequired = preProcessingRequired(foundInput, data);
        if (compoundBlobPassed && !preProcRequired) {
            THROW_IE_EXCEPTION << NOT_IMPLEMENTED_str
                               << "cannot set compound blob: supported only for input pre-processing";
        }

        if (preProcRequired) {
            if (_preProcData.find(name) == _preProcData.end()) {
                _preProcData.emplace(name, CreatePreprocDataHelper());
            }
            _preProcData[name]->isApplicable(data, _inputs[name]);
            // Stores the given blob as ROI blob. It will be used to fill in network input during
            // pre-processing.
            _preProcData[name]->setRoiBlob(data);
        } else {
            size_t inputSize = InferenceEngine::details::product(foundInput->getTensorDesc().getDims());
            if (dataSize != inputSize) {
                THROW_IE_EXCEPTION << "Input blob size is not equal network input size ("
                                   << dataSize << "!=" << inputSize << ").";
            }
            _inputs[name] = data;
        }
    } else {
        if (compoundBlobPassed) {
            THROW_IE_EXCEPTION << NOT_IMPLEMENTED_str
                               << "cannot set compound blob: supported only for input pre-processing";
        }
        size_t outputSize = InferenceEngine::details::product(foundOutput->getDims());
        if (dataSize != outputSize) {
            THROW_IE_EXCEPTION << "Output blob size is not equal network output size ("
                               << dataSize << "!=" << outputSize << ").";
        }
        if (foundOutput->getPrecision() != data->getTensorDesc().getPrecision()) {
            THROW_IE_EXCEPTION << PARAMETER_MISMATCH_str
                               << "Failed to set Blob with precision not corresponding to user output precision";
        }
        _outputs[name] = data;
    }
}

void MKLDNNPlugin::MKLDNNGraphlessInferRequest::SetBatch(int new_batch) {
    if (new_batch < 1) {
        THROW_IE_EXCEPTION << "Invalid dynamic batch size " << new_batch <<
                           " for this request.";
    }
    m_curBatch = new_batch;
}

}  // namespace MKLDNNPlugin
