// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <mock_icnn_network.hpp>
#include "gna_matcher.hpp"
#include <gna/gna_config.hpp>
#include <gna-api-types-xnn.h>
#include <gna_executable_network.hpp>
#include "gna_plugin.hpp"
#include "gna_mock_api.hpp"
#include "matchers/precision_matcher.hpp"
#include "matchers/pwl_matcher.hpp"
#include "matchers/copy_matcher.hpp"
#include "matchers/diag_matcher.hpp"
#include "matchers/pwl_quantization_metrics_matcher.hpp"
#include "matchers/conv_matcher.hpp"
#include "matchers/pool_matcher.hpp"
#include "matchers/fill_with_data.hpp"
#include "matchers/weights_matcher.hpp"

#include <gmock/gmock-generated-actions.h>
#include <gmock/gmock-more-actions.h>
#include "gmock/gmock.h"
#include "net_pass.h"
#include "matchers/input_data_matcher.hpp"
#include <blob_factory.hpp>
#include <details/ie_cnn_network_tools.h>

using namespace std;
using namespace InferenceEngine;
using namespace GNAPluginNS;
using namespace ::testing;

class NullAllocator : public IAllocator {
 void * ptr = nullptr;
public:
    NullAllocator() {
        ptr = new char[1];
    }
    ~NullAllocator() {
        delete[] static_cast<char*>(ptr);
    }
    void * lock(void * handle, LockOp = LOCK_FOR_WRITE)  noexcept override {
        return ptr;
    }
    void  unlock(void * handle) noexcept override {

    }
    void * alloc(size_t size) noexcept override {
        return ptr;
    }
    virtual bool   free(void* handle) noexcept {
        return true;
    }
    virtual void Release() noexcept {
        delete this;
    }
};

void GNAPropagateMatcher :: match() {
    try {
        // matching gna propagate forward call.
        GNAPlugin plugin(_env.config);
        plugin.SetPolicy(_env.policy);
        size_t inputSize = 10;
        size_t outputSize = 10;
        InputsDataMap inputsInfo;
        OutputsDataMap  outputsInfo;

        auto loadNetworkFromIR = [&] () -> InferenceEngine::CNNNetwork {
            CNNNetReader net_reader;

            net_reader.ReadNetwork(_env.model.data(), _env.model.length());
            auto weights_fake = make_shared_blob<uint8_t>(TensorDesc(Precision::U8,
                    SizeVector({std::numeric_limits<uint32_t>::max()}), Layout::C), make_shared<NullAllocator>());
            net_reader.SetWeights(weights_fake);

            auto net_original = net_reader.getNetwork();
            size_t weightsSize = 0;
            std::vector<std::string> dataBlobs = {
                    "weights",
                    "biases",
                    "custom"
            };

            std::vector<InferenceEngine::CNNLayerPtr> tiBodies;

            for (auto &layer : net_original) {
                if (layer->type == "TensorIterator") {
                    auto tiBody = NetPass::TIBodySortTopologically(std::dynamic_pointer_cast<InferenceEngine::TensorIterator>(layer)->body);
                    tiBodies.insert(tiBodies.end(), tiBody.begin(), tiBody.end());
                }
            }
            std::vector<CNNLayerPtr> sortedLayers = details::CNNNetSortTopologically(net_original);
            sortedLayers.insert(sortedLayers.end(), tiBodies.begin(), tiBodies.end());

            for (auto &layer : sortedLayers) {
                for (auto &blobName : dataBlobs) {
                    auto weights = layer->blobs[blobName];
                    if (weights) {
                        weightsSize += weights->byteSize();
                    }
                }
            }

            auto weights = make_shared_blob<uint8_t >({ Precision::U8, {weightsSize}, Layout::C });

            weights->allocate();
            if (!_env.weightsFillPattern.empty()) {
                fillWeights(weights, _env.weightsFillPattern);
            } else {
                fillWeights(weights);
            }
            net_reader.SetWeights(weights);

            for (auto &pattern : _env.weightsByLayerFillPattern) {
                for (auto &layer : sortedLayers) {
                    if (layer->name == pattern.first) {
                        auto weightableLayer = dynamic_pointer_cast<WeightableLayer>(layer);
                        if (!weightableLayer) {
                            THROW_IE_EXCEPTION << "given layer: " << layer->name <<" doesnt have weights";
                        }
                        fillWeights(weightableLayer->_weights, pattern.second);
                        break;
                    }
                }
            }
            return net_reader.getNetwork();
        };

        auto loadCNNNetwork = [&] (CNNNetwork net_original) {

            auto input_dims = net_original.getInputsInfo().begin()->second->getTensorDesc().getDims();
            auto output = net_original.getOutputsInfo();
            // sometimes network might be created without outputs - ex memory output only
            auto output_dims = !output.empty() ? output.begin()->second->getTensorDesc().getDims() : input_dims;

            inputSize = details::product(std::begin(input_dims), std::end(input_dims));
            outputSize = details::product(std::begin(output_dims), std::end(output_dims));

            if (_env.cb) {
                _env.cb(net_original);
            }

            plugin.LoadNetwork(net_original);

            inputsInfo = net_original.getInputsInfo();
            outputsInfo = net_original.getOutputsInfo();
        };

        auto loadNetworkFromAOT = [&] () {
            auto sp = plugin.ImportNetwork(_env.importedModelFileName);
            inputsInfo = plugin.GetInputs();
            outputsInfo = plugin.GetOutputs();
        };

        std::map<std::string, Blob::Ptr> input;
        std::map<std::string, TBlob<float>::Ptr> output;
        size_t in_N = 1;
        size_t out_N = in_N;
        size_t in_C;
        size_t out_C;

        auto loadNetwork = [&]() {
            if (_env.ngraph_model) {
                CNNNetwork network;
                ASSERT_NO_THROW_IE_EXCEPTION(network = CNNNetwork(_env.ngraph_model));
                ASSERT_NO_FATAL_FAILURE(loadCNNNetwork(network));
#ifdef GNA_DEBUG
                network.serialize("CNNNetworkFromNgraphModel.xml", "CNNNetworkFromNgraphModel.bin");
#endif
            }
            else if (!_env.importedModelFileName.empty()) {
                ASSERT_NO_FATAL_FAILURE(loadNetworkFromAOT());
            } else {
                CNNNetwork network;
                ASSERT_NO_THROW_IE_EXCEPTION(network = loadNetworkFromIR());
                ASSERT_NO_FATAL_FAILURE(loadCNNNetwork(network));
            }
            const int channel_idx = 1;
            bool haveInputs = !_env.input_init.empty();
            for (auto && info :inputsInfo) {
                decltype(_env.input_init)::iterator it;
                auto & inputBlob = input[info.first];
                if (haveInputs) {
                    if (inputsInfo.size() != 1) {
                        ASSERT_NE(it = _env.input_init.find(info.first), _env.input_init.end());
                    } else {
                        ASSERT_NE(0, _env.input_init.size());
                        it = _env.input_init.begin();
                    }
                    in_C = it->second.size();
                    ASSERT_EQ(in_C, info.second->getTensorDesc().getDims()[channel_idx]);
                }

                inputBlob = make_blob_with_precision({ _env.input_precision, info.second->getTensorDesc().getDims(),
                    info.second->getLayout() });
                inputBlob->allocate();
                if (haveInputs) {
                    if (_env.input_precision == Precision::FP32) {
                        std::copy_n(it->second.cbegin(), in_N * in_C, inputBlob->buffer().as<float *>());
                    } else if (_env.input_precision == Precision::U8) {
                        std::copy_n(it->second.cbegin(), in_N * in_C, inputBlob->buffer().as<uint8_t *>());
                    } else {
                        std::logic_error(std::string("Unsupported input precision: ") + _env.input_precision.name());
                    }
                }
            }
            int expectedOutputIdx = 0;
            for (auto&& out : outputsInfo) {

                auto& outputBlob = output[out.first];
                auto outsize2 = details::product(out.second->getDims());
                bool matchoutput = _env.matchOutput;
                // expectations not set for given output
                if (expectedOutputIdx >= _env.expected_output.size()) {
                    matchoutput = false;
                }
                out_C = matchoutput ? _env.expected_output[expectedOutputIdx].size() : outsize2;
                outputBlob.reset(new TBlob<float>({ Precision::FP32, {out_N, out_C}, Layout::NC }));
                outputBlob->allocate();
                expectedOutputIdx++;
            }
        };


        StrictMock<GNACppApi> mockApi;
        std::vector<uint8_t> data;

        if (_env.config[GNA_CONFIG_KEY(DEVICE_MODE)].compare(GNA_CONFIG_VALUE(SW_FP32)) != 0 &&
            !_env.matchThrows) {
#if GNA_LIB_VER == 1 // TODO: GNA2: handle new API
            EXPECT_CALL(mockApi, GNAAlloc(_,_,_)).WillOnce(Invoke([&data](
                intel_gna_handle_t nGNADevice,   // handle to GNA accelerator
                uint32_t           sizeRequested,
                uint32_t*          sizeGranted
            ) {
                data.resize(sizeRequested);
                *sizeGranted = sizeRequested;
                return &data.front();
            }));
            EXPECT_CALL(mockApi, GNADeviceOpenSetThreads(_, _)).WillOnce(Return(1));

            if(_env.is_profiling_enabled == false) {
                EXPECT_CALL(mockApi, GNAWait(_, _, _)).WillOnce(Return(GNA_NOERROR));
            } else {
                EXPECT_CALL(mockApi, GNAWaitPerfRes(_, _, _, _)).WillOnce(Return(GNA_NOERROR));
            }

            if(_env.is_setup_of_omp_theads_expected == true) {
                EXPECT_CALL(mockApi, gmmSetThreads(_)).Times(1);
            } else {
                EXPECT_CALL(mockApi, gmmSetThreads(_)).Times(0);
            }
#else
            EXPECT_CALL(mockApi, Gna2MemoryAlloc(_, _, _)).WillOnce(Invoke([&data](
                uint32_t sizeRequested,
                uint32_t *sizeGranted,
                void **memoryAddress
                ) {
                data.resize(sizeRequested);
                *sizeGranted = sizeRequested;
                *memoryAddress = &data.front();
                return Gna2StatusSuccess;
            }));
#endif
            std::unique_ptr<NNetComponentMatcher> combined(new NNetComponentMatcher());

            for (auto & matchWhat : _env.whatToMatch) {
                switch(matchWhat.type) {
                    case GnaPluginTestEnvironment::matchPrecision :
                        combined->add(new NNetPrecisionMatcher(_env.nnet_precision, INTEL_AFFINE));
                        break;
                    case GnaPluginTestEnvironment::matchProcType :
#if GNA_LIB_VER == 1 // TODO: GNA2: handle new API
                        EXPECT_CALL(mockApi, GNAPropagateForward(_, _, _, _, _, Eq(_env.proc_type)))
                            .WillOnce(Return(GNA_NOERROR));
#endif
                        break;
                    case GnaPluginTestEnvironment::matchPwlInserted :
                        combined->add(new PWLMatcher(_env.matchInserted, matchWhat.matchQuantity, _env.pwlsToMatchWith));
                        break;
                    case GnaPluginTestEnvironment::matchConvInserted:
                        combined->add(new ConvoluionLayerMatcher(_env.matchInserted, matchWhat.matchQuantity));
                        break;
                    case GnaPluginTestEnvironment::matchMaxPoolingInserted:
                        combined->add(new PoolingLayerMatcher(_env.matchInserted, matchWhat.matchQuantity, true));
                        break;
                    case GnaPluginTestEnvironment::matchPwlQuantizeMetrics :
                        combined->add(new PWLQuantizationMetricsMatcher(_env.type,
                                                                        _env.quantization_presicion_threshold,
                                                                        _env.quantization_segments_threshold));
                        break;
                    case GnaPluginTestEnvironment::matchCopyInserted :
                        combined->add(new CopyLayerMatcher(_env.matchInserted, matchWhat.matchQuantity));
                        break;
                    case GnaPluginTestEnvironment::matchDiagonalInserted :
                        combined->add(new DiagLayerMatcher(_env.matchInserted, matchWhat.matchQuantity));
                        break;
                    case GnaPluginTestEnvironment::saveArgs :
#if GNA_LIB_VER == 1 // TODO: GNA2: handle new API
                        EXPECT_CALL(mockApi, GNAPropagateForward(_, _, _, _, _, _))
                            .WillOnce(DoAll(SaveArgPointee<1>(savedNet), Return(GNA_NOERROR)));
#endif
                        break;
                    case GnaPluginTestEnvironment::matchInputData :
                        combined->add(new InputDataMatcher(_env.input_processed));
                        break;
                    case GnaPluginTestEnvironment::fillOutputValues :
                        combined->add(new OutputFiller(_env.fillValue, _env.fillValue));
                        break;
                    case GnaPluginTestEnvironment::matchAffineWeightsTranspose:
                        HasWeightsTranspozed(combined, _env.transposedData, _env.transposeArgs);
                        break;
                    case GnaPluginTestEnvironment::matchAffineWeights:
                        HasWeightsEq(combined, _env.transposedData);
                        break;
                    case GnaPluginTestEnvironment::saveAffineWeights:
                        SaveWeights(combined, _env.transposedData, _env.transposedArgsForSaving);
                        break;
                    default:
#if GNA_LIB_VER == 1 // TODO: GNA2: handle new API
                        EXPECT_CALL(mockApi, GNAPropagateForward(_, _, _, _, _, _))
                            .WillOnce(Return(GNA_NOERROR));
#endif
                        break;
                }
            }
            if (combined && !combined->empty()) {
#if GNA_LIB_VER == 1 // TODO: GNA2: handle new API
                EXPECT_CALL(mockApi, GNAPropagateForward(_, ::testing::MakeMatcher(combined.release()), _, _, _,_)).WillOnce(Return(GNA_NOERROR));
#endif
            }
        }

        loadNetwork();

        if (!inputsInfo.empty()) {
            BlobMap  input_blob_map;
            BlobMap  output_blob_map;
            for (auto info : inputsInfo) {
                size_t current_size = InferenceEngine::details::product(info.second->getTensorDesc().getDims());
                input_blob_map[info.first] = input[info.first];
            }
            size_t offset = 0;
            for (auto info : outputsInfo) {
                size_t current_size = InferenceEngine::details::product(info.second->getTensorDesc().getDims());
                output_blob_map[info.first] = make_shared_blob<float>(
                    { info.second->getPrecision(),
                    {1, details::product(info.second->getTensorDesc().getDims())}, NC },
                    output[info.first]->data(), current_size * sizeof(float));
                offset += current_size;
            }

            plugin.Infer(input_blob_map, output_blob_map);

        } else {
            plugin.Infer(*input.begin()->second, *output.begin()->second);
        }


        if (_env.matchOutput) {

            int outputIdx = 0;
            for (auto && output_ith : output) {
                std::vector<float> actual_output(output_ith.second->size());

                std::copy_n(output_ith.second->cbuffer().as<float *>(),
                            output_ith.second->size(),
                            actual_output.begin());

                for (auto ref = _env.expected_output[outputIdx].begin(); ref != _env.expected_output[outputIdx].end();
                     ref++) {
                    auto idx = std::distance(_env.expected_output[outputIdx].begin(), ref);
                    ASSERT_FLOAT_EQ(*ref, actual_output[idx]) << "at " << idx << " for " << outputIdx << " output";
                }
                outputIdx++;
            }
        }

        std::map<std::string, InferenceEngine::InferenceEngineProfileInfo> perfMap;
        plugin.GetPerformanceCounts(perfMap);

        if(_env.is_profiling_enabled != false) {
            ASSERT_NE(perfMap.empty(),true);
        } else {
            ASSERT_NE(perfMap.empty(),false);
        }

        if (_env.matchThrows) {
            FAIL() << "Test expected exception";
        }

    }
    catch(std::exception &ex) {
        if (!_env.matchThrows) {
            FAIL() << ex.what();
        }
    }
    catch(...) {
        if (!_env.matchThrows) {
            FAIL() << "unknown exception thrown";
        }
    }

}

void GNAPluginCreationMatcher :: match() {
    if (_env.matchThrows) {
        ASSERT_ANY_THROW(GNAPlugin(_env.config));
        return;
    }
    GNAPlugin(_env.config);
}


void GNAPluginAOTMatcher :: match() {
    // matching gna_propagate forward call.
    MockICNNNetwork net;
    CNNNetReader net_reader;
    ASSERT_NO_THROW_IE_EXCEPTION(net_reader.ReadNetwork(_env.model.data(), _env.model.length()));

    size_t weightsSize = 440*3;

    auto weights = make_shared_blob<uint8_t >({ Precision::U8, {weightsSize}, Layout::C });
    weights->allocate();
    fillWeights(weights);
    net_reader.SetWeights(weights);

    GNAPlugin plugin(_env.config);

    TBlob<float> input({ Precision::FP32, {1, 10}, Layout::NC });
    input.allocate();


    TBlob<float> output({ Precision::FP32, {1, 10}, Layout::NC });
    output.allocate();

    if (_env.cb) {
        auto network = net_reader.getNetwork();
        _env.cb(network);
    }

    GNACppApi mockApi;
    std::vector<uint8_t> data(10000);
#if GNA_LIB_VER == 1 // TODO: GNA2: handle new API
    EXPECT_CALL(mockApi, GNAAlloc(_,_,_)).WillOnce(DoAll(SetArgPointee<2>(10000), Return(&data.front())));
    EXPECT_CALL(mockApi, GNADeviceOpenSetThreads(_, _)).WillOnce(Return(1));
#endif
    plugin.LoadNetwork(net_reader.getNetwork());
    plugin.Export(_env.exportedModelFileName);
}


void GNADumpXNNMatcher::load(GNAPlugin & plugin) {

    // matching gna DumpXNN forward call.
    plugin = GNAPlugin(_env.config);

    auto loadNetworkFromIR = [&]() {
        MockICNNNetwork net;
        CNNNetReader net_reader;
        ASSERT_NO_THROW_IE_EXCEPTION(net_reader.ReadNetwork(_env.model.data(), _env.model.length()));

        size_t weightsSize = 440 * 3;

        auto weights = make_shared_blob<uint8_t>({ Precision::U8, {weightsSize}, Layout::C });
        weights->allocate();
        fillWeights(weights);
        net_reader.SetWeights(weights);

        if (_env.cb) {
            auto network = net_reader.getNetwork();
            _env.cb(network);
        }

        plugin.LoadNetwork(net_reader.getNetwork());
    };

    auto loadNetworkFromAOT = [&]() {
        plugin.ImportNetwork(_env.importedModelFileName);
    };

    auto loadNetwork = [&]() {
        if (!_env.importedModelFileName.empty()) {
            loadNetworkFromAOT();
        } else {
            loadNetworkFromIR();
        }
    };

    loadNetwork();
}

void GNADumpXNNMatcher::match() {

    GNACppApi mockApi;
    std::vector<uint8_t> data(10000);
    if (!_env.matchThrows) {
#if GNA_LIB_VER == 1 // TODO: GNA2: handle new API
        EXPECT_CALL(mockApi, GNAAlloc(_,_,_)).WillOnce(DoAll(SetArgPointee<2>(10000), Return(&data.front())));
        EXPECT_CALL(mockApi, GNADeviceOpenSetThreads(_, _)).WillOnce(Return(1));
        intel_gna_model_header header = {};
        header.model_size = 1;
        EXPECT_CALL(mockApi, GNADumpXnn(_, _, _, _, _,_)).WillOnce(DoAll(SetArgPointee<3>(header), Return((void*)::operator new[](1))));
        EXPECT_CALL(mockApi, GNAFree(_)).WillOnce(Return(GNA_NOERROR));
        EXPECT_CALL(mockApi, GNADeviceClose(_)).WillOnce(Return(GNA_NOERROR));
#endif
    }

    try {
        // matching gna DumpXNN forward call.
        GNAPluginNS::GNAPlugin plugin;
        load(plugin);
    }
    catch(std::exception &ex) {
        if (!_env.matchThrows) {
            FAIL() << ex.what();
        }
    }
    catch(...) {
        if (!_env.matchThrows) {
            FAIL() << "unknown exception thrown";
        }
    }

}

void GNAQueryStateMatcher :: match() {

   //  TODO : avoid copy pastes
    GNACppApi mockApi;
    std::vector<uint8_t> data(10000);

    std::shared_ptr<IExecutableNetworkInternal> executer;
    auto loadNetworkFromIR = [&]() {
        MockICNNNetwork net;
        CNNNetReader net_reader;
        ASSERT_NO_THROW_IE_EXCEPTION(net_reader.ReadNetwork(_env.model.data(), _env.model.length()));

        size_t weightsSize = 440 * 3;

        auto weights = make_shared_blob<uint8_t>({ Precision::U8, {weightsSize}, Layout::C });
        weights->allocate();
        fillWeights(weights);
        net_reader.SetWeights(weights);

        if (_env.cb) {
            auto network = net_reader.getNetwork();
            _env.cb(network);
        }

        executer.reset(new GNAExecutableNetwork(net_reader.getNetwork(), _env.config));
    };

    auto loadNetworkFromAOT = [&]() {
        executer.reset(new GNAExecutableNetwork(_env.importedModelFileName, _env.config));
    };

    auto loadNetwork = [&]() {
        if (!_env.importedModelFileName.empty()) {
            return loadNetworkFromAOT();
        } else {
            return loadNetworkFromIR();
        }
    };

#if GNA_LIB_VER == 1 // TODO: GNA2: handle new API
    EXPECT_CALL(mockApi, GNAAlloc(_,_,_)).WillOnce(DoAll(SetArgPointee<2>(10000), Return(&data.front())));
    EXPECT_CALL(mockApi, GNADeviceOpenSetThreads(_, _)).WillOnce(Return(1));
    EXPECT_CALL(mockApi, GNAFree(_)).WillOnce(Return(GNA_NOERROR));
    EXPECT_CALL(mockApi, GNADeviceClose(_)).WillOnce(Return(GNA_NOERROR));
#else
    EXPECT_CALL(mockApi, Gna2MemoryAlloc(_, _, _)).
        WillOnce(DoAll(SetArgPointee<1>(10000), SetArgPointee<2>(&data.front()), Return(Gna2StatusSuccess)));
#endif
    try {
        loadNetwork();
        if (GnaPluginTestEnvironment::kAnyNotNull == _env.numberOfStates) {
            auto states = executer->QueryState();
            ASSERT_NE(states.size(), 0);
            // usually states are callable
            for (auto & state : states) {
                state->Reset();
            }
        } else if (_env.numberOfStates >= 0) {
            ASSERT_EQ(executer->QueryState().size(), _env.numberOfStates);
        } else {
            FAIL() << "number of memory states expectation not set";
        }

    }
    catch(std::exception &ex) {
        FAIL() << ex.what();
    }
    catch(...) {
        FAIL() << "unknown exception thrown";
    }
}


void fillWeights(InferenceEngine::Blob::Ptr weights, std::vector<float> pattern) {
    float * p = weights->buffer().as<float *>();
    float * pEnd = p + weights->byteSize() / sizeof(float);

    for(; p!=pEnd ;) {
        for (int i = 0; i != (weights->byteSize() / sizeof(float) / 3) + 1; i++) {
            for (int j = 0; j != pattern.size() && p != pEnd; j++, p++) {
                *p = pattern[j];
            }
        }
    }
}

