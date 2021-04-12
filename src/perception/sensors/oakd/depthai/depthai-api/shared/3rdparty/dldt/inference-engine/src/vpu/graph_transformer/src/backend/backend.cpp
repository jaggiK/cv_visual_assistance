// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/backend/backend.hpp>

#include <memory>
#include <string>
#include <vector>
#include <utility>

#include <vpu/compile_env.hpp>
#include <vpu/utils/file_system.hpp>
#include <vpu/utils/io.hpp>

namespace vpu {

void BackEnd::extractDataInfo(
        const Model& model,
        DataInfo& inputInfo,
        DataInfo& outputInfo) {
    for (const auto& data : model->datas()) {
        if (DataUsage::Input == data->usage()) {
            IE_ASSERT(inputInfo.offset.count(data->name()) == 0);

            auto ioBufferOffset = data->attrs().get<int>("ioBufferOffset");
            IE_ASSERT(ioBufferOffset + data->totalByteSize() <= inputInfo.totalSize);

            inputInfo.offset[data->name()] = ioBufferOffset;
        } else if (DataUsage::Output == data->usage()) {
            IE_ASSERT(outputInfo.offset.count(data->name()) == 0);

            auto ioBufferOffset = data->attrs().get<int>("ioBufferOffset");
            IE_ASSERT(ioBufferOffset + data->totalByteSize() <= outputInfo.totalSize);

            outputInfo.offset[data->name()] = ioBufferOffset;
        }
    }
}

CompiledGraph::Ptr BackEnd::build(
        const Model& model,
        const std::vector<ie::CNNLayerPtr>& allLayers) {
    auto compiledGraph = std::make_shared<CompiledGraph>();

    compiledGraph->networkName = model->name();
    compiledGraph->networkBatch = model->batchSize();

    auto usedMemory = model->attrs().get<UsedMemory>("usedMemory");
    compiledGraph->inputBufSize = usedMemory.input;
    compiledGraph->outputBufSize = usedMemory.output;

    compiledGraph->numShaves = checked_cast<std::uint32_t>(model->attrs().get<Resources>("resources").numSHAVEs);
    compiledGraph->numSlices = checked_cast<std::uint32_t>(model->attrs().get<Resources>("resources").numCMXSlices);

    compiledGraph->inputInfo.totalSize  = usedMemory.input;
    compiledGraph->outputInfo.totalSize = usedMemory.output;

    extractDataInfo(model, compiledGraph->inputInfo, compiledGraph->outputInfo);

    serialize(model, compiledGraph->blob, compiledGraph->blobHeader, compiledGraph->numActiveStages);
    getMetaData(model, allLayers, compiledGraph->graphMeta);

    return compiledGraph;
}

void BackEnd::dumpModel(
        const Model& model,
        const std::string& postfix) {
    const auto replaceBadCharacters = [](std::string str) {
        for (auto& ch : str) {
            if (!std::isalnum(ch)) {
                ch = '_';
            }
        }
        return str;
    };

    const auto& env = CompileEnv::get();

    std::string fileName;

    if (!env.config.dumpInternalGraphFileName.empty()) {
        fileName = fileNameNoExt(env.config.dumpInternalGraphFileName);
    } else if (!env.config.dumpInternalGraphDirectory.empty()) {
        fileName = formatString(
            "%s/vpu_graph_%f%f%i_%s",
            env.config.dumpInternalGraphDirectory,
            std::setw(2), std::setfill('0'),
            model->attrs().get<int>("index"),
            replaceBadCharacters(model->name()));
    } else {
        return;
    }

    if (!postfix.empty()) {
        if (!env.config.dumpAllPasses) {
            return;
        }

        fileName = formatString("%s_%s", fileName, replaceBadCharacters(postfix));
    }

    const auto dotFileName = formatString("%s.dot", fileName);
    dumpModelToDot(model, dotFileName);
}

}  // namespace vpu
