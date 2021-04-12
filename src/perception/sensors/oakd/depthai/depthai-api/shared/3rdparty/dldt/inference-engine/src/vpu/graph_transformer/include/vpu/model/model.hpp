// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>
#include <functional>
#include <set>

#include <ie_icnn_network.hpp>

#include <vpu/model/base.hpp>
#include <vpu/model/edges.hpp>
#include <vpu/model/data.hpp>
#include <vpu/model/stage.hpp>
#include <vpu/utils/enums.hpp>
#include <vpu/utils/io.hpp>
#include <vpu/utils/dot_io.hpp>
#include <vpu/middleend/allocator/allocator.hpp>

#include <utility>

namespace vpu {

//
// Resources
//

// TODO: get rid of `cmxLimit`.

struct Resources final {
    int numCMXSlices = 0;
    int numSHAVEs = 0;
    int cmxLimit = 0;
};

void printTo(std::ostream& os, const Resources& res);
void printTo(DotLabel& lbl, const Resources& res);

//
// Model
//

class ModelObj final :
        public EnableHandle,
        public EnableCustomAttributes {
private:
    // Need to declare here to use decltype
    DataList _dataList;
    mutable StageList _orderedStageList;
    StageNode::IdOrderedSet _initialStages;
    int _stagesIdCount = 0;

    //
    // Main attributes
    //

    VPU_MODEL_ATTRIBUTE(std::string, name, std::string())

    VPU_MODEL_ATTRIBUTE(int, batchSize, 1)

    VPU_MODEL_ATTRIBUTE(InferenceEngine::NetworkStatsMap, nodesStats, {})

public:
    //
    // Constructor
    //

    inline explicit ModelObj(const std::string& name) :
            _dataList(&DataNode::_posInModel),
            _orderedStageList(&StageNode::_posInModel),
            _name(name) {
    }

    //
    // Main attributes
    //

    void setBatchSize(int batchSize);

    inline void setNodesStats(const ie::NetworkStatsMap& stats) { _nodesStats = stats; }

    //
    // Data nodes
    //

    Data addInputData(
            const std::string& name,
            const DataDesc& desc);

    Data addOutputData(
            const std::string& name,
            const DataDesc& desc);

    Data addConstData(
            const std::string& name,
            const DataDesc& desc,
            const DataContent::Ptr& content);

    Data addNewData(
            const std::string& name,
            const DataDesc& desc);

    Data addFakeData();

    Data duplicateData(
            const Data& origData,
            const std::string& postfix,
            const DataDesc& newDesc = DataDesc(),
            const DataContent::Ptr& newContent = nullptr);

    //
    // Stage nodes
    //

    template <class StageImpl>
    Stage addNewStage(
            const std::string& name,
            StageType type,
            const ie::CNNLayerPtr& origLayer,
            const DataVector& inputs,
            const DataVector& outputs);

    Stage duplicateStage(
            const Stage& origStage,
            const std::string& postfix,
            const DataVector& inputs,
            const DataVector& outputs);

    //
    // Stage <-> Data edges
    //

    StageInput addStageInput(
            const Stage& stage,
            const Data& data);

    StageOutput addStageOutput(
            const Stage& stage,
            const Data& data);

    StageTempBuffer addTempBuffer(
            const Stage& stage,
            const DataDesc& desc);

    void replaceStageInput(
            const StageInput& edge,
            const Data& newInput);

    void replaceStageOutput(
            const StageOutput& edge,
            const Data& newOutput);

    //
    // Stage <-> Stage edges
    //

    class InjectStageHelper final {
    public:
        inline InjectStageHelper(InjectStageHelper&&) = default;

        InjectStageHelper(const InjectStageHelper&) = delete;
        InjectStageHelper& operator=(const InjectStageHelper&) = delete;
        InjectStageHelper& operator=(InjectStageHelper&&) = delete;

        ~InjectStageHelper();

        InjectStageHelper& parentHW(const Stage& parent);
        InjectStageHelper& childSW(const Stage& child);

        Injection done();

    private:
        inline explicit InjectStageHelper(const Model& model) : _model(model) {}

    private:
        Model _model;

        Stage _parent;
        Stage _child;

        friend ModelObj;
    };

    inline InjectStageHelper injectStage() { return InjectStageHelper(this); }

    void revertInjection(const Injection& edge);

    //
    // Data<->Data edges
    //

    class DataEdgeHelper final {
    public:
        inline DataEdgeHelper(DataEdgeHelper&&) = default;

        DataEdgeHelper(const DataEdgeHelper&) = delete;
        DataEdgeHelper& operator=(const DataEdgeHelper&) = delete;
        DataEdgeHelper& operator=(DataEdgeHelper&&) = delete;

        ~DataEdgeHelper();

        DataEdgeHelper& parent(const Data& parent);
        DataEdgeHelper& child(const Data& child);

        DataEdgeHelper& mode(SharedDataMode mode);
        DataEdgeHelper& order(SharedDataOrder order);

        DataEdgeHelper& offset(const DimValues& offset);

        DataEdgeHelper& connectionMode(SharedConnectionMode);

        SharedAllocation done();

    private:
        inline explicit DataEdgeHelper(const Model& model) : _model(model) {}

    private:
        Model _model;

        Data _parent;
        Data _child;

        SharedDataMode _mode = SharedDataMode::ROI;
        bool _modeSet = false;

        SharedDataOrder _order = SharedDataOrder::ParentWritesToChild;
        bool _orderSet = false;

        DimValues _offset;
        bool _offsetSet = false;

        SharedConnectionMode _connectionMode = SharedConnectionMode::SINGLE_STAGE;

        friend ModelObj;
    };

    inline DataEdgeHelper connectDatas() {
        return DataEdgeHelper(this);
    }

    void replaceParentData(
            const SharedAllocation& edge,
            const Data& newParent);
    void replaceChildData(
            const SharedAllocation& edge,
            const Data& newChild);

    void disconnectDatas(const SharedAllocation& edge);

    //
    // Nodes removal
    //

    void disconnectStage(const Stage& stage);

    void removeStage(const Stage& stage);

    void removeUnusedData(const Data& data);

    void cleanUp();

    //
    // Stage order
    //

    using StageComparator = std::function<bool(const Stage& stageLeft, const Stage& stageRight)>;

    void buildStageOrder() const;

    void reorderStages(const StageComparator& comparator = {});

    //
    // Nodes accessors
    //

    inline int numDatas() const { return _dataPtrList.size(); }
    inline auto datas() const -> decltype(_dataList | asRange()) {
        return _dataList | asRange();
    }

    inline int numStages() const { return _stagePtrList.size(); }
    inline auto initialStages() const -> decltype(_initialStages | asRange()) {
        return _initialStages | asRange();
    }
    inline auto getStages() const -> decltype(_orderedStageList | asRange()) {
        buildStageOrder();
        return _orderedStageList | asRange();
    }

    //
    // Allocator
    //

    inline Allocator& getAllocator() { return _allocator; }

    template<class Functor>
    void setOnNewStageCallback(Functor&& functor) {
        onNewStageCallback = std::forward<Functor>(functor);
    }

    void unsetOnNewStageCallback() {
        setOnNewStageCallback(nullptr);
    }

private:
    Stage addNewStageImpl(
        const std::string& name,
        StageType type,
        const ie::CNNLayerPtr& origLayer,
        const DataVector& inputs,
        const DataVector& outputs,
        const FuncRef<StagePtr()>& creator);

    Injection injectStageImpl(
            const Stage& parent,
            const Stage& child);

    SharedAllocation connectDatasImpl(
            const Data& parent,
            const Data& child,
            SharedDataMode mode,
            SharedDataOrder order,
            const DimValues& offset,
            SharedConnectionMode connectionMode = SharedConnectionMode::SINGLE_STAGE);

    void runDFS(
            const Stage& stage,
            StageMap<bool>& visitedMap) const;

private:
    DataPtrList _dataPtrList;
    StagePtrList _stagePtrList;

    StageInputPtrList _inEdgePtrList;
    StageOutputPtrList _outEdgePtrList;
    StageTempBufferPtrList _tempBufferEdgePtrList;
    SharedAllocationPtrList _dataEdgePtrList;
    InjectionPtrList _stageEdgePtrList;

    Allocator _allocator;

    mutable bool _resetStageOrder = true;
    StageComparator _nextStagesComparator;

    std::function<void(Stage&)> onNewStageCallback = nullptr;

    friend class InjectStageHelper;
    friend class DataEdgeHelper;
};

template <class StageImpl>
inline Stage ModelObj::addNewStage(
        const std::string& name,
        StageType type,
        const ie::CNNLayerPtr& origLayer,
        const DataVector& inputs,
        const DataVector& outputs) {
    auto newStage = addNewStageImpl(name, type, origLayer, inputs, outputs, []() { return std::make_shared<StageImpl>(); });
    if (onNewStageCallback) {
        onNewStageCallback(newStage);
    }
    return newStage;
}

//
// runAllocator
//

AllocationResult runAllocator(
        const Model& model,
        bool onlyCheckCMX = false);

}  // namespace vpu
