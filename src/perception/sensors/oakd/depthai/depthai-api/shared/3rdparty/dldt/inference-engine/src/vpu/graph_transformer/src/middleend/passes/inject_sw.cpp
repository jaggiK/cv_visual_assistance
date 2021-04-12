// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/middleend/pass_manager.hpp>

#include <vector>
#include <algorithm>
#include <unordered_set>
#include <unordered_map>
#include <limits>
#include <string>
#include <memory>
#include <set>
#include <list>

#include <vpu/middleend/allocator/allocator.hpp>
#include <vpu/compile_env.hpp>

namespace vpu {

void printTo(std::ostream&, const std::list<Stage>::iterator&) {
}

namespace {

class PassImpl final : public Pass {
public:
    void run(const Model& model) override;
};

void PassImpl::run(const Model& model) {
    VPU_PROFILE(injectSw);

    const int nMaxStagesForInjectSw = 30000;
    const auto& env = CompileEnv::get();

    //
    // Collect HW and SW candidates
    //

    if (!env.config.injectSwOps.hasValue() &&
        model->numStages() > nMaxStagesForInjectSw) {
        env.log->warning(
            "Pass [injectSw] SKIPPED : number of stages (%d) is larger than threshold %d",
            model->numStages(), nMaxStagesForInjectSw);
        return;
    }

    StageVector hwStages;
    std::list<Stage> swStages;

    hwStages.reserve(checked_cast<size_t>(model->numStages()));
    for (const auto& stage : model->getStages()) {
        if (stage->category() == StageCategory::HW) {
            hwStages.emplace_back(stage);
        } else if (stage->category() == StageCategory::DMA || stage->category() == StageCategory::SHAVE) {
            if (stage->getSHAVEsRequirements() != StageSHAVEsRequirements::NeedMax) {
                auto it = swStages.emplace(swStages.end(), stage);
                stage->attrs().set<std::list<Stage>::iterator>("swStagesPos", it);
            }
        }
    }

    //
    // Try to merge HW and SW stages
    //

    StageVector swCandidates;

    for (const auto& hwStage : hwStages) {
        swCandidates.clear();

        model->buildStageOrder();

        for (const auto& swStage : swStages) {
            auto hwInd = hwStage->index();
            IE_ASSERT(hwInd >= 0);

            auto swInd = swStage->index();
            IE_ASSERT(swInd >= 0);
            IE_ASSERT(swInd != hwInd);

            //
            // Check execution order
            //

            bool isOK = true;

            if (swInd > hwInd) {
                //
                // SW producer must be executed after HW stage
                //

                for (const auto& swProducer : swStage->prevStages()) {
                    auto swProducerInd = swProducer->index();
                    IE_ASSERT(swProducerInd >= 0);
                    IE_ASSERT(swProducerInd < swInd);

                    if (swProducerInd >= hwInd) {
                        isOK = false;
                        break;
                    }
                }
            } else {
                //
                // HW producer must be executed after SW stage
                //

                for (const auto& hwProducer : hwStage->prevStages()) {
                    auto hwProducerInd = hwProducer->index();
                    IE_ASSERT(hwProducerInd >= 0);
                    IE_ASSERT(hwProducerInd < hwInd);

                    if (hwProducerInd >= swInd) {
                        isOK = false;
                        break;
                    }
                }
            }

            isOK = isOK && hwStage->attrs().getOrDefault<Stage>("loop", nullptr) == swStage->attrs().getOrDefault<Stage>("loop", nullptr);

            if (isOK) {
                swCandidates.push_back(swStage);
            }
        }

        for (const auto& swStage : swCandidates) {
            //
            // Try to inject and check allocation, if it is failed -> revert
            //

            auto edge = model->injectStage()
                    .parentHW(hwStage)
                    .childSW(swStage)
                    .done();

            auto allocRes = runAllocator(model, true);
            if (allocRes.status == AllocationStatus::OK) {
                // TODO: try to merge more than one SW stage?
                break;
            } else {
                model->revertInjection(edge);
            }
        }

        //
        // Remove injected stages from candidates list
        //

        if (const auto injectedStage = hwStage->injectedStage()) {
            auto it = injectedStage->attrs().get<std::list<Stage>::iterator>("swStagesPos");

            IE_ASSERT(it != swStages.end());
            swStages.erase(it);

            injectedStage->attrs().erase("swStagesPos");
        }
    }
}

}  // namespace

Pass::Ptr PassManager::injectSw() {
    return std::make_shared<PassImpl>();
}

}  // namespace vpu
