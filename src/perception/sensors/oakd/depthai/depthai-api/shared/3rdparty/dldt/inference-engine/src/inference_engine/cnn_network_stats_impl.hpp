// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_icnn_network.hpp>
#include <ie_icnn_network_stats.hpp>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "description_buffer.hpp"
#include "ie_api.h"
#include "ie_blob.h"
#include "ie_common.h"
#include "ie_data.h"

namespace InferenceEngine {
namespace details {
class INFERENCE_ENGINE_API_CLASS(CNNNetworkStatsImpl): public ICNNNetworkStats {
public:
    CNNNetworkStatsImpl() = default;
    virtual ~CNNNetworkStatsImpl();

public:
    const NetworkStatsMap& getNodesStats() const override;
    void setNodesStats(const NetworkStatsMap& stats) override;
    bool isEmpty() const override {
        return netNodesStats.empty();
    }

    void Release() noexcept override {
        delete this;
    }

protected:
    std::map<std::string, NetworkNodeStatsPtr> netNodesStats;
};

typedef std::shared_ptr<CNNNetworkStatsImpl> CNNNetworkStatsImplPtr;
}  // namespace details
}  // namespace InferenceEngine
