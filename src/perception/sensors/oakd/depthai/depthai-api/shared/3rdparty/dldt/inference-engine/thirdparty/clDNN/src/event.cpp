/*
// Copyright (c) 2016-2019 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
*/

///////////////////////////////////////////////////////////////////////////////////////////////////
#include "api/event.hpp"
#include "event_impl.h"
#include "engine_impl.h"
#include <list>
#include <vector>
#include <algorithm>

namespace cldnn {

event event::create_user_event(const engine& engine, uint32_t net_id) {
    return event(engine.get()->create_user_event(net_id).detach());
}

void event::wait() const {
    _impl->wait();
}

void event::set() const {
    if (auto user_ev = dynamic_cast<user_event*>(_impl))
        user_ev->set();
    else
        throw std::invalid_argument("Event passed to cldnn_set_event should be an user event");
}

void event::set_event_handler(event_handler handler, void* param) const {
    _impl->add_event_handler(handler, param);
}

std::vector<instrumentation::profiling_interval> event::get_profiling_info() const {
    auto interval_list = _impl->get_profiling_info();
    std::vector<instrumentation::profiling_interval> result(interval_list.size());
    std::copy(interval_list.begin(), interval_list.end(), result.begin());
    return result;
}

void event::retain() {
    _impl->add_ref();
}

void event::release() {
    _impl->release();
}

void event_impl::wait() {
    if (_set)
        return;

    // TODO: refactor in context of multiple simultaneous calls (for generic engine)
    wait_impl();
    _set = true;
    return;
}

bool event_impl::is_set() {
    if (_set)
        return true;

    // TODO: refactor in context of multiple simultaneous calls (for generic engine)
    _set = is_set_impl();
    return _set;
}

bool event_impl::add_event_handler(event_handler handler, void* data) {
    if (is_set()) {
        handler(data);
        return true;
    }

    std::lock_guard<std::mutex> lock(_handlers_mutex);
    auto itr = _handlers.insert(_handlers.end(), {handler, data});
    auto ret = add_event_handler_impl(handler, data);
    if (!ret)
        _handlers.erase(itr);

    return ret;
}

const std::list<instrumentation::profiling_interval>& event_impl::get_profiling_info() {
    if (_profiling_captured)
        return _profiling_info;

    _profiling_captured = get_profiling_info_impl(_profiling_info);
    return _profiling_info;
}

void event_impl::call_handlers() {
    std::lock_guard<std::mutex> lock(_handlers_mutex);
    for (auto& pair : _handlers) {
        try {
            pair.first(pair.second);
        } catch (...) {
        }
    }
    _handlers.clear();
}

}  // namespace cldnn
