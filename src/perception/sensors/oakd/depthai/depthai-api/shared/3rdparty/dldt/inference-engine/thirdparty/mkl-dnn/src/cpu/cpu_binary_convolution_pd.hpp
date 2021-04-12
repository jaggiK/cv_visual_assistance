/*******************************************************************************
* Copyright 2019 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#ifndef CPU_BINARY_CONVOLUTION_FWD_PD_HPP
#define CPU_BINARY_CONVOLUTION_FWD_PD_HPP

#include <assert.h>

#include "c_types_map.hpp"
#include "binary_convolution_pd.hpp"
#include "cpu_engine.hpp"
#include "cpu_memory.hpp"
#include "cpu_primitive.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

struct _cpu_binary_convolution_fwd_pd_t: public _binary_convolution_fwd_pd_t {
    using cpu_memory_pd_t = cpu_memory_t::pd_t;

    _cpu_binary_convolution_fwd_pd_t(engine_t *engine,
            const typename _cpu_binary_convolution_fwd_pd_t::base_desc_t *adesc,
            const primitive_attr_t *attr,
            const typename _cpu_binary_convolution_fwd_pd_t::base_class *hint_fwd_pd)
        : _binary_convolution_fwd_pd_t(engine, adesc, attr, hint_fwd_pd)
        , src_pd_(this->engine_, &this->cdesc_().src_desc)
        , dst_pd_(this->engine_, &this->cdesc_().dst_desc)
        , weights_pd_(this->engine_, &this->cdesc_().weights_desc) {}
    virtual ~_cpu_binary_convolution_fwd_pd_t() {}

    virtual const cpu_memory_pd_t *src_pd(int index = 0) const override
    { return index == 0 ? &src_pd_ : nullptr; }
    virtual const cpu_memory_pd_t *dst_pd(int index = 0) const override
    { return index == 0 ? &dst_pd_ : nullptr; }
    virtual const cpu_memory_pd_t *weights_pd(int index = 0) const override {
        if (index == 0) return &weights_pd_;
        return nullptr;
    }

protected:
    cpu_memory_pd_t src_pd_, dst_pd_;
    cpu_memory_pd_t weights_pd_;

    inline memory_format_t src_format()
    {
        using namespace memory_format;
        return utils::pick(this->cdesc_().src_desc.ndims - 3, ncw, nchw, ncdhw);
    }
    inline memory_format_t wei_format()
    {
        using namespace memory_format;
        return this->with_groups()
            ? utils::pick(this->cdesc_().src_desc.ndims - 3, goiw, goihw, goidhw)
            : utils::pick(this->cdesc_().src_desc.ndims - 3, oiw, oihw, oidhw);
    }

    virtual status_t set_default_params() {
        using namespace memory_format;
        if (src_pd_.desc()->format == any)
            CHECK(src_pd_.set_format(src_format()));
        if (dst_pd_.desc()->format == any)
            CHECK(dst_pd_.set_format(src_pd_.desc()->format));
        if (weights_pd_.desc()->format == any)
            CHECK(weights_pd_.set_format(wei_format()));
        return status::success;
    }
};

using cpu_binary_convolution_fwd_pd_t = _cpu_binary_convolution_fwd_pd_t;

}
}
}

#endif
