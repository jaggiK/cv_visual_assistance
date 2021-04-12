// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <map>
#include <memory>
#include <atomic>
#include <ie_parameter.hpp>
#include <cpp_interfaces/impl/ie_plugin_internal.hpp>
#include "cldnn_config.h"
#include <api/memory.hpp>
#include <api/engine.hpp>
#include "cldnn_common_utils.h"
#ifdef WIN32
#include <gpu/gpu_context_api_dx.hpp>
#else
#include <gpu/gpu_context_api_va.hpp>
#endif

namespace CLDNNPlugin {
class CLDNNRemoteAllocator;

class CLDNNRemoteBlobImpl : public gpu::details::param_map_obj_getter {
    friend class CLDNNRemoteAllocator;
public:
    enum BlobType {
        BT_EMPTY,
        BT_BUF_INTERNAL,
        BT_BUF_SHARED,
        BT_IMG_SHARED,
        BT_SURF_SHARED,
        BT_DX_BUF_SHARED,
    };

    explicit CLDNNRemoteBlobImpl(gpu::ClContext::Ptr context,
        const cldnn::layout& layout,
        cldnn::shared_handle mem,
        cldnn::shared_surface surf,
        uint32_t plane = 0,
        BlobType mem_type = BT_BUF_INTERNAL);

    void allocate() noexcept;
    bool deallocate() noexcept;
    ParamMap getParams() const;
    std::string getDeviceName() const noexcept;
    std::shared_ptr<RemoteContext> getContext() const noexcept;
    LockedMemory<void> buffer() noexcept;
    LockedMemory<const void> cbuffer() const noexcept;
    LockedMemory<void> rwmap()noexcept;
    LockedMemory<const void> rmap() const noexcept;
    LockedMemory<void> wmap()noexcept;
    const std::shared_ptr<IAllocator> &getAllocator() const noexcept;
    void *getHandle() const noexcept { return _handle; }

    bool is_allocated() const noexcept;
    bool is_locked() const noexcept;
    void allocate_if_needed();
    cldnn::memory& getMemory() { return *m_memObject; }

protected:
    static CLDNNRemoteAllocator m_allocator;
    gpu::ClContext::Ptr m_context;

    // constructor stuff
    cldnn::shared_handle m_mem;
    cldnn::shared_surface m_surf;

    uint32_t m_plane;
    cldnn::layout m_layout;
    BlobType m_mem_type;

    std::unique_ptr<cldnn::memory> m_memObject;

    mutable std::unique_ptr<cldnn::pointer<uint8_t>> lockedHolder;
    mutable void* _handle;
    mutable std::shared_ptr<IAllocator> _allocator;

    void  lock() const;
    void  unlock() const;
};

template<typename TpublicAPI>
class typedCLDNNRemoteBlob : public TpublicAPI {
public:
    using Ptr = std::shared_ptr<typedCLDNNRemoteBlob>;

    explicit typedCLDNNRemoteBlob(gpu::ClContext::Ptr context,
        const TensorDesc& desc,
        const cldnn::layout& layout,
        cldnn::shared_handle mem,
        cldnn::shared_surface surf,
        uint32_t plane,
        CLDNNRemoteBlobImpl::BlobType mem_type)
        : _impl(context, layout, mem,
            surf,
            plane, mem_type), TpublicAPI(desc) {}

    void allocate() noexcept override { _impl.allocate(); }
    bool deallocate() noexcept override { return _impl.deallocate(); }
    ParamMap getParams() const override { return _impl.getParams(); }
    std::string getDeviceName() const noexcept override { return _impl.getDeviceName(); }
    std::shared_ptr<RemoteContext> getContext() const noexcept override { return _impl.getContext(); }
    LockedMemory<void> buffer() noexcept override { return _impl.buffer(); }
    LockedMemory<const void> cbuffer() const noexcept override { return _impl.cbuffer(); }
    LockedMemory<void> rwmap() noexcept override { return _impl.rwmap(); }
    LockedMemory<const void> rmap() const noexcept override { return _impl.rmap(); }
    LockedMemory<void> wmap()noexcept override { return _impl.wmap(); }
    CLDNNRemoteBlobImpl* getImpl() { return &_impl; }

protected:
    const std::shared_ptr<IAllocator> &getAllocator() const noexcept override { return _impl.getAllocator(); }
    void *getHandle() const noexcept override { return _impl.getHandle(); }
    CLDNNRemoteBlobImpl _impl;
};

using CLDNNRemoteCLbuffer = typedCLDNNRemoteBlob<gpu::ClBufferBlob>;
using CLDNNRemoteCLImage2D = typedCLDNNRemoteBlob<gpu::ClImage2DBlob>;
#ifdef WIN32
using CLDNNRemoteD3DBuffer = typedCLDNNRemoteBlob<gpu::D3DBufferBlob>;
using CLDNNRemoteD3DSurface = typedCLDNNRemoteBlob<gpu::D3DSurface2DBlob>;
#else
using CLDNNRemoteVASurface = typedCLDNNRemoteBlob<gpu::VASurfaceBlob>;
#endif

inline CLDNNRemoteBlobImpl* getBlobImpl(gpu::ClBlob* blobPtr) {
#ifdef WIN32
    {
        auto ptr = blobPtr->as<CLDNNRemoteD3DSurface>();
        if (ptr) return ptr->getImpl();
    }
    {
        auto ptr = blobPtr->as<CLDNNRemoteD3DBuffer>();
        if (ptr) return ptr->getImpl();
    }
#else
    {
        auto ptr = blobPtr->as<CLDNNRemoteVASurface>();
        if (ptr) return ptr->getImpl();
    }
#endif
    {
        auto ptr = blobPtr->as<CLDNNRemoteCLbuffer>();
        if (ptr) return ptr->getImpl();
    }
    {
        auto ptr = blobPtr->as<CLDNNRemoteCLImage2D>();
        if (ptr) return ptr->getImpl();
    }
    return nullptr;
}

class CLDNNRemoteAllocator : public IAllocator {
protected:
    friend class CLDNNRemoteBlobImpl;
    std::atomic_flag _lock;
    std::map<void*, const CLDNNRemoteBlobImpl*> m_lockedBlobs;

    void regLockedBlob(void* handle, const CLDNNRemoteBlobImpl* blob);

    void acquire_lock() {
        while (_lock.test_and_set(std::memory_order_acquire)) {}
    }

    void release_lock() {
        _lock.clear(std::memory_order_release);
    }

public:
    using Ptr = std::shared_ptr<CLDNNRemoteAllocator>;

    CLDNNRemoteAllocator() { _lock.clear(std::memory_order_relaxed); }
    /**
    * @brief Maps handle to heap memory accessible by any memory manipulation routines.
    * @return Generic pointer to memory
    */
    void* lock(void* handle, LockOp = LOCK_FOR_WRITE)  noexcept override { return nullptr; };
    /**
    * @brief Unmaps memory by handle with multiple sequential mappings of the same handle.
    * The multiple sequential mappings of the same handle are suppose to get the same
    * result while there isn't a ref counter supported.
    */
    void  unlock(void* handle) noexcept override;
    /**
    * @brief Allocates memory
    * @param size The size in bytes to allocate
    * @return Handle to the allocated resource
    */
    void* alloc(size_t size) noexcept override { return nullptr; }
    /**
    * @brief Releases handle and all associated memory resources which invalidates the handle.
    * @return false if handle cannot be released, otherwise - true.
    */
    bool   free(void* handle) noexcept override { return true; }

    void Release() noexcept override {}
};

class CLDNNExecutionContextImpl : public gpu::details::param_map_obj_getter {
public:
    enum ContextType {
        OCL,
        DEV_SHARED
    };

    using Ptr = std::shared_ptr<CLDNNExecutionContextImpl>;
    using CPtr = std::shared_ptr<const CLDNNExecutionContextImpl>;

    explicit CLDNNExecutionContextImpl(std::shared_ptr<InferencePluginInternal> plugin,
        const ParamMap& params,
        const Config& config = {});

    ParamMap getParams() const;
    std::string getDeviceName() const noexcept;

    std::shared_ptr<cldnn::engine> GetEngine() const { return m_engine; }
    Config& GetConfig() { return m_config; }
    ContextType GetType() const { return m_type; }
    const std::shared_ptr<InferencePluginInternal> GetPlugin() const { return m_plugin; }

    void acquire_lock() {
        while (lock.test_and_set(std::memory_order_acquire)) {}
    }

    void release_lock() {
        lock.clear(std::memory_order_release);
    }

protected:
    std::shared_ptr<cldnn::engine> m_engine;
    gpu_handle_param m_va_display;
    Config m_config;

    ContextType m_type;
    std::shared_ptr<InferencePluginInternal> m_plugin;
    std::atomic_flag lock;
};

template<typename TpublicContextAPI>
class typedCLDNNExecutionContext : public TpublicContextAPI,
    public std::enable_shared_from_this<typedCLDNNExecutionContext<TpublicContextAPI>> {
    template<typename T1, typename T2>
    struct _Key {
        T1 _surf;
        T2 _plane;

        _Key(T1 surf, T2 plane) : _surf(surf), _plane(plane) {}

        bool operator<(const _Key &that) const {
            return _surf < that._surf || (_surf == that._surf && _plane < that._plane);
        }
    };

#ifdef WIN32
    using surf_key = _Key<cldnn::shared_handle, uint32_t>;
#else
    using surf_key = _Key<cldnn::shared_surface, uint32_t>;
#endif
    std::map<surf_key, RemoteBlob::Ptr> shared_surf_reg;
    std::map<cldnn::shared_handle, RemoteBlob::Ptr> shared_obj_reg;

    RemoteBlob::Ptr reuse_surf(const TensorDesc& tensorDesc,
        const ParamMap& params) {
        RemoteBlob::Ptr ret = nullptr;
        uint32_t plane = gpu::details::param_map_obj_getter::_ObjFromParamSimple<uint32_t>(params, GPU_PARAM_KEY(VA_PLANE));
#ifdef WIN32
        cldnn::shared_handle mem = gpu::details::param_map_obj_getter::_ObjFromParamSimple<cldnn::shared_handle>(params, GPU_PARAM_KEY(DEV_OBJECT_HANDLE));
        surf_key skey(mem, plane);
#else
        cldnn::shared_surface surf = gpu::details::param_map_obj_getter::_ObjFromParamSimple<cldnn::shared_surface>(params, GPU_PARAM_KEY(DEV_OBJECT_HANDLE));
        surf_key skey(surf, plane);
#endif
        _impl.acquire_lock();

        // try to locate previously shared surface
        auto itr = shared_surf_reg.find(skey);
        if (itr != shared_surf_reg.end()) {
            ret = itr->second;
        } else {
            // unlickily, not found - create new and insert into registry
            cldnn::layout layout(DataTypeFromPrecision(tensorDesc.getPrecision()),
                ImageFormatFromLayout(tensorDesc.getLayout()),
                CldnnTensorFromIEDims(tensorDesc.getDims()));
            auto smart_this =
                std::dynamic_pointer_cast<gpu::ClContext>
                (std::enable_shared_from_this<typedCLDNNExecutionContext<TpublicContextAPI>>::shared_from_this());
#ifdef WIN32
            ret = std::make_shared<CLDNNRemoteD3DSurface>(smart_this,
                tensorDesc, layout, mem, 0, plane,
                CLDNNRemoteBlobImpl::BlobType::BT_SURF_SHARED);
#else
            ret = std::make_shared<CLDNNRemoteVASurface>(smart_this,
                tensorDesc, layout, nullptr, surf, plane,
                CLDNNRemoteBlobImpl::BlobType::BT_SURF_SHARED);
#endif
            shared_surf_reg[skey] = ret;
        }

        _impl.release_lock();
        return ret;
    }

    RemoteBlob::Ptr reuse_obj(const TensorDesc& tensorDesc,
        cldnn::shared_handle mem,
        CLDNNRemoteBlobImpl::BlobType blob_type) {
        RemoteBlob::Ptr ret = nullptr;

        _impl.acquire_lock();

        // try to locate previously shared object
        auto itr = shared_obj_reg.find(mem);
        if (itr != shared_obj_reg.end()) {
            ret = itr->second;
        } else {
            // unlickily, not found - create new and insert into registry
            cldnn::layout layout(DataTypeFromPrecision(tensorDesc.getPrecision()),
                FormatFromLayout(tensorDesc.getLayout()),
                CldnnTensorFromIEDims(tensorDesc.getDims()));
            auto smart_this =
                std::dynamic_pointer_cast<gpu::ClContext>
                (std::enable_shared_from_this<typedCLDNNExecutionContext<TpublicContextAPI>>::shared_from_this());

            switch (blob_type) {
            case CLDNNRemoteBlobImpl::BlobType::BT_BUF_SHARED:
                ret = std::make_shared<CLDNNRemoteCLbuffer>(smart_this,
                    tensorDesc, layout, mem, 0, 0, blob_type);
                break;
            case CLDNNRemoteBlobImpl::BlobType::BT_IMG_SHARED:
                layout.format = ImageFormatFromLayout(tensorDesc.getLayout());
                ret = std::make_shared<CLDNNRemoteCLImage2D>(smart_this,
                    tensorDesc, layout, mem, 0, 0, blob_type);
                break;
#ifdef WIN32
            case CLDNNRemoteBlobImpl::BlobType::BT_DX_BUF_SHARED:
                ret = std::make_shared<CLDNNRemoteD3DBuffer>(smart_this,
                    tensorDesc, layout, mem, 0, 0, blob_type);
                break;
#endif
            default:
                break;
            }
            shared_obj_reg[mem] = ret;
        }

        _impl.release_lock();
        return ret;
    }

    RemoteBlob::Ptr create_buffer(const TensorDesc& tensorDesc) {
        cldnn::layout layout(DataTypeFromPrecision(tensorDesc.getPrecision()),
            FormatFromLayout(tensorDesc.getLayout()),
            CldnnTensorFromIEDims(tensorDesc.getDims()));
        auto smart_this = std::dynamic_pointer_cast<gpu::ClContext>
            (std::enable_shared_from_this<typedCLDNNExecutionContext<TpublicContextAPI>>::shared_from_this());
        return std::make_shared<CLDNNRemoteCLbuffer>(smart_this,
            tensorDesc,
            layout,
            nullptr, 0, 0,
            CLDNNRemoteBlobImpl::BlobType::BT_BUF_INTERNAL);
    }

    void check_if_shared() {
        if (GetType() != CLDNNExecutionContextImpl::ContextType::DEV_SHARED)
            THROW_IE_EXCEPTION << "Shared context is required to to share this type of memory";
    }
public:
    using Ptr = std::shared_ptr<typedCLDNNExecutionContext>;
    using CPtr = std::shared_ptr<const typedCLDNNExecutionContext>;

    explicit typedCLDNNExecutionContext(std::shared_ptr<InferencePluginInternal> plugin,
        const ParamMap& params,
        const Config& config = {})
        : _impl(plugin, params, config) {}

    ParamMap getParams() const noexcept override { return _impl.getParams(); }
    std::string getDeviceName() const noexcept override { return _impl.getDeviceName(); }

    RemoteBlob::Ptr CreateBlob(const TensorDesc& tensorDesc, const ParamMap& params = {}) override {
        if (params.empty()) {
            // user wants clDNN to allocate blob by itself and return handle
            return create_buffer(tensorDesc);
        } else {
            // user will supply shared object handle
            std::string memTypeStr = gpu::details::param_map_obj_getter::_StrFromParams(params, GPU_PARAM_KEY(SHARED_MEM_TYPE));

            if (GPU_PARAM_VALUE(VA_SURFACE) == memTypeStr) {
                check_if_shared();
                return reuse_surf(tensorDesc, params);
            } else {
                CLDNNRemoteBlobImpl::BlobType blob_type;
                cldnn::shared_handle mem = nullptr;

                if (GPU_PARAM_VALUE(OCL_BUFFER) == memTypeStr) {
                    blob_type = CLDNNRemoteBlobImpl::BlobType::BT_BUF_SHARED;
                    mem = gpu::details::param_map_obj_getter::_ObjFromParamSimple<cldnn::shared_handle>(params, GPU_PARAM_KEY(MEM_HANDLE));
                } else if (GPU_PARAM_VALUE(OCL_IMAGE2D) == memTypeStr) {
                    blob_type = CLDNNRemoteBlobImpl::BlobType::BT_IMG_SHARED;
                    mem = gpu::details::param_map_obj_getter::_ObjFromParamSimple<cldnn::shared_handle>(params, GPU_PARAM_KEY(MEM_HANDLE));
#ifdef WIN32
                } else if (GPU_PARAM_VALUE(DX_BUFFER) == memTypeStr) {
                    blob_type = CLDNNRemoteBlobImpl::BlobType::BT_DX_BUF_SHARED;
                    mem = gpu::details::param_map_obj_getter::_ObjFromParamSimple<cldnn::shared_handle>(params, GPU_PARAM_KEY(DEV_OBJECT_HANDLE));
                    check_if_shared();
#endif
                } else {
                    THROW_IE_EXCEPTION << "Unsupported shared object type " << memTypeStr;
                }

                return reuse_obj(tensorDesc, mem, blob_type);
            }
        }
    }

    std::shared_ptr<cldnn::engine> GetEngine() const { return _impl.GetEngine(); }
    Config& GetConfig() { return _impl.GetConfig(); }
    CLDNNExecutionContextImpl::ContextType GetType() const { return _impl.GetType(); }
    const std::shared_ptr<InferencePluginInternal> GetPlugin() const { return _impl.GetPlugin(); }

    CLDNNExecutionContextImpl* getImpl() { return &_impl; }
protected:
    CLDNNExecutionContextImpl _impl;
};

using CLDNNRemoteCLContext = typedCLDNNExecutionContext<gpu::ClContext>;
#ifdef WIN32
using CLDNNRemoteD3DContext = typedCLDNNExecutionContext<gpu::D3DContext>;
#else
using CLDNNRemoteVAContext = typedCLDNNExecutionContext<gpu::VAContext>;
#endif

inline CLDNNExecutionContextImpl* getContextImpl(gpu::ClContext::Ptr ctxPtr) {
#ifdef WIN32
    {
        auto ptr = ctxPtr->as<CLDNNRemoteD3DContext>();
        if (ptr) return ptr->getImpl();
    }
#else
    {
        auto ptr = ctxPtr->as<CLDNNRemoteVAContext>();
        if (ptr) return ptr->getImpl();
    }
#endif
    {
        auto ptr = ctxPtr->as<CLDNNRemoteCLContext>();
        if (ptr) return ptr->getImpl();
    }
    return nullptr;
}

}  // namespace CLDNNPlugin
