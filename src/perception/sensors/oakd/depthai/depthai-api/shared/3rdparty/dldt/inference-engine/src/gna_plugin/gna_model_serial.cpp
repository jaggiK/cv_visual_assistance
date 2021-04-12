// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <array>
#include <details/ie_exception.hpp>
#include <ios>
#include <iomanip>
#include <map>
#ifdef _WIN32
#include <malloc.h>
#else
#include <mm_malloc.h>
#endif

#include "gna_plugin.hpp"
#include "gna_model_serial.hpp"

inline void writeNBytes(const void *ptr, uint32_t size, std::ostream & os) {
    os.write(static_cast<const char*>(ptr), size);
}

template <class T>
inline void writeBits(const T & obj, std::ostream & os) {
    os.write(reinterpret_cast<const char *>(&obj), sizeof(T));
}

template <class T>
inline void readBits(T & obj, std::istream & is) {
    is.read(reinterpret_cast<char *>(&obj), sizeof(T));
}

inline void readNBytes(void * ptr, uint32_t size, std::istream & is) {
    is.read(reinterpret_cast<char *>(ptr), size);
}

template <int nBits, class T>
inline void readNBits(T & obj, std::istream & is) {
    std::array<uint8_t, nBits / 8> tmp;
    is.read(reinterpret_cast<char *>(&tmp), nBits / 8);

    obj = * reinterpret_cast<T*>(&tmp.front());
}

inline void * offsetToPointer(void * const base, uint64_t offset) {
    return reinterpret_cast<uint8_t *>(base) + offset;
}

template <class T>
inline void readOffset(T & ptr, void *base,  std::istream & is) {
    uint64_t offset = 0ull;
    readBits(offset, is);
    ptr = reinterpret_cast<T>(offsetToPointer(base, offset));
}

union {
    uint16_t s;
    uint8_t  c[2];
} constexpr static  LECheck {1};

bool is_little_endian() {
    return LECheck.c[0] == 1;
}

const int gna_header_magic = is_little_endian() ?  0x4d414e47 : 0x474e414d;

ModelHeader GNAModelSerial::ReadHeader(std::istream &is) {
    is.exceptions(std::istream::failbit);

    ModelHeader header;
    readBits(header, is);
    if (*reinterpret_cast<int*>(header.gnam) != gna_header_magic) {
        THROW_GNA_EXCEPTION << "Imported file unsupported: magic number should be GNAM(0x474e414d), but was 0x"
                           << std::setfill('0') <<
                           std::hex << std::setw(2) << static_cast<short>(header.gnam[0]) <<
                           std::hex << std::setw(2) << static_cast<short>(header.gnam[1]) <<
                           std::hex << std::setw(2) << static_cast<short>(header.gnam[2]) <<
                           std::hex << std::setw(2) << static_cast<short>(header.gnam[3]);
    }
    if (header.version.major != HEADER_MAJOR) {
        THROW_GNA_EXCEPTION << "Imported file unsupported: major version should be == " << HEADER_MAJOR;
    }
    if (header.headerSize < sizeof(header)) {
        THROW_GNA_EXCEPTION << "Unsupported header size minimal value is : " << sizeof (header) << ", but read: " << header.headerSize;
    }
    /*
     * extra data need to be added into new header and modify check as appropriate
     */

    //  forward compatible
    if (header.headerSize > sizeof(header)) {
        is.seekg(header.headerSize - sizeof(header), std::ios_base::cur);
    }
    return header;
}

#define offsetFromBase(field)\
getOffsetFromBase(field, #field)

#if GNA_LIB_VER == 2

bool IsEmptyTensor(const Gna2Tensor& t) {
    return t.Type == Gna2DataTypeNone &&
        t.Data == nullptr &&
        t.Layout[0] == '\0' &&
        t.Mode == Gna2TensorModeDefault &&
        t.Shape.NumberOfDimensions == 0;
}

const std::map<Gna2OperationType, std::vector<uint32_t>> GnaParamSize{
    {Gna2OperationTypeFullyConnectedAffine, {sizeof(Gna2BiasMode), sizeof(uint32_t)}},
    {Gna2OperationTypeConvolution, {
        sizeof(Gna2Shape),
        sizeof(Gna2BiasMode),
        sizeof(Gna2PoolingMode),
        sizeof(Gna2Shape),
        sizeof(Gna2Shape),
        sizeof(Gna2Shape)}},
};

void GNAModelSerial::Import(void *basePointer, size_t gnaGraphSize, std::istream & is) {
    is.exceptions(std::istream::failbit);

    for (auto operation = gna2Model->Operations; operation != gna2Model->Operations + gna2Model->NumberOfOperations; ++operation) {
        readNBits<32>(operation->Type, is);
        readBits(operation->NumberOfOperands, is);
        operation->Operands = static_cast<Gna2Tensor const **>(gnaUserAllocator(sizeof(Gna2Tensor*) * operation->NumberOfOperands));
        for (uint32_t i = 0; i < operation->NumberOfOperands; i++) {
            Gna2Tensor t{};
            readBits(t, is);
            if (IsEmptyTensor(t)) {
                operation->Operands[i] = nullptr;
            } else {
                operation->Operands[i] = static_cast<Gna2Tensor const *>(gnaUserAllocator(sizeof(Gna2Tensor)));
                t.Data = offsetToPointer(basePointer, reinterpret_cast<uint64_t>(t.Data));
                const_cast<Gna2Tensor&>(*operation->Operands[i]) = t;
            }
        }
        readBits(operation->NumberOfParameters, is);
        switch (operation->Type) {
        case Gna2OperationTypeElementWiseAffine:
        case Gna2OperationTypeFullyConnectedAffine:
        case Gna2OperationTypeConvolution:
            break;
        case Gna2OperationTypeRecurrent:
            THROW_GNA_EXCEPTION << "Importing of recurrent operation not supported";
        case Gna2OperationTypeTransposition:
            THROW_GNA_EXCEPTION << "Importing of transposition operation not supported";
        case Gna2OperationTypeCopy:
            THROW_GNA_EXCEPTION << "Importing of copy operation not supported";
        default:
            THROW_GNA_EXCEPTION << "Importing of unknown GNA operation type(" << operation->Type << ")  not supported";
        }
        if (operation->NumberOfParameters > 0)
            operation->Parameters = static_cast<void **>(gnaUserAllocator(sizeof(void*) * operation->NumberOfParameters));
        else
            operation->Parameters = nullptr;
        for (uint32_t i = 0; i < operation->NumberOfParameters; i++) {
            uint32_t paramSize;
            readBits(paramSize, is);
            if (paramSize == 0) {
                operation->Parameters[i] = nullptr;
                continue;
            }
            operation->Parameters[i] = gnaUserAllocator(paramSize);
            readNBytes(operation->Parameters[i], paramSize, is);

            if (GnaParamSize.at(operation->Type).size() <= i) {
                THROW_GNA_EXCEPTION << "Cannot import parameter of index: " << i;
            }
            if (paramSize != GnaParamSize.at(operation->Type).at(i)) {
                THROW_GNA_EXCEPTION << "Parameter size mismatch on import: " << i;
            }
        }
    }

    // writing memory information
    uint32_t nStates = 0;
    readBits(nStates, is);
    if (pstates != nullptr) {
        pstates->resize(nStates);
    }

    for (int i = 0; i != nStates; i++) {
        void *pSegment;
        readOffset(pSegment, basePointer, is);
        uint32_t segmentSz;
        readBits(segmentSz, is);
        if (pstates) {
            (*pstates)[i] = { pSegment, segmentSz };
        }
    }


    // once structure has been read lets read whole gna graph
    is.read(reinterpret_cast<char*>(basePointer), gnaGraphSize);
}


uint32_t guessGrouping(Gna2Model const& model) {
    if (model.NumberOfOperations == 0 ||
        model.Operations == nullptr ||
        model.Operations[0].Operands == nullptr ||
        model.Operations[0].NumberOfOperands == 0 ||
        model.Operations[0].Operands[0]->Shape.NumberOfDimensions < 2) {
        THROW_GNA_EXCEPTION << "Can not guess grouping";
    }
    return (std::min)(model.Operations[0].Operands[0]->Shape.Dimensions[0], model.Operations[0].Operands[0]->Shape.Dimensions[1]);
}

void GNAModelSerial::Export(void * basePointer, size_t gnaGraphSize, std::ostream & os) const {
    os.exceptions(std::ostream::failbit);

    const std::vector<Gna2Operation>
        layers(gna2Model->Operations, gna2Model->Operations + gna2Model->NumberOfOperations);


    // all offsets will be from this pointer
    auto getOffsetFromBase = [basePointer, &gnaGraphSize](void * pointer, const char * name = nullptr) {
        auto offset = static_cast<uint64_t>(std::distance(reinterpret_cast<uint8_t*>(basePointer), reinterpret_cast<uint8_t*>(pointer)));
        if (offset > gnaGraphSize) {
            THROW_GNA_EXCEPTION << "offset to " << (name == nullptr ? "" : name) << "(0x" << pointer
                << ") not in range segment retuned from GNAAlloc(0x" << basePointer << "-0x"
                << reinterpret_cast<void*>(reinterpret_cast<uint8_t*>(basePointer) + gnaGraphSize) << ")";
        }
        return offset;
    };

    auto getTensorWithProperOffset = [&getOffsetFromBase](const Gna2Tensor& tensor) {
        Gna2Tensor out = tensor;
        out.Data = reinterpret_cast<void*>(getOffsetFromBase(tensor.Data));
        return out;
    };

    auto convert_to_serial = [getOffsetFromBase](const GNAModelSerial::RuntimeEndPoint& ep) {
        ModelHeader::EndPoint out;
        out.elements_count = ep.elements_count;
        out.element_size = ep.element_size;
        out.descriptor_offset = offsetFromBase(ep.descriptor_ptr);
        out.scaleFactor = ep.scaleFactor;
        return out;
    };
    /**
     * writing header
     */
    ModelHeader header;
    header.gnam[0] = 'G';
    header.gnam[1] = 'N';
    header.gnam[2] = 'A';
    header.gnam[3] = 'M';
    header.headerSize = sizeof(ModelHeader);
    header.version.major = HEADER_MAJOR;
    header.version.minor = HEADER_MINOR;
    header.gnaMemSize = gnaGraphSize;
    header.layersCount = layers.size();
    header.nGroup = guessGrouping(*gna2Model);
    header.input = convert_to_serial(input);
    header.output = convert_to_serial(output);

    header.nRotateRows = nRotateRows;
    header.nRotateColumns = nRotateColumns;


    writeBits(header, os);

    for (const auto & layer : layers) {
        writeBits(static_cast<uint32_t>(layer.Type), os);
        writeBits(layer.NumberOfOperands, os);

        for (uint32_t i = 0; i < layer.NumberOfOperands; i++) {
            if (layer.Operands[i] == nullptr)
                writeBits(Gna2Tensor{}, os);
            else
                writeBits(getTensorWithProperOffset(*layer.Operands[i]), os);
        }

        writeBits(layer.NumberOfParameters, os);

        // writing parameters
        switch (layer.Type) {
        case Gna2OperationTypeElementWiseAffine:
        case Gna2OperationTypeFullyConnectedAffine:
        case Gna2OperationTypeConvolution:
            break;
        case Gna2OperationTypeRecurrent:
            THROW_GNA_EXCEPTION << "Exporting of recurrent operation not supported";
        case Gna2OperationTypeTransposition:
            THROW_GNA_EXCEPTION << "Exporting of interleave operation not supported";
        case Gna2OperationTypeCopy:
            THROW_GNA_EXCEPTION << "Exporting of copy operation not supported";
        default:
            THROW_GNA_EXCEPTION << "Exporting of unknown GNA operation type(" << layer.Type << ")  not supported";
        }
        for (uint32_t i = 0; i < layer.NumberOfParameters; i++) {
            if (layer.Parameters[i] == nullptr) {
                writeBits(static_cast<uint32_t>(0), os);
                continue;
            }
            const auto paramSize = GnaParamSize.at(layer.Type).at(i);
            writeBits(paramSize, os);
            writeNBytes(layer.Parameters[i], paramSize, os);
        }
    }
    // writing memory information
    writeBits(static_cast<uint32_t>(states.size()), os);
    for (auto && state : states) {
        writeBits(offsetFromBase(state.first), os);
        writeBits(state.second, os);
    }

    // once structure has been written lets push gna graph
    os.write(reinterpret_cast<char*>(basePointer), gnaGraphSize);
}
#else

void GNAModelSerial::Import(void *basePointer, size_t gnaGraphSize, std::istream & is) {
    is.exceptions(std::istream::failbit);

    auto readPwl = [&is, basePointer](intel_pwl_func_t & value) {
        readBits(value.nSegments, is);
        if (value.nSegments != 0) {
            readOffset(value.pSegments, basePointer, is);
        } else {
            value.pSegments = nullptr;
        }
    };

    for (auto layer = ptr_nnet->pLayers; layer != ptr_nnet->pLayers + ptr_nnet->nLayers; ++layer) {
        readBits(layer->nInputColumns, is);
        readBits(layer->nInputRows, is);
        readBits(layer->nOutputColumns, is);
        readBits(layer->nOutputRows, is);
        readBits(layer->nBytesPerInput, is);
        readBits(layer->nBytesPerOutput, is);
        readBits(layer->nBytesPerIntermediateOutput, is);
        readNBits<32>(layer->nLayerKind, is);

        // reading layers structs
        switch (layer->nLayerKind) {
        case INTEL_AFFINE_DIAGONAL:
        case INTEL_AFFINE: {
            layer->pLayerStruct = _mm_malloc(sizeof(intel_affine_layer_t), 64);
            if (layer->pLayerStruct == nullptr) {
                THROW_GNA_EXCEPTION << "could not allocate memory for intel_affine_layer_t structure.";
            }

            auto &affine = *reinterpret_cast<intel_affine_layer_t *>(layer->pLayerStruct);
            readBits(affine.affine.nBytesPerWeight, is);
            readBits(affine.affine.nBytesPerBias, is);
            readOffset(affine.affine.pWeights, basePointer, is);
            readOffset(affine.affine.pBiases, basePointer, is);
            readPwl(affine.pwl);
            break;
        }
        case INTEL_CONVOLUTIONAL: {
            layer->pLayerStruct = _mm_malloc(sizeof(intel_convolutional_layer_t), 64);
            if (layer->pLayerStruct == nullptr) {
                THROW_GNA_EXCEPTION << "could not allocate memory for intel_convolutional_layer_t structure.";
            }

            auto &convolution = *reinterpret_cast<intel_convolutional_layer_t *>(layer->pLayerStruct);
            readBits(convolution.nFilterCoefficients, is);
            readBits(convolution.nBytesFilterCoefficient, is);
            readBits(convolution.nBytesBias, is);
            readBits(convolution.nFilters, is);
            readBits(convolution.nFeatureMaps, is);
            readBits(convolution.nFeatureMapRows, is);
            readBits(convolution.nFeatureMapColumns, is);
            readBits(convolution.nFilterRows, is);
            readOffset(convolution.pFilters, basePointer, is);
            readOffset(convolution.pBiases, basePointer, is);
            readBits(convolution.nPoolSize, is);
            readBits(convolution.nPoolStride, is);
            readBits(convolution.poolType, is);
            readPwl(convolution.pwl);
            break;
        }

        case INTEL_RECURRENT:
            THROW_GNA_EXCEPTION << "Importing of recurrent layer not supported";
        case INTEL_INTERLEAVE:
            THROW_GNA_EXCEPTION << "Importing of interleave layer not supported";
        case INTEL_DEINTERLEAVE:
            THROW_GNA_EXCEPTION << "Importing of deinterleave layer not supported";
        case INTEL_COPY:
            THROW_GNA_EXCEPTION << "Importing of copy layer not supported";
        default:
            THROW_GNA_EXCEPTION << "Importing of unknown GNA layer kind(" << layer->nLayerKind << ")  not supported";
        }

        // reading offsets of inputs/outputs
        readOffset(layer->pInputs, basePointer, is);
        readOffset(layer->pOutputsIntermediate, basePointer, is);
        readOffset(layer->pOutputs, basePointer, is);
    }

    // writing memory information
    uint32_t nStates = 0;
    readBits(nStates, is);
    if (pstates != nullptr) {
        pstates->resize(nStates);
    }

    for (int i = 0; i != nStates; i++) {
        void *pSegment;
        readOffset(pSegment, basePointer, is);
        uint32_t segmentSz;
        readBits(segmentSz, is);
        if (pstates) {
            (*pstates)[i] = { pSegment, segmentSz };
        }
    }


    // once structure has been read lets read whole gna graph
    is.read(reinterpret_cast<char*>(basePointer), gnaGraphSize);
}

/**
 *
 * @param ptr_nnet
 * @param gnaAllocSize - it can be calculated based on nnet, however it will overcomplicate export
 * about base adress it is relatively easy to calculate
 * @param os
 */

void GNAModelSerial::Export(void * basePointer, size_t gnaGraphSize, std::ostream & os) const {
    os.exceptions(std::ostream::failbit);

    std::vector<intel_nnet_layer_t>
        layers(ptr_nnet->pLayers, ptr_nnet->pLayers + ptr_nnet->nLayers);


    // all offsets will be from this pointer
    auto getOffsetFromBase = [basePointer, &gnaGraphSize](void * pointer, const char * name = nullptr) {
        auto offset = static_cast<uint64_t >(std::distance(reinterpret_cast<uint8_t*>(basePointer), reinterpret_cast<uint8_t*>(pointer)));
        if (offset > gnaGraphSize) {
            THROW_GNA_EXCEPTION << "offset to " << (name == nullptr ? "" : name) << "(0x" << pointer
                               << ") not in range segment retuned from GNAAlloc(0x" << basePointer << "-0x"
                               << reinterpret_cast<void*>(reinterpret_cast<uint8_t*>(basePointer) + gnaGraphSize) << ")";
        }
        return offset;
    };

    auto writePwl = [&os, getOffsetFromBase] (intel_pwl_func_t & value) {
        writeBits(value.nSegments, os);
        // export require certain offset, since offset from base to nullptr cannot be correct, we are not store it at all
        if (value.nSegments != 0) {
            writeBits(offsetFromBase(value.pSegments), os);
        }
    };

    auto convert_to_serial = [getOffsetFromBase](const GNAModelSerial::RuntimeEndPoint& ep){
        ModelHeader::EndPoint out;
        out.elements_count = ep.elements_count;
        out.element_size = ep.element_size;
        out.descriptor_offset = offsetFromBase(ep.descriptor_ptr);
        out.scaleFactor = ep.scaleFactor;
        return out;
    };
    /**
     * writing header
     */
    ModelHeader header;
    header.gnam[0] = 'G';
    header.gnam[1] = 'N';
    header.gnam[2] = 'A';
    header.gnam[3] = 'M';
    header.version.major = HEADER_MAJOR;
    header.version.minor = HEADER_MINOR;
    header.gnaMemSize = gnaGraphSize;
    header.layersCount = layers.size();
    header.nGroup = ptr_nnet->nGroup;
    header.input  = convert_to_serial(input);
    header.output = convert_to_serial(output);
    header.headerSize = sizeof(ModelHeader);
    header.nRotateRows = nRotateRows;
    header.nRotateColumns = nRotateColumns;


    writeBits(header, os);

    for (auto & layer : layers) {
        writeBits(layer.nInputColumns, os);
        writeBits(layer.nInputRows, os);
        writeBits(layer.nOutputColumns, os);
        writeBits(layer.nOutputRows, os);
        writeBits(layer.nBytesPerInput, os);
        writeBits(layer.nBytesPerOutput, os);
        writeBits(layer.nBytesPerIntermediateOutput, os);
        writeBits(static_cast<uint32_t>(layer.nLayerKind), os);

        // writing layers structs
        switch (layer.nLayerKind) {
            case INTEL_AFFINE_DIAGONAL:
            case INTEL_AFFINE: {
                auto &affine = *reinterpret_cast<intel_affine_layer_t *>(layer.pLayerStruct);
                writeBits(affine.affine.nBytesPerWeight, os);
                writeBits(affine.affine.nBytesPerBias, os);
                writeBits(offsetFromBase(affine.affine.pWeights), os);
                writeBits(offsetFromBase(affine.affine.pBiases), os);
                writePwl(affine.pwl);
                break;
            }
            case INTEL_CONVOLUTIONAL: {
                auto &convolution = *reinterpret_cast<intel_convolutional_layer_t *>(layer.pLayerStruct);
                writeBits(convolution.nFilterCoefficients, os);
                writeBits(convolution.nBytesFilterCoefficient, os);
                writeBits(convolution.nBytesBias, os);
                writeBits(convolution.nFilters, os);
                writeBits(convolution.nFeatureMaps, os);
                writeBits(convolution.nFeatureMapRows, os);
                writeBits(convolution.nFeatureMapColumns, os);
                writeBits(convolution.nFilterRows, os);
                writeBits(offsetFromBase(convolution.pFilters), os);
                writeBits(offsetFromBase(convolution.pBiases), os);
                writeBits(convolution.nPoolSize, os);
                writeBits(convolution.nPoolStride, os);
                writeBits(convolution.poolType, os);
                writePwl(convolution.pwl);
                break;
            }

            case INTEL_RECURRENT:
                THROW_GNA_EXCEPTION << "Exporting of recurrent layer not supported";
            case INTEL_INTERLEAVE:
                THROW_GNA_EXCEPTION << "Exporting of interleave layer not supported";
            case INTEL_DEINTERLEAVE:
                THROW_GNA_EXCEPTION << "Exporting of deinterleave layer not supported";
            case INTEL_COPY:
                THROW_GNA_EXCEPTION << "Exporting of copy layer not supported";
            default:
                THROW_GNA_EXCEPTION << "Exporting of unknown GNA layer kind(" << layer.nLayerKind << ")  not supported";
        }

        // writing offsets from base.
        writeBits(offsetFromBase(layer.pInputs), os);
        writeBits(offsetFromBase(layer.pOutputsIntermediate), os);
        writeBits(offsetFromBase(layer.pOutputs), os);
    }
    // writing memory information
    writeBits(static_cast<uint32_t>(states.size()), os);
    for (auto && state : states) {
        writeBits(offsetFromBase(state.first), os);
        writeBits(state.second, os);
    }

    // once structure has been written lets push gna graph
    os.write(reinterpret_cast<char*>(basePointer), gnaGraphSize);
}

#endif
