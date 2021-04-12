// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ie_blob.h>
#include <gtest/gtest.h>
#include <random>
#include <chrono>

#include <cpp/ie_cnn_net_reader.h>
#include <blob_transform.hpp>

using namespace ::testing;
using namespace InferenceEngine;

using ChannelNum = size_t;
using BatchNum = size_t;
using PrecisionType = InferenceEngine::Precision::ePrecision;
using IsInterleaved =  bool;            // true = interleaved, false = deinterleaved.
using Dims = std::vector<size_t>;       // dimensions are in the form of (N x C x D1 x D2 ... Dn), so Dims is vector (D1 x D2 ... Dn)


InferenceEngine::Layout setLayout(IsInterleaved isInterleaved, int dimsSize) {
    if (dimsSize == 3) {
        return (isInterleaved) ?  InferenceEngine::Layout::NDHWC : InferenceEngine::Layout::NCDHW;
    }
    else if (dimsSize == 2) {
        return (isInterleaved) ?  InferenceEngine::Layout::NHWC : InferenceEngine::Layout::NCHW;
    }
    THROW_IE_EXCEPTION << "Can't set layout";
}

//  Support only for 4d and 5d blobs
SizeVector  SetDimVector(BatchNum batchNum, ChannelNum channelNum, Dims dims) {
    if (dims.size() == 2) {
            return SizeVector {batchNum,channelNum, dims[0], dims[1] };
    }
    else if (dims.size() == 3) {
            return SizeVector {batchNum, channelNum, dims[0], dims[1], dims[2] };
    }
    THROW_IE_EXCEPTION << "Can't set dimVector";
}

//  For FP16 and Q78 precision we use int16_t type
InferenceEngine::Blob::Ptr createBlob(InferenceEngine::Precision precision, SizeVector dimsVector, InferenceEngine::Layout layout) {
    InferenceEngine::TensorDesc tensorDesc(precision, dimsVector, layout);
    switch (precision)
    {
        case  InferenceEngine::Precision::FP32 :
             return make_shared_blob<float>(tensorDesc);
        case InferenceEngine::Precision::FP16:
        case InferenceEngine::Precision::I16:
        case InferenceEngine::Precision::Q78:
            return make_shared_blob<int16_t>(tensorDesc);
        case InferenceEngine::Precision::I32:
            return make_shared_blob<int32_t>(tensorDesc);
        case InferenceEngine::Precision::I64:
            return make_shared_blob<int64_t>(tensorDesc);
        case InferenceEngine::Precision::U16:
            return make_shared_blob<uint16_t>(tensorDesc);
        case InferenceEngine::Precision::I8:
        case InferenceEngine::Precision::BIN:
            return make_shared_blob<int8_t>(tensorDesc);
        case InferenceEngine::Precision::U8:
            return make_shared_blob<uint8_t>(tensorDesc);
        default:
            THROW_IE_EXCEPTION<<"Unsupported precision";
    }
}

// returns a random value in the range [0 , elem)
size_t GenerateRandom(size_t elem) {
    size_t result;
    do {
        result = std::floor(std::rand() / (float)RAND_MAX * elem);
    }
    while (result >= elem);
    return result;
}

// returns index of random element of the blob:
// dims is the blob shape, e.g. {1, 3, 640, 480}
// random index[i] lays between 0 and dims[i]-1
SizeVector GenerateRandomVector(SizeVector dims) {
   SizeVector idx(dims.size());

   for (auto i = 0; i < dims.size(); ++i){
        idx[i] = GenerateRandom(dims[i]);
   }
   return idx;
}


void PrintParams (InferenceEngine::Layout layout, SizeVector dims, std::string blobType, InferenceEngine::Precision precision) {
    std::cout <<blobType <<"Blob params: " << layout << ", precision: "<< precision << ", dims: {";

    for(int i = 0; i <  dims.size(); i++){
        std::cout << (i > 0 ? ", ": "") << dims[i];
    }

    std::cout << "}" << std::endl;
}

//  For FP16 and Q78 precision we use int16_t type
template<typename T>
void FillBlobRandom(Blob::Ptr& inputBlob) {
    srand(1);
    auto inputBlobData = inputBlob->buffer().as<T*>();
    for (size_t i = 0; i < inputBlob->size(); i++) {
        inputBlobData[i] = (T) (rand() / (float)RAND_MAX * 100);
    }
}

//  For FP16 and Q78 precision we use int16_t type
void FillBlob(Blob::Ptr& inputBlob) {
    auto precision = inputBlob->getTensorDesc().getPrecision();
    switch (precision)
    {
        case  InferenceEngine::Precision::FP32 :
            return FillBlobRandom<float>(inputBlob);
        case InferenceEngine::Precision::FP16:
        case InferenceEngine::Precision::I16:
        case InferenceEngine::Precision::Q78:
            return FillBlobRandom<int16_t>(inputBlob);
        case InferenceEngine::Precision::I32:
            return FillBlobRandom<int32_t>(inputBlob);
        case InferenceEngine::Precision::I64:
            return FillBlobRandom<int64_t>(inputBlob);
        case InferenceEngine::Precision::U16:
            return FillBlobRandom<uint16_t>(inputBlob);
        case InferenceEngine::Precision::I8:
        case InferenceEngine::Precision::BIN:
            return FillBlobRandom<int8_t>(inputBlob);
        case InferenceEngine::Precision::U8:
            return FillBlobRandom<uint8_t>(inputBlob);
        default:
            THROW_IE_EXCEPTION << "Cant fill blob with \""<<precision<<"\" precision\n";
    }
}


template <typename T>
T GetElem(Blob::Ptr& blob, SizeVector idx) {
    T* src = blob->buffer().as<T*>() + blob->getTensorDesc().getBlockingDesc().getOffsetPadding();

    auto blobLayout = blob->getTensorDesc().getLayout();

    SizeVector strides = blob->getTensorDesc().getBlockingDesc().getStrides();
    if (blobLayout == NHWC || blobLayout == NDHWC) {
        for(int i = 2; i < strides.size(); i++)
            std::swap(strides[1], strides[i]);
    }

    int offset = 0;

    for(int i = 0; i < idx.size(); i++) {
        offset += idx[i] * strides[i];
    }

    return src[offset];
}

int SetExperimentsNum (int blobSize) {
    if(blobSize < 1000) return blobSize;
    else if(blobSize < 10000) return 1000;
    else if(blobSize < 100000) return blobSize / 10;
    else return blobSize / 100;
}

template <typename T>
bool IsCorrectBlobCopy_Impl(Blob::Ptr& srcBlob, Blob::Ptr& dstBlob) {
    EXPECT_TRUE(srcBlob->size() == dstBlob->size());
    int experimentsNum = SetExperimentsNum(srcBlob->size());
    int errorsCount = 0;
    for( ; experimentsNum > 0; --experimentsNum)
    {
        SizeVector randomElemIdx = GenerateRandomVector(srcBlob->getTensorDesc().getDims());
        auto srcElem = GetElem<T>(srcBlob, randomElemIdx);
        auto dstElem = GetElem<T>(dstBlob, randomElemIdx);
        if(srcElem != dstElem) {
           if(errorsCount < 10) {
               errorsCount++;
               std::cout << "ERROR: srcElem = " << srcElem << ", dstElem = " << dstElem << std::endl;
           }
           else
               errorsCount++;
        }
    }
    if (errorsCount > 0) {
        std::cout << "errorsCount = " << errorsCount << std::endl;
    }
    return errorsCount == 0 ;
}


bool IsCorrectBlobCopy(Blob::Ptr& srcBlob, Blob::Ptr& dstBlob) {
    switch (srcBlob->getTensorDesc().getPrecision())
    {
        case  InferenceEngine::Precision::FP32 :
            return IsCorrectBlobCopy_Impl<float>(srcBlob, dstBlob);
        case InferenceEngine::Precision::FP16:
        case InferenceEngine::Precision::I16:
        case InferenceEngine::Precision::Q78:
            return IsCorrectBlobCopy_Impl<int16_t>(srcBlob, dstBlob);
        case InferenceEngine::Precision::I32:
            IsCorrectBlobCopy_Impl<int32_t>(srcBlob, dstBlob);
        case InferenceEngine::Precision::I64:
            return IsCorrectBlobCopy_Impl<int64_t >(srcBlob, dstBlob);
        case InferenceEngine::Precision::U16:
            return IsCorrectBlobCopy_Impl<uint16_t>(srcBlob, dstBlob);
        case InferenceEngine::Precision::I8:
        case InferenceEngine::Precision::BIN:
            return IsCorrectBlobCopy_Impl<int8_t>(srcBlob, dstBlob);
        case InferenceEngine::Precision::U8:
            return IsCorrectBlobCopy_Impl<uint8_t>(srcBlob, dstBlob);
        default:
            return false;
    }
}


class BlobCopyTest : public ::testing::TestWithParam <std::tuple<IsInterleaved, IsInterleaved, BatchNum, ChannelNum, Dims, PrecisionType >> {
};


TEST_P (BlobCopyTest, BlobCopy) {
    IsInterleaved srcIsInterleaved = get<0>(GetParam());
    IsInterleaved dstIsInterleaved = get<1>(GetParam());
    BatchNum batchNum = get<2>(GetParam());
    ChannelNum channelNum = get<3>(GetParam());
    Dims dims = get<4>(GetParam());
    PrecisionType precisionType = get<5>(GetParam());

    SizeVector srcDims = SetDimVector(batchNum, channelNum, dims);
    SizeVector dstDims = SetDimVector(batchNum, channelNum, dims);

    InferenceEngine::Layout srcLayout = setLayout(srcIsInterleaved, dims.size());
    InferenceEngine::Layout dstLayout = setLayout(dstIsInterleaved, dims.size());

    PrintParams (srcLayout, srcDims, "src", precisionType);
    PrintParams (dstLayout, dstDims, "dst", precisionType);

    Blob::Ptr srcBlob = createBlob(precisionType, srcDims, srcLayout);
    Blob::Ptr dstBlob = createBlob(precisionType, dstDims, dstLayout);

    srcBlob->allocate();
    dstBlob->allocate();

    FillBlob(srcBlob);

    auto start =  std::chrono::high_resolution_clock::now();
    blob_copy(srcBlob, dstBlob);
    auto finish =  std::chrono::high_resolution_clock::now();

    std::cout << "Blob_copy execution time : " <<std::chrono::duration_cast<std::chrono::microseconds>(finish - start).count()<<" micros" << std::endl << std::endl;

    ASSERT_TRUE(IsCorrectBlobCopy(srcBlob, dstBlob)) << "'blob_copy' function is't correct";
}
// is interleaved srcBlob?
static std::vector<IsInterleaved> BlobCopy_srcLayoutParam = {
        true, false,
};
// is interleaved dstBlob?
static std::vector<IsInterleaved> BlobCopy_dstLayoutParam = {
        false, true,
};

static std::vector<BatchNum> BlobCopy_BatchNum = {
        1, 3,
};

static std::vector<ChannelNum > BlobCopy_ChannelNum = {
        3, 7,
};

static std::vector<Dims> BlobCopy_Dims = {
        {{10, 20, 30}},
        {{60, 80}},
};

//  The 'blob_copy(4/5)_d' function is a template with the parameter-list  <InferenceEngine::Precision::ePrecision PRC>
//  FP32 is used for cases with the following accuracy:  FP32, I32
//  FP16 is used for cases with the following accuracy:  FP16, U16, I16
//  U8 is used for cases with the following accuracy:  U8, I8
//  Cases with other precision are not supported
static std::vector<PrecisionType> BlobCopy_PrecisionParams = {
        InferenceEngine::Precision::FP32,
        InferenceEngine::Precision::FP16,
        InferenceEngine::Precision::U8,
};


INSTANTIATE_TEST_CASE_P(accuracy, BlobCopyTest,
                        ::testing::Combine(::testing::ValuesIn(BlobCopy_srcLayoutParam),
                           ::testing::ValuesIn(BlobCopy_dstLayoutParam),
                           ::testing::ValuesIn(BlobCopy_BatchNum),
                           ::testing::ValuesIn(BlobCopy_ChannelNum),
                           ::testing::ValuesIn(BlobCopy_Dims),
                           ::testing::ValuesIn(BlobCopy_PrecisionParams)));


