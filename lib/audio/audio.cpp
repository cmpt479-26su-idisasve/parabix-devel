#include "audio/audio.h"
#include <iostream>
#include <kernel/io/source_kernel.h>
#include <kernel/core/kernel_builder.h>
#include <llvm/IR/Value.h>
#include <kernel/streamutils/stream_shift.h>
#include <kernel/core/relationship.h>
#include <kernel/basis/s2p_kernel.h>
#include <kernel/basis/p2s_kernel.h>
#include <kernel/streamutils/deletion.h>
#include <kernel/streamutils/stream_select.h>
#include "audio/stream_manipulation.h"
#include <llvm/IR/Intrinsics.h>
#include <pablo/bixnum/bixnum.h>

#define SHOW_STREAM(name)           \
    if (codegen::EnableIllustrator) \
    P->captureBitstream(#name, name)
#define SHOW_BIXNUM(name)           \
    if (codegen::EnableIllustrator) \
    P->captureBixNum(#name, name)
#define SHOW_BYTES(name)            \
    if (codegen::EnableIllustrator) \
    P->captureByteData(#name, name)

#define NUM_HEADER_BYTES 44

namespace audio
{
    void ExtractWAVData(
        const std::unique_ptr<ProgramBuilder> &P,
        Scalar *const fileDescriptor,
        unsigned int numChannels,
        unsigned int numSamples,
        unsigned int sampleRate,
        unsigned int bitPerSample,
        const bool includedHeader,
        StreamSet *&outputDataStreams)
    {
        if (numChannels != 1 && numChannels != 2)
        {
            throw std::invalid_argument("Error: numChannels " + std::to_string(numChannels) + " is not valid");
        }

        StreamSet *ByteStream = P->CreateStreamSet(1, 8);
        P->CreateKernelCall<ReadSourceKernel>(fileDescriptor, ByteStream);

        StreamSet *TrimByteStream;
        if (includedHeader)
        {
            StreamSet *BitStreams = P->CreateStreamSet(8);
            P->CreateKernelCall<S2PKernel>(ByteStream, BitStreams);
            StreamSet *ones = P->CreateStreamSet(1);
            StreamSet *shiftedOnes = P->CreateStreamSet(1);
            P->CreateKernelCall<CreateOnes>(BitStreams, ones);
            P->CreateKernelCall<ShiftBack>(ones, shiftedOnes, NUM_HEADER_BYTES);
            StreamSet *headerMask = P->CreateStreamSet(1);
            P->CreateKernelCall<ShiftForward>(shiftedOnes, headerMask, NUM_HEADER_BYTES);
            StreamSet *TrimBitStreams = P->CreateStreamSet(8);
            FilterByMask(P, headerMask, BitStreams, TrimBitStreams);
            TrimByteStream = P->CreateStreamSet(1, 8);
            P->CreateKernelCall<P2SKernel>(TrimBitStreams, TrimByteStream);
        }
        else
        {
            TrimByteStream = ByteStream;
        }

        SHOW_BYTES(TrimByteStream);
        StreamSet *DataStreams = P->CreateStreamSet(numChannels, 8);
        if (numChannels == 2)
        {
            P->CreateKernelCall<SplitKernel>(TrimByteStream, DataStreams, bitPerSample);
        }
        else
        {
            DataStreams = ByteStream;
        }
        outputDataStreams = DataStreams;
    }

    void readWAVHeader(const int &fd,
                       unsigned int &numChannels,
                       unsigned int &sampleRate,
                       unsigned int &bitPerSample,
                       unsigned int &numSamples)
    {
        char temp_buffer[11];

        // validate file format
        ssize_t bytesRead = read(fd, &temp_buffer, 8);
        temp_buffer[4] = '\0';
        if (bytesRead <= 0 || std::string(temp_buffer) != "RIFF")
        {
            throw std::runtime_error("Error parsing file format: Chunk ID does not match Wav format.");
        }

        bytesRead = read(fd, &temp_buffer, 4);
        temp_buffer[4] = '\0';
        if (bytesRead <= 0 || std::string(temp_buffer) != "WAVE")
        {
            throw std::runtime_error("Error parsing file format: Header does not match Wav format.");
        }

        bytesRead = read(fd, &temp_buffer, 10);
        temp_buffer[4] = '\0';
        if (bytesRead <= 0 || std::string(temp_buffer) != "fmt\x20")
        {
            throw std::runtime_error("Error parsing file format: Subchunk ID does not match Wav format.");
        }

        // read 2 bytes for num channels
        uint16_t num_channels;
        bytesRead = read(fd, reinterpret_cast<char *>(&num_channels), 2);
        if (bytesRead <= 0)
        {
            throw std::runtime_error("Error parsing file format: Cannot interpret number channels.");
        }
        numChannels = num_channels;

        // read 4 bytes for sample rate
        bytesRead = read(fd, reinterpret_cast<char *>(&sampleRate), 4);
        if (bytesRead <= 0)
        {
            throw std::runtime_error("Error parsing file format: Cannot interpret sample rate.");
        }

        // skip the next 6 bytes
        bytesRead = read(fd, temp_buffer, 6);
        if (bytesRead <= 0)
        {
            throw std::runtime_error("Error parsing file format.");
        }

        // read 4 bytes for num bits per sample
        uint16_t bits_per_sample;
        bytesRead = read(fd, reinterpret_cast<char *>(&bits_per_sample), 2);
        if (bytesRead <= 0)
        {
            throw std::runtime_error("Error parsing file format.");
        }
        bitPerSample = bits_per_sample;

        bytesRead = read(fd, temp_buffer, 4);
        temp_buffer[4] = '\0';
        if (bytesRead <= 0 || std::string(temp_buffer) != "data")
        {
            throw std::runtime_error("Error parsing file format: Subchunk 2 ID does not match Wav format.");
        }

        int subchunk2_size;
        bytesRead = read(fd, reinterpret_cast<char *>(&subchunk2_size), 4);
        if (bytesRead <= 0)
        {
            throw std::runtime_error("Error parsing file format: Cannot interpret subchunk 2 size.");
        }

        // copy over the data buffer
        std::vector<u_char> data_buffer(subchunk2_size);
        bytesRead = read(fd, reinterpret_cast<char *>(&data_buffer[0]), subchunk2_size);

        if (bytesRead <= 0)
        {
            throw std::runtime_error("Error parsing file format: Cannot interpret data chunk.");
        }

        numSamples = subchunk2_size / (numChannels * bitPerSample / 8);
    }

    void S2P(
        const std::unique_ptr<ProgramBuilder> &P,
        unsigned int bitPerSample,
        StreamSet * const inputStream,
        StreamSet *&outputStreams)
    {
        if (bitPerSample == 16)
        {
            StreamSet * ParallelStreams = P->CreateStreamSet(2, 8);
            P->CreateKernelCall<SplitKernel>(inputStream, ParallelStreams, 8);
            
            std::vector<StreamSet *> BitsBasis;
            BitsBasis.reserve(2);
            for (int i=0;i<2;++i)
            {
                BitsBasis.push_back(P->CreateStreamSet(8));
            }
            
            for (int i=0;i<2;++i)
            {
                StreamSet *SingleStream = P->CreateStreamSet(1, 8);
                P->CreateKernelCall<IStreamSelect>(SingleStream, Select(ParallelStreams, {(unsigned)i}));
                P->CreateKernelCall<S2PKernel>(SingleStream, BitsBasis[i]);
            }

            P->CreateKernelCall<ConcatenateKernel>(BitsBasis[0], BitsBasis[1], outputStreams);
        }
        else if (bitPerSample == 8)
        {
            P->CreateKernelCall<S2PKernel>(inputStream, outputStreams);
        }
        else 
        {
            throw std::invalid_argument("Only 8 and 16 bit depths are supported");
        }
    }

    FlexS2PKernel::FlexS2PKernel(KernelBuilder &b, const unsigned int bitsPerSample, StreamSet *const inputStream, StreamSet *const outputStreams) 
        :
         bitsPerSample(bitsPerSample),
         MultiBlockKernel(b, "FlexS2PKernel_" + std::to_string(bitsPerSample),
                           {Binding{"inputStream", inputStream, FixedRate((bitsPerSample < 8) ? 1 : bitsPerSample / 8)}},
                           {Binding{"outputStreams", outputStreams, FixedRate((bitsPerSample < 8) ? 8 / bitsPerSample : 1)}}, {}, {}, {})
    {
        if (bitsPerSample != 4 && bitsPerSample % 8 != 0)
        {
            throw std::invalid_argument("bitsPerSample: " + std::to_string(bitsPerSample) + ". bitsPerSample must be 4 or multiple of 8");
        }
        if (inputStream->getNumElements() != 1)
        {
            throw std::invalid_argument("numInputStreams: " + std::to_string(inputStream->getNumElements()) + ". Input must be a mono stream");
        }
    }

    void FlexS2PKernel::generateMultiBlockLogic(KernelBuilder &b, Value *const numOfStrides)
    {
        const unsigned fw = 1;
        const unsigned inputRate = (bitsPerSample < 8) ? 8 / bitsPerSample : 1;
        const unsigned outputRate = (bitsPerSample < 8) ? 8 / bitsPerSample : 1;
        const unsigned inputPacksPerStride = fw * inputRate;
        const unsigned outputPacksPerStride = fw * outputRate;
        const unsigned packSize = b.getBitBlockWidth();
        const unsigned numElementsPerPack = packSize / bitsPerSample;

        BasicBlock *entry = b.GetInsertBlock();
        BasicBlock *loop = b.CreateBasicBlock("loop");
        BasicBlock *exit = b.CreateBasicBlock("exit");
        Constant *const ZERO = b.getSize(0);

        Type *vecType = FixedVectorType::get(b.getIntNTy(bitsPerSample), static_cast<unsigned>(numElementsPerPack));
        Type *vec1Type = FixedVectorType::get(b.getIntNTy(1), static_cast<unsigned>(numElementsPerPack));

        Value *numOfBlocks = numOfStrides;
        b.CreateBr(loop);
        b.SetInsertPoint(loop);
        PHINode *blockOffsetPhi = b.CreatePHI(b.getSizeTy(), 2);
        blockOffsetPhi->addIncoming(ZERO, entry);
        Value* bytepack[inputPacksPerStride];
        for (unsigned i = 0; i < inputPacksPerStride; ++i)
        {
            bytepack[i] = b.loadInputStreamPack("inputStream", ZERO, b.getInt32(i), blockOffsetPhi);
            bytepack[i] = b.CreateBitCast(bytepack[i], vecType);
        }

        for (unsigned i = 0;i<outputPacksPerStride;++i)
        {
            for (unsigned j = 0;j<bitsPerSample;++j)
            {   
                Value *mask = b.getSplat(numElementsPerPack, ConstantInt::get(b.getIntNTy(bitsPerSample), 1 << j));
                Value *extractedBit = b.simd_pext(bitsPerSample, bytepack[i], mask);
                extractedBit = b.CreateZExtOrTrunc(extractedBit, vec1Type);
                b.storeOutputStreamPack("outputStreams", b.getSize(j), b.getInt32(i), blockOffsetPhi, extractedBit);
            }
        }

        Value *nextBlk = b.CreateAdd(blockOffsetPhi, b.getSize(1));
        blockOffsetPhi->addIncoming(nextBlk, loop);
        Value *moreToDo = b.CreateICmpNE(nextBlk, numOfBlocks);

        b.CreateCondBr(moreToDo, loop, exit);
        b.SetInsertPoint(exit);
    }

    Stereo2MonoKernel::Stereo2MonoKernel(KernelBuilder &b, const unsigned int bitsPerSample, StreamSet *const inputStreams, StreamSet *const outputStreams)
        : MultiBlockKernel(b, "Stereo2MonoKernel_" + std::to_string(bitsPerSample),
                           {Binding{"inputStreams", inputStreams, FixedRate(1)}},
                           {Binding{"outputStreams", outputStreams, FixedRate(1)}}, {}, {}, {}),
          bitsPerSample(bitsPerSample), numInputStreams(inputStreams->getNumElements())
    {
        if (numInputStreams != 2)
        {
            throw std::invalid_argument("numInputStreams: " + std::to_string(numInputStreams) + ". Input must be a stereo audio stream");
        }
    }

    void Stereo2MonoKernel::generateMultiBlockLogic(KernelBuilder &b, Value *const numOfStrides)
    {
        const unsigned fw = 8;
        const unsigned inputPacksPerStride = fw * 1;
        const unsigned packSize = b.getBitBlockWidth();
        const unsigned numElementsPerPack = packSize / bitsPerSample;

        BasicBlock *entry = b.GetInsertBlock();
        BasicBlock *loop = b.CreateBasicBlock("loop");
        BasicBlock *exit = b.CreateBasicBlock("exit");
        Constant *const ZERO = b.getSize(0);
        Constant *const ONE = b.getSize(1);

        Type *vec16x16Type = FixedVectorType::get(b.getIntNTy(bitsPerSample), static_cast<unsigned>(numElementsPerPack));
        Value *shiftAmount = b.getSplat(numElementsPerPack, ConstantInt::get(b.getIntNTy(bitsPerSample), 1));

        Value *numOfBlocks = numOfStrides;
        b.CreateBr(loop);
        b.SetInsertPoint(loop);
        PHINode *blockOffsetPhi = b.CreatePHI(b.getSizeTy(), 2);
        blockOffsetPhi->addIncoming(ZERO, entry);

        for (unsigned i = 0; i < inputPacksPerStride; ++i)
        {
            Value *bytepack_1, *bytepack_2;
            bytepack_1 = b.loadInputStreamPack("inputStreams", ZERO, b.getInt32(i), blockOffsetPhi);
            bytepack_1 = b.CreateBitCast(bytepack_1, vec16x16Type);
            bytepack_2 = b.loadInputStreamPack("inputStreams", ONE, b.getInt32(i), blockOffsetPhi);
            bytepack_2 = b.CreateBitCast(bytepack_2, vec16x16Type);
            Value *sumBytePack = b.CreateAdd(bytepack_1, bytepack_2);
            Value *meanBytePack = b.CreateAShr(sumBytePack, shiftAmount);
            b.storeOutputStreamPack("outputStreams", ZERO, b.getInt32(i), blockOffsetPhi, meanBytePack);
        }

        Value *nextBlk = b.CreateAdd(blockOffsetPhi, b.getSize(1));
        blockOffsetPhi->addIncoming(nextBlk, loop);
        Value *moreToDo = b.CreateICmpNE(nextBlk, numOfBlocks);

        b.CreateCondBr(moreToDo, loop, exit);
        b.SetInsertPoint(exit);
    }

    AmplifyKernel::AmplifyKernel(KernelBuilder &b, const unsigned int bitsPerSample, StreamSet *const inputStreams, const unsigned int &factor, StreamSet *const outputStreams)
        : MultiBlockKernel(b, "AmplifyKernel_" + std::to_string(factor) + "_" + std::to_string(inputStreams->getNumElements()) + "_" + std::to_string(bitsPerSample),
                           {Binding{"inputStreams", inputStreams, FixedRate(1)}},
                           {Binding{"outputStreams", outputStreams, FixedRate(1)}}, {}, {}, {}),
          bitsPerSample(bitsPerSample), numInputStreams(inputStreams->getNumElements()), factor(factor)
    {
        if (inputStreams->getNumElements() != outputStreams->getNumElements())
        {
            throw std::invalid_argument("numInputStreams: " + std::to_string(inputStreams->getNumElements()) + " != numOutputStreams: " + std::to_string(outputStreams->getNumElements()));
        }
    }

    void AmplifyKernel::generateMultiBlockLogic(KernelBuilder &b, Value *const numOfStrides)
    {
        const unsigned fw = 8;
        const unsigned inputPacksPerStride = fw * 1;
        const unsigned packSize = b.getBitBlockWidth();
        const unsigned numElementsPerPack = packSize / bitsPerSample;

        BasicBlock *entry = b.GetInsertBlock();
        BasicBlock *loop = b.CreateBasicBlock("loop");
        BasicBlock *exit = b.CreateBasicBlock("exit");
        Constant *const ZERO = b.getSize(0);
        
        Type *vec16x16Type = FixedVectorType::get(b.getIntNTy(bitsPerSample), static_cast<unsigned>(numElementsPerPack));

        Function *smulWithOverflow = llvm::Intrinsic::getDeclaration(getModule(), llvm::Intrinsic::smul_with_overflow, {vec16x16Type});
        Value *factorVec = b.getSplat(numElementsPerPack, ConstantInt::get(b.getIntNTy(bitsPerSample), factor));
        Value *zeroVec = b.getSplat(numElementsPerPack, ConstantInt::get(b.getIntNTy(bitsPerSample), 0));
        Value *minVal = b.getSplat(numElementsPerPack, ConstantInt::get(b.getIntNTy(bitsPerSample), -(1 << (bitsPerSample - 1))));
        Value *maxVal = b.getSplat(numElementsPerPack, ConstantInt::get(b.getIntNTy(bitsPerSample), (1 << (bitsPerSample - 1)) - 1));

        Value *numOfBlocks = numOfStrides;
        b.CreateBr(loop);
        b.SetInsertPoint(loop);
        PHINode *blockOffsetPhi = b.CreatePHI(b.getSizeTy(), 2);
        blockOffsetPhi->addIncoming(ZERO, entry);

        Value *bytepack[inputPacksPerStride];
        Value *amplifiedBytepack[inputPacksPerStride];
        Value *clipBytepack[inputPacksPerStride];
        Value *signMask[inputPacksPerStride];

        for (unsigned k = 0; k < numInputStreams; ++k)
        {
            Constant *const STREAMINDEX = b.getSize(k);
            for (unsigned i = 0; i < inputPacksPerStride; ++i)
            {
                bytepack[i] = b.loadInputStreamPack("inputStreams", STREAMINDEX, b.getInt32(i), blockOffsetPhi);
                bytepack[i] = b.CreateBitCast(bytepack[i], vec16x16Type);
                signMask[i] = b.CreateICmpSLT(bytepack[i], zeroVec);
            }
            
            for (unsigned i = 0; i < inputPacksPerStride; ++i)
            {
                Value *mulResults = b.CreateCall(smulWithOverflow, {bytepack[i], factorVec});
                amplifiedBytepack[i] = b.CreateExtractValue(mulResults, 0);
                Value *overflow = b.CreateExtractValue(mulResults, 1);

                Value *maxClip = b.CreateAnd(b.CreateNot(signMask[i]), overflow);
                clipBytepack[i] = b.CreateSelect(maxClip, maxVal, amplifiedBytepack[i]);

                Value *minClip = b.CreateAnd(signMask[i], overflow);
                clipBytepack[i] = b.CreateSelect(minClip, minVal, clipBytepack[i]);
                b.storeOutputStreamPack("outputStreams", STREAMINDEX, b.getInt32(i), blockOffsetPhi, clipBytepack[i]);
            }
        }

        Value *nextBlk = b.CreateAdd(blockOffsetPhi, b.getSize(1));
        blockOffsetPhi->addIncoming(nextBlk, loop);
        Value *moreToDo = b.CreateICmpNE(nextBlk, numOfBlocks);

        b.CreateCondBr(moreToDo, loop, exit);
        b.SetInsertPoint(exit);
    }

    AmplifyPabloKernel::AmplifyPabloKernel(KernelBuilder &b, const unsigned int bitsPerSample, StreamSet *const inputStreams, const unsigned int &factor, StreamSet *const outputStreams)
        : PabloKernel(b, "AmplifyPabloKernel_" + std::to_string(factor) + "_" + std::to_string(inputStreams->getNumElements()) + "_" + std::to_string(bitsPerSample),
                           {Binding{"inputStreams", inputStreams}},
                           {Binding{"outputStreams", outputStreams}}),
          bitsPerSample(bitsPerSample), numInputStreams(inputStreams->getNumElements()), factor(factor)
    {
        if (inputStreams->getNumElements() != outputStreams->getNumElements())
        {
            throw std::invalid_argument("numInputStreams: " + std::to_string(inputStreams->getNumElements()) + " != numOutputStreams: " + std::to_string(outputStreams->getNumElements()));
        }
    }

    void AmplifyPabloKernel::generatePabloMethod()
    {
        pablo::PabloBuilder pb(getEntryScope());
        BixNumCompiler bnc(pb);
        std::vector<PabloAST *> inputStreams = getInputStreamSet("inputStreams");
        std::vector<PabloAST *> resultStreams = bnc.MulModular(inputStreams, factor);
        Var * result = getOutputStreamVar("outputStreams");
        for (unsigned i = 0; i < bitsPerSample; i++) {
            pb.createAssign(pb.createExtract(result, pb.getInteger(i)), resultStreams[i]);
        }
    }


    ConcatenateKernel::ConcatenateKernel(KernelBuilder &b, StreamSet *const firstInputStreams, StreamSet *const secondInputStreams, StreamSet *const outputStreams)
        : PabloKernel(b, "ConcatenateKernel_" + std::to_string(firstInputStreams->getNumElements()) + "_" + std::to_string(secondInputStreams->getNumElements()),
                           {Binding{"firstInputStreams", firstInputStreams}, Binding{"secondInputStreams", secondInputStreams}},
                           {Binding{"outputStreams", outputStreams}}),
        numFirstInputStreams(firstInputStreams->getNumElements()), numSecondInputStreams(secondInputStreams->getNumElements())
    {
        if (firstInputStreams->getNumElements() + secondInputStreams->getNumElements() != outputStreams->getNumElements())
        {
            throw std::invalid_argument("numOutputStreams(" + std::to_string(firstInputStreams->getNumElements()) + ") != numFirstInputStreams(" + std::to_string(secondInputStreams->getNumElements()) + ") + numSecondInputStreams("+ std::to_string(outputStreams->getNumElements()) + ")");
        }
    }

    void ConcatenateKernel::generatePabloMethod()
    {
        pablo::PabloBuilder pb(getEntryScope());
        BixNumCompiler bnc(pb);
        std::vector<PabloAST *> firstInputStreams = getInputStreamSet("firstInputStreams");
        std::vector<PabloAST *> secondInputStreams = getInputStreamSet("secondInputStreams");
        Var * result = getOutputStreamVar("outputStreams");
        for (unsigned i = 0; i < numFirstInputStreams; ++i) {
            pb.createAssign(pb.createExtract(result, pb.getInteger(i)), firstInputStreams[i]);
        }

        for (unsigned i = 0; i < numSecondInputStreams; ++i) {
            pb.createAssign(pb.createExtract(result, pb.getInteger(i + numFirstInputStreams)), secondInputStreams[i]);
        }
    }
}

#undef NUM_HEADER_BYTES