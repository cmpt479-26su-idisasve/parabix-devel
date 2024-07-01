#include "audio/stream_manipulation.h"


namespace audio
{
    void CreateOnes::generatePabloMethod()
    {
        pablo::PabloBuilder pb(getEntryScope());
        PabloAST *inputStream = getInputStreamSet("dataStream")[0];


        PabloAST *ones = pb.createOr(pb.createOnes(), inputStream);
        
        Var *onesVar = getOutputStreamVar("onesStream");
        pb.createAssign(pb.createExtract(onesVar, pb.getInteger(0)), ones);
    }

    mS2PKernel::mS2PKernel(KernelBuilder &b, StreamSet *const inputStreams, StreamSet *const outputStreams, const unsigned int bitsPerSample)
        : MultiBlockKernel(b, "mS2PKernel_" + std::to_string(inputStreams->getNumElements()) + "_" + std::to_string(bitsPerSample),
                           {Binding{"inputStreams", inputStreams, FixedRate(2)}},
                           {Binding{"outputStreams", outputStreams, FixedRate(1)}}, {}, {}, {}),
          numInputStreams(inputStreams->getNumElements()), bitsPerSample(bitsPerSample) {}

    void mS2PKernel::generateMultiBlockLogic(KernelBuilder &b, Value *const numOfStrides)
    {
        const unsigned fw = 8;
        const unsigned inputPacksPerStride = fw * 2;
        const unsigned outputPacksPerStride = fw * 1;

        BasicBlock *entry = b.GetInsertBlock();
        BasicBlock *packLoop = b.CreateBasicBlock("packLoop");
        BasicBlock *packFinalize = b.CreateBasicBlock("packFinalize");
        Constant *const ZERO = b.getSize(0);
        Value *numOfBlocks = numOfStrides;
        b.CreateBr(packLoop);
        b.SetInsertPoint(packLoop);
        PHINode *blockOffsetPhi = b.CreatePHI(b.getSizeTy(), 2);
        blockOffsetPhi->addIncoming(ZERO, entry);
        for (int streamIndex = 0; streamIndex < numInputStreams; ++streamIndex)
        {
            Value *bytepack[inputPacksPerStride];
            Constant *const STREAMINDEX = b.getSize(streamIndex);
            Constant *const LOWSTREAMINDEX = b.getSize(2 * streamIndex);
            Constant *const HIGHSTREAMINDEX = b.getSize(2 * streamIndex + 1);
            for (unsigned i = 0; i < inputPacksPerStride; i++)
            {
                bytepack[i] = b.loadInputStreamPack("inputStreams", STREAMINDEX, b.getInt32(i), blockOffsetPhi);
            }

            Value *lo[outputPacksPerStride];
            Value *hi[outputPacksPerStride];
            for (unsigned i = 0; i < outputPacksPerStride; i++)
            {
                lo[i] = b.hsimd_packl(2 * bitsPerSample, bytepack[2 * i], bytepack[2 * i + 1]);
                hi[i] = b.hsimd_packh(2 * bitsPerSample, bytepack[2 * i], bytepack[2 * i + 1]);
                b.storeOutputStreamPack("outputStreams", LOWSTREAMINDEX, b.getInt32(i), blockOffsetPhi, lo[i]);
                b.storeOutputStreamPack("outputStreams", HIGHSTREAMINDEX, b.getInt32(i), blockOffsetPhi, hi[i]);
            }
        }

        Value *nextBlk = b.CreateAdd(blockOffsetPhi, b.getSize(1));
        blockOffsetPhi->addIncoming(nextBlk, packLoop);
        Value *moreToDo = b.CreateICmpNE(nextBlk, numOfBlocks);

        b.CreateCondBr(moreToDo, packLoop, packFinalize);
        b.SetInsertPoint(packFinalize);
    }
}
