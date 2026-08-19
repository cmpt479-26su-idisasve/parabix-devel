#pragma once

#include <idisa/CBuilder.h>
#include <idisa/idisa_builder.h>
#include <kernel/core/kernel.h>

namespace kernel {

class KernelBuilder;

class Proxy_IDISA_Builder : public IDISA::IDISA_Builder {
    friend class KernelBuilder;

  public:
    void CreateBaseFunctions() override final { mTgt->CreateBaseFunctions(); }
    std::string getBuilderCacheName() override final { return mTgt->getBuilderCacheName(); }

  protected:
    llvm::Value *simd_fill_impl(unsigned fw, llvm::Value *a) override final { return mTgt->simd_fill(fw, a); }
    llvm::Value *simd_fill_impl(unsigned vector_width, unsigned fw, llvm::Value *a) override final {
        return mTgt->simd_fill(vector_width, fw, a);
    }
    llvm::Value *simd_add_impl(unsigned fw, llvm::Value *a, llvm::Value *b) override final {
        return mTgt->simd_add(fw, a, b);
    }
    llvm::Value *simd_sub_impl(unsigned fw, llvm::Value *a, llvm::Value *b) override final {
        return mTgt->simd_sub(fw, a, b);
    }
    llvm::Value *simd_mult_impl(unsigned fw, llvm::Value *a, llvm::Value *b) override final {
        return mTgt->simd_mult(fw, a, b);
    }
    llvm::Value *simd_eq_impl(unsigned fw, llvm::Value *a, llvm::Value *b) override final {
        return mTgt->simd_eq(fw, a, b);
    }
    llvm::Value *simd_ne_impl(unsigned fw, llvm::Value *a, llvm::Value *b) override final {
        return mTgt->simd_ne(fw, a, b);
    }
    llvm::Value *simd_gt_impl(unsigned fw, llvm::Value *a, llvm::Value *b) override final {
        return mTgt->simd_gt(fw, a, b);
    }
    llvm::Value *simd_ge_impl(unsigned fw, llvm::Value *a, llvm::Value *b) override final {
        return mTgt->simd_ge(fw, a, b);
    }
    llvm::Value *simd_lt_impl(unsigned fw, llvm::Value *a, llvm::Value *b) override final {
        return mTgt->simd_lt(fw, a, b);
    }
    llvm::Value *simd_le_impl(unsigned fw, llvm::Value *a, llvm::Value *b) override final {
        return mTgt->simd_le(fw, a, b);
    }
    llvm::Value *simd_ugt_impl(unsigned fw, llvm::Value *a, llvm::Value *b) override final {
        return mTgt->simd_ugt(fw, a, b);
    }
    llvm::Value *simd_ult_impl(unsigned fw, llvm::Value *a, llvm::Value *b) override final {
        return mTgt->simd_ult(fw, a, b);
    }
    llvm::Value *simd_ule_impl(unsigned fw, llvm::Value *a, llvm::Value *b) override final {
        return mTgt->simd_ule(fw, a, b);
    }
    llvm::Value *simd_uge_impl(unsigned fw, llvm::Value *a, llvm::Value *b) override final {
        return mTgt->simd_uge(fw, a, b);
    }
    llvm::Value *simd_max_impl(unsigned fw, llvm::Value *a, llvm::Value *b) override final {
        return mTgt->simd_max(fw, a, b);
    }
    llvm::Value *simd_umax_impl(unsigned fw, llvm::Value *a, llvm::Value *b) override final {
        return mTgt->simd_umax(fw, a, b);
    }
    llvm::Value *simd_min_impl(unsigned fw, llvm::Value *a, llvm::Value *b) override final {
        return mTgt->simd_min(fw, a, b);
    }
    llvm::Value *simd_umin_impl(unsigned fw, llvm::Value *a, llvm::Value *b) override final {
        return mTgt->simd_umin(fw, a, b);
    }
    llvm::Value *simd_if_impl(unsigned fw, llvm::Value *cond, llvm::Value *a, llvm::Value *b) override final {
        return mTgt->simd_if(fw, cond, a, b);
    }
    virtual llvm::Value *simd_ternary_impl(unsigned char mask, llvm::Value *bit_2, llvm::Value *bit_1,
                                           llvm::Value *bit_0) {
        return mTgt->simd_ternary(mask, bit_2, bit_1, bit_0);
    }
    llvm::Value *simd_slli_impl(unsigned fw, llvm::Value *a, unsigned shift) override final {
        return mTgt->simd_slli(fw, a, shift);
    }
    llvm::Value *simd_srli_impl(unsigned fw, llvm::Value *a, unsigned shift) override final {
        return mTgt->simd_srli(fw, a, shift);
    }
    llvm::Value *simd_srai_impl(unsigned fw, llvm::Value *a, unsigned shift) override final {
        return mTgt->simd_srai(fw, a, shift);
    }
    llvm::Value *simd_sllv_impl(unsigned fw, llvm::Value *a, llvm::Value *shifts) override final {
        return mTgt->simd_sllv(fw, a, shifts);
    }
    llvm::Value *simd_srlv_impl(unsigned fw, llvm::Value *a, llvm::Value *shifts) override final {
        return mTgt->simd_srlv(fw, a, shifts);
    }
    llvm::Value *simd_rotl_impl(unsigned fw, llvm::Value *a, llvm::Value *rotates) override final {
        return mTgt->simd_rotl(fw, a, rotates);
    }
    llvm::Value *simd_rotr_impl(unsigned fw, llvm::Value *a, llvm::Value *rotates) override final {
        return mTgt->simd_rotr(fw, a, rotates);
    }
    virtual std::vector<llvm::Value *> simd_pext_impl(unsigned fw, std::vector<llvm::Value *> vs,
                                                      llvm::Value *extract_mask) {
        return mTgt->simd_pext(fw, vs, extract_mask);
    }
    llvm::Value *simd_pdep_impl(unsigned fw, llvm::Value *v, llvm::Value *deposit_mask) override final {
        return mTgt->simd_pdep(fw, v, deposit_mask);
    }
    llvm::Value *simd_any_impl(unsigned fw, llvm::Value *a) override final { return mTgt->simd_any(fw, a); }
    llvm::Value *simd_popcount_impl(unsigned fw, llvm::Value *a) override final { return mTgt->simd_popcount(fw, a); }
    llvm::Value *hsimd_partial_sum_impl(unsigned fw, llvm::Value *a) override final {
        return mTgt->hsimd_partial_sum(fw, a);
    }
    llvm::Value *simd_cttz_impl(unsigned fw, llvm::Value *a) override final { return mTgt->simd_cttz(fw, a); }
    llvm::Value *simd_bitreverse_impl(unsigned fw, llvm::Value *a) override final {
        return mTgt->simd_bitreverse(fw, a);
    }
    llvm::Value *esimd_mergeh_impl(unsigned fw, llvm::Value *a, llvm::Value *b) override final {
        return mTgt->esimd_mergeh(fw, a, b);
    }
    llvm::Value *esimd_mergel_impl(unsigned fw, llvm::Value *a, llvm::Value *b) override final {
        return mTgt->esimd_mergel(fw, a, b);
    }
    llvm::Value *esimd_bitspread_impl(unsigned vec_width, unsigned fw, llvm::Value *bitmask) override final {
        return mTgt->esimd_bitspread(vec_width, fw, bitmask);
    }
    llvm::Value *hsimd_packh_impl(unsigned fw, llvm::Value *a, llvm::Value *b) override final {
        return mTgt->hsimd_packh(fw, a, b);
    }
    llvm::Value *hsimd_packl_impl(unsigned fw, llvm::Value *a, llvm::Value *b) override final {
        return mTgt->hsimd_packl(fw, a, b);
    }
    llvm::Value *hsimd_packss_impl(unsigned fw, llvm::Value *a, llvm::Value *b) override final {
        return mTgt->hsimd_packss(fw, a, b);
    }
    llvm::Value *hsimd_packus_impl(unsigned fw, llvm::Value *a, llvm::Value *b) override final {
        return mTgt->hsimd_packus(fw, a, b);
    }
    llvm::Value *hsimd_packh_in_lanes_impl(unsigned lanes, unsigned fw, llvm::Value *a, llvm::Value *b) override final {
        return mTgt->hsimd_packh_in_lanes(lanes, fw, a, b);
    }
    llvm::Value *hsimd_packl_in_lanes_impl(unsigned lanes, unsigned fw, llvm::Value *a, llvm::Value *b) override final {
        return mTgt->hsimd_packl_in_lanes(lanes, fw, a, b);
    }
    llvm::Value *hsimd_signmask_impl(unsigned fw, llvm::Value *a) override final { return mTgt->hsimd_signmask(fw, a); }
    llvm::Value *mvmd_extract_impl(unsigned fw, llvm::Value *a, unsigned fieldIndex) override final {
        return mTgt->mvmd_extract(fw, a, fieldIndex);
    }
    llvm::Value *mvmd_insert_impl(unsigned fw, llvm::Value *blk, llvm::Value *elt, unsigned fieldIndex) override final {
        return mTgt->mvmd_insert(fw, blk, elt, fieldIndex);
    }
    llvm::Value *mvmd_sll_impl(unsigned fw, llvm::Value *value, llvm::Value *shift, const bool safe) override final {
        return mTgt->mvmd_sll(fw, value, shift, safe);
    }
    llvm::Value *mvmd_srl_impl(unsigned fw, llvm::Value *value, llvm::Value *shift, const bool safe) override final {
        return mTgt->mvmd_srl(fw, value, shift, safe);
    }
    llvm::Value *mvmd_slli_impl(unsigned fw, llvm::Value *a, unsigned shift) override final {
        return mTgt->mvmd_slli(fw, a, shift);
    }
    llvm::Value *mvmd_srli_impl(unsigned fw, llvm::Value *a, unsigned shift) override final {
        return mTgt->mvmd_srli(fw, a, shift);
    }
    llvm::Value *mvmd_dslli_impl(unsigned fw, llvm::Value *a, llvm::Value *b, unsigned shift) override final {
        return mTgt->mvmd_dslli(fw, a, b, shift);
    }
    llvm::Value *mvmd_dsll_impl(unsigned fw, llvm::Value *a, llvm::Value *b, llvm::Value *shift) override final {
        return mTgt->mvmd_dsll(fw, a, b, shift);
    }
    virtual llvm::Value *mvmd_shuffle_impl(unsigned fw, llvm::Value *data_table, llvm::Value *index_vector,
                                           IDISA::ShuffleMode m) {
        return mTgt->mvmd_shuffle(fw, data_table, index_vector, m);
    }
    virtual llvm::Value *mvmd_shuffle2_impl(unsigned fw, llvm::Value *table0, llvm::Value *table1,
                                            llvm::Value *index_vector, IDISA::ShuffleMode m) {
        return mTgt->mvmd_shuffle2(fw, table0, table1, index_vector, m);
    }
    llvm::Value *mvmd_compress_impl(unsigned fw, llvm::Value *a, llvm::Value *select_mask) override final {
        return mTgt->mvmd_compress(fw, a, select_mask);
    }
    llvm::Value *mvmd_expand_impl(unsigned fw, llvm::Value *a, llvm::Value *select_mask) override final {
        return mTgt->mvmd_expand(fw, a, select_mask);
    }
    llvm::Value *bitblock_any_impl(llvm::Value *a) override final { return mTgt->bitblock_any(a); }
    std::pair<llvm::Value *, llvm::Value *> bitblock_add_with_carry_impl(llvm::Value *a, llvm::Value *b,
                                                                         llvm::Value *carryin) override final {
        return mTgt->bitblock_add_with_carry(a, b, carryin);
    }
    std::pair<llvm::Value *, llvm::Value *> bitblock_subtract_with_borrow_impl(llvm::Value *a, llvm::Value *b,
                                                                               llvm::Value *borrowin) override final {
        return mTgt->bitblock_subtract_with_borrow(a, b, borrowin);
    }
    std::pair<llvm::Value *, llvm::Value *> bitblock_advance_impl(llvm::Value *a, llvm::Value *shiftin,
                                                                  unsigned shift) override final {
        return mTgt->bitblock_advance(a, shiftin, shift);
    }
    std::pair<llvm::Value *, llvm::Value *> bitblock_indexed_advance_impl(llvm::Value *a, llvm::Value *index_strm,
                                                                          llvm::Value *shiftin,
                                                                          unsigned shift) override final {
        return mTgt->bitblock_indexed_advance(a, index_strm, shiftin, shift);
    }
    llvm::Value *bitblock_mask_from_impl(llvm::Value *const position, const bool safe) override final {
        return mTgt->bitblock_mask_from(position, safe);
    }
    llvm::Value *bitblock_mask_to_impl(llvm::Value *const position, const bool safe) override final {
        return mTgt->bitblock_mask_to(position, safe);
    }
    llvm::Value *bitblock_set_bit_impl(llvm::Value *const position, const bool safe) override final {
        return mTgt->bitblock_set_bit(position, safe);
    }

  protected:
    IDISA::IDISA_Builder *mTgt;

    Proxy_IDISA_Builder(CBuilder *cb) : IDISA_Builder(cb), mTgt() {}

    void setTarget(IDISA::IDISA_Builder *tgt);
};

class KernelBuilder : public CBuilder, public Proxy_IDISA_Builder {
    friend class Kernel;
    friend class KernelCompiler;

  public:
    using Rational = ProcessingRate::Rational;

    using ScalarRef = std::pair<llvm::Value *, llvm::Type *>;

    using IDISA::IDISA_Builder::getContext;

    enum TerminationCode : unsigned { None = 0, Terminated = 1, Fatal = 2 };

    llvm::Value *getHandle() const noexcept;

    llvm::Value *getThreadLocalHandle() const noexcept;

    bool hasScalarField(const llvm::StringRef name) const;

    ScalarRef getScalarFieldPtr(const llvm::StringRef fieldName);

    llvm::Value *getScalarField(const llvm::StringRef fieldName);

    llvm::LoadInst *CreateMonitoredScalarFieldLoad(const llvm::StringRef fieldName, llvm::Value *ptr);

    llvm::StoreInst *CreateMonitoredScalarFieldStore(const llvm::StringRef fieldName, llvm::Value *toStore,
                                                     llvm::Value *ptr);

    // Set the value of a scalar field for the current instance.
    void setScalarField(const llvm::StringRef fieldName, llvm::Value *value);

    llvm::Value *getAvailableItemCount(const llvm::StringRef name) const noexcept;

    llvm::Value *getAccessibleItemCount(const llvm::StringRef name) const noexcept;

    llvm::Value *getProcessedItemCount(const llvm::StringRef name);

    void setProcessedItemCount(const llvm::StringRef name, llvm::Value *value);

    llvm::Value *getProducedItemCount(const llvm::StringRef name);

    void setProducedItemCount(const llvm::StringRef name, llvm::Value *value);

    llvm::Value *getWritableOutputItems(const llvm::StringRef name) const noexcept;

    llvm::Value *getConsumedItemCount(const llvm::StringRef name) const noexcept;

    llvm::Value *getTerminationSignal();

    void setTerminationSignal() { setTerminationSignal(getSize(TerminationCode::Terminated)); }

    void setFatalTerminationSignal() { setTerminationSignal(getSize(TerminationCode::Fatal)); }

    void setTerminationSignal(llvm::Value *const value);

    // Run-time access of Kernel State and parameters of methods for
    // use in implementing kernels.

    llvm::Value *getInputStreamBlockPtr(const llvm::StringRef name, llvm::Value *streamIndex) {
        return getInputStreamBlockPtr(name, streamIndex, nullptr);
    }

    llvm::Value *getInputStreamBlockPtr(const llvm::StringRef name, llvm::Value *streamIndex, llvm::Value *blockOffset);

    llvm::Value *getInputStreamLogicalBasePtr(const Binding &input);

    llvm::Value *loadInputStreamBlock(const llvm::StringRef name, llvm::Value *streamIndex) {
        return loadInputStreamBlock(name, streamIndex, nullptr);
    }

    llvm::Value *loadInputStreamBlock(const llvm::StringRef name, llvm::Value *streamIndex, llvm::Value *blockOffset);

    llvm::Value *getInputStreamPackPtr(const llvm::StringRef name, llvm::Value *streamIndex, llvm::Value *packIndex) {
        return getInputStreamPackPtr(name, streamIndex, packIndex, nullptr);
    }

    llvm::Value *getInputStreamPackPtr(const llvm::StringRef name, llvm::Value *streamIndex, llvm::Value *packIndex,
                                       llvm::Value *blockOffset);

    llvm::Value *loadInputStreamPack(const llvm::StringRef name, llvm::Value *streamIndex, llvm::Value *packIndex) {
        return loadInputStreamPack(name, streamIndex, packIndex, nullptr);
    }

    llvm::Value *loadInputStreamPack(const llvm::StringRef name, llvm::Value *streamIndex, llvm::Value *packIndex,
                                     llvm::Value *blockOffset);

    llvm::Value *getInputStreamSetCount(const llvm::StringRef name);

    llvm::Value *getOutputStreamBlockPtr(const llvm::StringRef name, llvm::Value *streamIndex) {
        return getOutputStreamBlockPtr(name, streamIndex, nullptr);
    }

    llvm::Value *getOutputStreamBlockPtr(const llvm::StringRef name, llvm::Value *streamIndex,
                                         llvm::Value *blockOffset);

    llvm::Value *getOutputStreamLogicalBasePtr(const Binding &output);

    llvm::StoreInst *storeOutputStreamBlock(const llvm::StringRef name, llvm::Value *streamIndex,
                                            llvm::Value *toStore) {
        return storeOutputStreamBlock(name, streamIndex, nullptr, toStore);
    }

    llvm::StoreInst *storeOutputStreamBlock(const llvm::StringRef name, llvm::Value *streamIndex,
                                            llvm::Value *blockOffset, llvm::Value *toStore);

    llvm::Value *getOutputStreamPackPtr(const llvm::StringRef name, llvm::Value *streamIndex, llvm::Value *packIndex) {
        return getOutputStreamPackPtr(name, streamIndex, packIndex, nullptr);
    }

    llvm::Value *getOutputStreamPackPtr(const llvm::StringRef name, llvm::Value *streamIndex, llvm::Value *packIndex,
                                        llvm::Value *blockOffset);

    llvm::StoreInst *storeOutputStreamPack(const llvm::StringRef name, llvm::Value *streamIndex, llvm::Value *packIndex,
                                           llvm::Value *toStore) {
        return storeOutputStreamPack(name, streamIndex, packIndex, nullptr, toStore);
    }

    llvm::StoreInst *storeOutputStreamPack(const llvm::StringRef name, llvm::Value *streamIndex, llvm::Value *packIndex,
                                           llvm::Value *blockOffset, llvm::Value *toStore);

    llvm::Value *getOutputStreamSetCount(const llvm::StringRef name);

    llvm::Value *getRawInputPointer(const llvm::StringRef name, llvm::Value *absolutePosition);

    llvm::Value *getRawOutputPointer(const llvm::StringRef name, llvm::Value *absolutePosition);

    llvm::Value *getRawInputPointer(const llvm::StringRef name, llvm::Value *const streamIndex,
                                    llvm::Value *absolutePosition);

    llvm::Value *getRawOutputPointer(const llvm::StringRef name, llvm::Value *const streamIndex,
                                     llvm::Value *absolutePosition);

    llvm::Value *readRawInputPointer(llvm::Type *ty, const llvm::StringRef name, llvm::Value *absolutePosition);

    llvm::Value *writeRawOutputPointer(const llvm::StringRef name, llvm::Value *absolutePosition, llvm::Value *value);

    llvm::Value *readRawInputPointer(llvm::Type *ty, const llvm::StringRef name, llvm::Value *const streamIndex,
                                     llvm::Value *absolutePosition);

    llvm::Value *writeRawOutputPointer(const llvm::StringRef name, llvm::Value *const streamIndex,
                                       llvm::Value *absolutePosition, llvm::Value *value);

    llvm::Value *getBaseAddress(const llvm::StringRef name);

    void setBaseAddress(const llvm::StringRef name, llvm::Value *addr);

    llvm::Value *getCapacity(const llvm::StringRef name);

    void setCapacity(const llvm::StringRef name, llvm::Value *capacity);

    void reserveCapacity(const llvm::StringRef name, llvm::Value *capacity);

    // internal state

    llvm::Value *getNumOfStrides() const noexcept;

    llvm::Value *getExternalSegNo() const noexcept;

    llvm::Value *isFinal() const noexcept;

    // input streamset bindings

    const Bindings &getInputStreamSetBindings() const noexcept;

    const Binding &getInputStreamSetBinding(const unsigned i) const noexcept;

    const Binding &getInputStreamSetBinding(const llvm::StringRef name) const noexcept;

    StreamSet *getInputStreamSet(const unsigned i) const noexcept;

    StreamSet *getInputStreamSet(const llvm::StringRef name) const noexcept;

    void setInputStreamSet(const llvm::StringRef name, StreamSet *value) noexcept;

    unsigned getNumOfStreamInputs() const noexcept;

    // input streamsets

    StreamSetBuffer *getInputStreamSetBuffer(const unsigned i) const noexcept;

    StreamSetBuffer *getInputStreamSetBuffer(const llvm::StringRef name) const noexcept;

    // output streamset bindings

    const Bindings &getOutputStreamSetBindings() const noexcept;

    const Binding &getOutputStreamSetBinding(const unsigned i) const noexcept;

    const Binding &getOutputStreamSetBinding(const llvm::StringRef name) const noexcept;

    StreamSet *getOutputStreamSet(const unsigned i) const noexcept;

    StreamSet *getOutputStreamSet(const llvm::StringRef name) const noexcept;

    void setOutputStreamSet(const llvm::StringRef name, StreamSet *value) noexcept;

    unsigned getNumOfStreamOutputs() const noexcept;

    // output streamsets

    StreamSetBuffer *getOutputStreamSetBuffer(const unsigned i) const noexcept;

    StreamSetBuffer *getOutputStreamSetBuffer(const llvm::StringRef name) const noexcept;

    // input scalar bindings

    const Bindings &getInputScalarBindings() const noexcept;

    const Binding &getInputScalarBinding(const unsigned i) const noexcept;

    const Binding &getInputScalarBinding(const llvm::StringRef name) const noexcept;

    unsigned getNumOfScalarInputs() const noexcept;

    // input scalars

    Scalar *getInputScalar(const unsigned i) noexcept;

    Scalar *getInputScalar(const llvm::StringRef name) noexcept;

    // output scalar bindings

    const Bindings &getOutputScalarBindings() const noexcept;

    const Binding &getOutputScalarBinding(const unsigned i) const noexcept;

    const Binding &getOutputScalarBinding(const llvm::StringRef name) const noexcept;

    unsigned getNumOfScalarOutputs() const noexcept;

    // output scalars

    Scalar *getOutputScalar(const unsigned i) noexcept;

    Scalar *getOutputScalar(const llvm::StringRef name) noexcept;

    // rational math functions

    llvm::Value *CreateCeilAddRational(llvm::Value *const number, const Rational divisor, const llvm::Twine &Name = "");

    llvm::Value *CreateUDivRational(llvm::Value *const number, const Rational divisor, const llvm::Twine &Name = "");

    llvm::Value *CreateCeilUDivRational(llvm::Value *const number, const Rational divisor,
                                        const llvm::Twine &Name = "");

    llvm::Value *CreateMulRational(llvm::Value *const number, const Rational factor, const llvm::Twine &Name = "");

    llvm::Value *CreateCeilUMulRational(llvm::Value *const number, const Rational factor, const llvm::Twine &Name = "");

    llvm::Value *CreateURemRational(llvm::Value *const number, const Rational factor, const llvm::Twine &Name = "");

    llvm::Value *CreateRoundDownRational(llvm::Value *const number, const Rational factor,
                                         const llvm::Twine &Name = "");

    llvm::Value *CreateRoundUpRational(llvm::Value *const number, const Rational factor, const llvm::Twine &Name = "");

    std::string getKernelName() const noexcept final;

    enum class MemoryOrdering : uint8_t { ColumnMajor, RowMajor };

    void captureByteData(llvm::StringRef streamName, llvm::Value *byteData, llvm::Value *from = nullptr,
                         llvm::Value *to = nullptr, const MemoryOrdering ordering = MemoryOrdering::RowMajor,
                         const char nonASCIIsubstitute = '.') {
        return captureByteData(streamName, byteData->getType(), byteData, from, to, ordering, nonASCIIsubstitute);
    }

    void captureByteData(llvm::StringRef streamName, llvm::Type *type, llvm::Value *byteData,
                         llvm::Value *from = nullptr, llvm::Value *to = nullptr,
                         const MemoryOrdering ordering = MemoryOrdering::RowMajor, const char nonASCIIsubstitute = '.');

    void captureBitstream(llvm::StringRef streamName, llvm::Value *bitstream, llvm::Value *from = nullptr,
                          llvm::Value *to = nullptr, const MemoryOrdering ordering = MemoryOrdering::RowMajor,
                          const char zeroCh = '.', const char oneCh = '1') {
        return captureBitstream(streamName, bitstream->getType(), bitstream, from, to, ordering, zeroCh, oneCh);
    }

    void captureBitstream(llvm::StringRef streamName, llvm::Type *type, llvm::Value *bitstream,
                          llvm::Value *from = nullptr, llvm::Value *to = nullptr,
                          const MemoryOrdering ordering = MemoryOrdering::RowMajor, const char zeroCh = '.',
                          const char oneCh = '1');

    void captureBixNum(llvm::StringRef streamName, llvm::Value *bixnum, llvm::Value *from = nullptr,
                       llvm::Value *to = nullptr, const MemoryOrdering ordering = MemoryOrdering::RowMajor,
                       const char hexBase = 'A') {
        captureBixNum(streamName, bixnum->getType(), bixnum, from, to, ordering, hexBase);
    }

    void captureBixNum(llvm::StringRef streamName, llvm::Type *type, llvm::Value *bixnum, llvm::Value *from = nullptr,
                       llvm::Value *to = nullptr, const MemoryOrdering ordering = MemoryOrdering::RowMajor,
                       const char hexBase = 'A');

    KernelCompiler *getCompiler() const noexcept { return mCompiler; }

  private:
    struct AddressableValue {
        llvm::Value *Address;
        llvm::Value *From;
        llvm::Value *To;
    };

    AddressableValue makeAddressableValue(llvm::Type *type, llvm::Value *value, llvm::Value *from, llvm::Value *to,
                                          const MemoryOrdering ordering);

  protected:
    KernelBuilder(llvm::LLVMContext &C, const codegen::FeatureSet &featureSet);

    void setCompiler(KernelCompiler *const compiler) noexcept { mCompiler = compiler; }

  protected:
    KernelCompiler *mCompiler = nullptr;
};

template <class SpecializedBuilder> class KernelBuilderImpl final : public KernelBuilder {
  public:
    KernelBuilderImpl(llvm::LLVMContext &C, const codegen::FeatureSet &featureSet, unsigned vectorWidth,
                      unsigned laneWidth)
        : KernelBuilder(C, featureSet), mSpecialization(this, vectorWidth, laneWidth) {
        setTarget(&mSpecialization);
    }

  private:
    SpecializedBuilder mSpecialization;
};

} // namespace kernel
