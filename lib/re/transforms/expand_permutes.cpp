#include <re/transforms/re_transformer.h>
#include <re/transforms/expand_permutes.h>
#include <re/printer/re_printer.h>
#include <llvm/Support/raw_ostream.h>
#include <re/adt/adt.h>

using namespace llvm;

namespace re {



class ExpandPermutes final : public RE_Transformer {
public:
    ExpandPermutes() : RE_Transformer("ExpandPermutes") {}
    RE * transformPermute(Permute * p) override;
};

RE * ExpandPermutes::transformPermute(Permute * p) {
    unsigned perm_size = p->size();
    std::vector<RE *> alts;
    for (auto perm : *p) {
        alts.push_back(perm);
    }
    std::vector<RE *> elems;
    elems.push_back(makeRep(makeAlt(alts.begin(), alts.end()), 0, perm_size));
    llvm::errs() << "elem0: " << Printer_RE::PrintRE(elems[0]) << "\n";
    for (auto perm : *p) {
        RE * negated = makeDiff(makeAny(), perm);
        RE * r = makeSeq({perm, makeRep(negated, 0, perm_size - 1)});
        llvm::errs() << "r: " << Printer_RE::PrintRE(r) << "\n";
        elems.push_back(makeLookBehindAssertion(r));
    }
    return makeSeq(elems.begin(), elems.end());
}

RE * expandPermutes(RE * re) {
    llvm::errs() << Printer_RE::PrintRE(re);
    return ExpandPermutes().transformRE(re);
}

}
