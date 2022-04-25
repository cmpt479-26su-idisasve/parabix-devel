/*
 *  Copyright (c) 2019 International Characters.
 *  This software is licensed to the public under the Open Software License 3.0.
 *  icgrep is a trademark of International Characters.
 */

#include <re/analysis/re_local.h>

#include <re/adt/adt.h>
#include <re/analysis/nullable.h>
#include <re/analysis/re_analysis.h>
#include <re/analysis/cc_sequence_search.h>
#include <re/transforms/re_transformer.h>
#include <boost/container/flat_map.hpp>
#include <boost/range/adaptor/reversed.hpp>
#include <iostream>
using namespace boost::container;
using namespace llvm;

namespace re {

using FollowMap = flat_map<const CC *, const CC*>;

inline const CC * combine(const CC * a, const CC * b) {
    if (a && b) {
        return makeCC(a, b);
    } else if (b) {
        return b;
    }
    return a;
}

const CC * first(const RE * re) {
    if (const Name * name = dyn_cast<Name>(re)) {
        if (LLVM_LIKELY(name->getDefinition() != nullptr)) {
            return first(name->getDefinition());
        } else {
            UndefinedNameError(name);
        }
    } else if (const CC * cc = dyn_cast<CC>(re)) {
        return cc;
    } else if (isa<Any>(re)) {
        return makeCC(0, UCD::UNICODE_MAX);
    } else if (const Seq * seq = dyn_cast<Seq>(re)) {
        const CC * cc = nullptr;
        for (auto & si : *seq) {
            cc = combine(cc, first(si));
            if (!isNullable(si)) {
                break;
            }
        }
        return cc;
    } else if (const Alt * alt = dyn_cast<Alt>(re)) {
        const CC * cc = nullptr;
        for (auto & ai : *alt) {
            cc = combine(cc, first(ai));
        }
        return cc;
    } else if (const Rep * rep = dyn_cast<Rep>(re)) {
        return first(rep->getRE());
    } else if (const Diff * diff = dyn_cast<Diff>(re)) {
        if (const CC * lh = first(diff->getLH())) {
            if (const CC * rh = first(diff->getRH())) {
                return subtractCC(lh, rh);
            }
        }
    } else if (const Intersect * ix = dyn_cast<Intersect>(re)) {
        if (const CC * lh = first(ix->getLH())) {
            if (const CC * rh = first(ix->getRH())) {
                return intersectCC(lh, rh);
            }
        }
    }
    return nullptr;
}

const CC * final(const RE * re) {
    if (const Name * name = dyn_cast<Name>(re)) {
        if (LLVM_LIKELY(name->getDefinition() != nullptr)) {
            return final(name->getDefinition());
        } else {
            UndefinedNameError(name);
        }
    } else if (const CC * cc = dyn_cast<CC>(re)) {
        return cc;
    } else if (isa<Any>(re)) {
        return makeCC(0, UCD::UNICODE_MAX);
    } else if (const Seq * seq = dyn_cast<Seq>(re)) {
        const CC * cc = nullptr;
        for (auto & si : boost::adaptors::reverse(*seq)) {
            cc = combine(cc, final(si));
            if (!isNullable(si)) {
                break;
            }
        }
        return cc;
    } else if (const Alt * alt = dyn_cast<Alt>(re)) {
        const CC * cc = nullptr;
        for (auto & ai : *alt) {
            cc = combine(cc, final(ai));
        }
        return cc;
    } else if (const Rep * rep = dyn_cast<Rep>(re)) {
        return final(rep->getRE());
    } else if (const Diff * diff = dyn_cast<Diff>(re)) {
        if (const CC * lh = final(diff->getLH())) {
            if (const CC * rh = final(diff->getRH())) {
                return subtractCC(lh, rh);
            }
        }
    } else if (const Intersect * ix = dyn_cast<Intersect>(re)) {
        if (const CC * lh = final(ix->getLH())) {
            if (const CC * rh = final(ix->getRH())) {
                return intersectCC(lh, rh);
            }
        }
    }
    return nullptr;

}

void follow(const RE * re, FollowMap & follows) {
    if (const Name * name = dyn_cast<Name>(re)) {
        if (LLVM_LIKELY(name->getDefinition() != nullptr)) {
            return follow(name->getDefinition(), follows);
        } else {
            UndefinedNameError(name);
        }
    } else if (const Seq * seq = dyn_cast<Seq>(re)) {
        if (!seq->empty()) {
            const RE * const re_first = *(seq->begin());
            const RE * const re_follow = makeSeq(seq->begin() + 1, seq->end());
            auto e1 = final(re_first);
            auto e2 = first(re_follow);
            if (e1 && e2) {
                auto e = follows.find(e1);
                if (e != follows.end()) {
                    e->second = makeCC(e->second, e2);
                } else {
                    follows.emplace(e1, e2);
                }
            }
            follow(re_first, follows);
            follow(re_follow, follows);
        }
    } else if (const Alt * alt = dyn_cast<Alt>(re)) {
        for (auto ai = alt->begin(); ai != alt->end(); ++ai) {
            follow(*ai, follows);
        }
    } else if (const Rep * rep = dyn_cast<Rep>(re)) {
        const auto e1 = final(rep->getRE());
        auto e2 = first(rep->getRE());
        if (e1 && e2) {
            auto e = follows.find(e1);
            if (e != follows.end()) {
                e->second = makeCC(e->second, e2);
            } else {
                follows.emplace(e1, e2);
            }
        }
        follow(rep->getRE(), follows);
    }
}

RE * getReExceptFirst(RE* re)
{
    if(const Seq * seq = dyn_cast<Seq>(re))
    {
        if(seq->size() > 1)
        {
            return makeSeq(seq->begin()+1, seq->end());
        }
    }

    return nullptr;
}

// Used for debugging
void analyze(RE* re)
{
    if(const Seq * seq = dyn_cast<Seq>(re))
    {
        for(int i=0; i < seq->size(); ++i)
        {
            std::cout<<"i = " << i ;
            RE * item = (*seq)[i];
            if (const CC * cc = dyn_cast<CC>(item))
            {
                std::cout<<"  CC " << std::endl ;
            }else if(const Alt * alt = dyn_cast<Alt>(item))
            {
                std::cout<<"  Alt " << std::endl ;
                int j=0;
                for (const RE * res : *alt) {
                    std::cout<<"Alt " <<  j++  ;
                    if(const Start *start = dyn_cast<Start>(res))
                    {
                        std::cout<<"  Start "  << std::endl ;
                    }else if(const CC * cc = dyn_cast<CC>(res))
                    {
                        std::cout<<"  CC " << std::endl ;
                    }else if(const Assertion * as = dyn_cast<Assertion>(res))
                    {
                        std::cout<<"  Assertion BOOM "  << std::endl ;
                        auto asRe = as->getAsserted();
                        if(const CC * cc = dyn_cast<CC>(asRe))
                            std::cout << "Assertion CC: " << cc->canonicalName() << std::endl;
                        
                    }
                    else{
                        std::cout<< std::endl;
                    }
                }
            }else if(const Name * name = dyn_cast<Name>(item))
            {
                std::cout<<"  Name " << std::endl ;
            }else if(const Rep * name = dyn_cast<Rep>(item))
            {
                std::cout<<"  Rep " << std::endl ;
            }else{
                std::cout<< std::endl;
            }
        }
    }
}

// E = APQ: A(single CC)
// AP not occurrs in PQ 
RE * RE_Local::getUniquePrefix(RE *re, int &length)
{
    RE * re_except_first = getReExceptFirst(re);
    if(re_except_first == nullptr)  return nullptr;
    RE * prefix_AP = nullptr;  
    length = 0;
    if(const Seq * seq = dyn_cast<Seq>(re))
    {
        std::vector<CC *> CC_seq;
        int endPoint = -1;
        bool isUniqueStart = false;
        for(int i=0; i < seq->size()-1; ++i)
        {
            RE * item = (*seq)[i];
            if (CC * cc = dyn_cast<CC>(item))
            {
                CC_seq.push_back(cc);
                bool search = CC_Sequence_Search(CC_seq, re_except_first);
                if(search) {
                    if(isUniqueStart) break;
                    else continue;
                    }
                else {
                    endPoint = i;
                    isUniqueStart = true;
                }

            }else{
                break;
            }
        }
        // Only extract if there consists unique prefix
        if(endPoint != -1 && isUniqueStart)
        {
            prefix_AP = makeSeq(seq->begin(), seq->begin()+endPoint+1);
            // Calculate the length of Fixed Prefix in UTF8 
            length = getLengthRange(prefix_AP,&cc::UTF8 ).first;
        }        
    }
    
    return prefix_AP;
}


void RE_Local::reAnalyze(RE * re)
{
    return analyze(re);
}

CC * RE_Local::getFirstUniqueSymbol(RE * const re) {
    const CC * const re_first = first(re);
    if (re_first) {
        FollowMap follows;
        follow(re, follows);
        for (const auto & entry : follows) {
            if (entry.second->intersects(*re_first)) {
                return nullptr;
            }
        }
    }
    return const_cast<CC *>(re_first);
}

bool RE_Local::noInterCCFromFirstNLast(RE *re)
{
    const CC * const re_first  = first(re);
    const CC * const re_final  = final(re);
    if(re_first)
    {
        FollowMap follows;
        follow(re, follows);
        if(follows.size() == 0) return false;
        for (const auto & entry : follows) {
            if (entry.second->intersects(*re_first)) {
                return false;
            }
            if (entry.first->intersects(*re_final))
            {
                return false;
            }
        }
    }
    return true;
}

RE * RE_Local::getFirstCCAsRE(RE* re)
{
    if (const Seq * seq = dyn_cast<Seq>(re)) { 
        return makeSeq(seq->begin(), seq->begin()+1);
    }else if (const Alt * alt = dyn_cast<Alt>(re)){
        vector<RE*> firstReVec;
        std::vector<RE*> vec;
        for(const RE* altRE : * alt)
        {
            if(const Seq * seq = dyn_cast<Seq>(altRE))
            {
                auto fir = makeSeq(seq->begin(), seq->begin()+1);
                vec.push_back(fir);
            }
        }
        return makeAlt(vec.begin(), vec.end());
    }
    return nullptr;
}

RE * RE_Local::getLastCCAsRE(RE* re)
{
    if (const Seq * seq = dyn_cast<Seq>(re)) { 
        return makeSeq(seq->end()-1, seq->end());

    }else if (const Alt * alt = dyn_cast<Alt>(re)){
        vector<RE*> firstReVec;
        std::vector<RE*> vec;
        for(const RE* altRE : * alt)
        {
            if(const Seq * seq = dyn_cast<Seq>(altRE))
            {
                auto fir = makeSeq(seq->end()-1, seq->end());
                vec.push_back(fir);
            }
        }

        return makeAlt(vec.begin(), vec.end());
    }
    return nullptr;
}



}
