#ifndef RE_LOCAL_H
#define RE_LOCAL_H

namespace re {

class RE; class CC;

struct RE_Local {
    static CC * getFirstUniqueSymbol(RE * re);

    static bool noInterCCFromFirstNLast(RE *re);

    static RE * getFirstCCAsRE(RE* re);

    static RE * getLastCCAsRE(RE* re);

    static RE * getUniquePrefix(RE * re, int& length);

    static void reAnalyze(RE* re);
};

}

#endif
