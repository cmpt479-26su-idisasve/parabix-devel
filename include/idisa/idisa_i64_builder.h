#pragma once

/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 */
#include <idisa/idisa_generic_builder.h>

namespace IDISA {

constexpr unsigned I64_width = 64;

class IDISA_I64_Builder : public IDISA_Generic_Builder {
  public:
    explicit IDISA_I64_Builder(CBuilder *cb, unsigned vectorWidth, unsigned laneWidth,
                               unsigned overrideNativeVectorWidth = I64_width)
        : IDISA_Generic_Builder(cb, vectorWidth, laneWidth, overrideNativeVectorWidth) {}

    std::string getBuilderCacheName() override;
};

} // namespace IDISA
