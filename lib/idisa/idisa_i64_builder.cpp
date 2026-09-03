/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 */

#include <idisa/idisa_i64_builder.h>
#include <sstream>

using namespace llvm;

namespace IDISA {

DEFINE_BUILDER_CACHE_NAME(IDISA_I64_Builder, "C", I64_width)

} // namespace IDISA
