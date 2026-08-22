#pragma once

#include <limits.h>
#include <stdint.h>

#include <fcntl.h>
#include <unistd.h>

#include <chrono>
#include <format>
#include <functional>
#include <memory>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

#include <llvm/ADT/StringRef.h>
#include <llvm/Support/CommandLine.h>

#include <kernel/core/kernel_builder.h>
#include <kernel/core/streamset.h>
#include <kernel/pipeline/driver/cpudriver.h>
#include <kernel/pipeline/pipeline_builder.h>
#include <kernel/pipeline/program_builder.h>

#include "entropy.h"
#include "xorshiro.h"

// using std::unique_ptr;
// using std::make_unique;
// using std::string;
// using std::to_string;

extern Entropy ent;
extern XorShift256pp rng;
