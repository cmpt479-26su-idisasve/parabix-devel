/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 */

#include "idisa_exerciser.h"

#include <kernel/io/stdout_kernel.h>
#include <kernel/util/hex_convert.h>
#include <toolchain/toolchain.h>

#include "idisa_operations.h"

using namespace std;
using namespace llvm;
using namespace kernel;

Entropy ent;
XorShift256pp rng(ent.nextWord());

#if 0
// Cribbed from LLVM documentation August 2026
struct FileSizeParser : public cl::parser<uint64_t> {
    // parse - Return true on error.
    bool parse(cl::Option &opt, StringRef argName, const std::string &arg, unsigned &val);
};

bool FileSizeParser::parse(cl::Option &opt, StringRef argName, const std::string &arg, uint64_t &val) {
    char *p;
    // Parse integer part, leaving 'p' pointing to the first non-integer char
    val = (uint64_t)strtoll(arg.c_str(), &p, 0);

    switch (tolower(*p)) {
    case 'g':
        val *= 1024 * 1024 * 1024;
        ++p;
        break;
    case 'm':
        val *= 1024 * 1024;
        ++p;
        break;
    case 'k':
        val *= 1024;
        ++p;
        break;
    case 0:
        goto done;
    default:
        goto fail;
    }

    // Ignore the 'i' in KiB if people use that
    if (tolower(*p) == 'i') {
        ++p;
    }
    // Ignore B suffix
    if (tolower(*end) == 'b') {
        ++p;
    }
    // Check for extra junk of any other kind
    if (*p != 0) {
        goto fail;
    }
    // fall through!
done:
    return false; // No error
fail:
    return opt.error("'" + arg + "' value invalid for file size argument!");
}
#endif

const char *const ProgramName = "idisa_exerciser";

string supportedOperationsDesc = []() {
    stringstream result;
    result << "\nSUPPORTED OPERATIONS:\n\n";
    for (auto t : allOperationHelpDescs()) {
        StringRef name, operands, desc;
        tie(name, operands, desc) = t;
        result << setw(4) << " " << setw(22) << left << name.str() << " " << setw(20) << operands.str();
        if (!desc.empty()) {
            result << " - " << desc.str();
        }
        result << '\n';
    }
    result << '\n';
    return result.str();
}();

// static cl::OptionCategory SupportedOperations("A. Supported IDISA Operations", supportedOperationsDesc);
cl::extrahelp SupportedOperationsHelp(supportedOperationsDesc);

static cl::opt<string> OperationName(cl::Positional, cl::desc("<IDISA op>"), cl::Required);
static cl::opt<unsigned> OperationFieldWidth(cl::Positional, cl::desc("<field width>"), cl::Required);
static cl::list<string> OperationArgs(cl::ConsumeAfter, cl::desc("[operation args..]"));

static cl::OptionCategory ExerciserFlags("C. Command Flags");

static cl::opt<string> OperationOutputFile("output", cl::value_desc("output"),
                                           cl::desc("Write the output of the operation to a file."),
                                           cl::cat(ExerciserFlags));
static cl::opt<bool> OperationOutputHex("x", cl::desc("Write the output as a hex dump."), cl::cat(ExerciserFlags));

static cl::opt<bool> DisableChecks("disable-checks",
                                   cl::desc("Don't run checks (for more precise timing comparisons)."),
                                   cl::cat(ExerciserFlags));

static cl::opt<unsigned>
    WarmupCount("warmup", cl::init(0), cl::value_desc("runs"),
                cl::desc("Run the operation on all input a number of times before recording timings."),
                cl::cat(ExerciserFlags));
static cl::opt<unsigned> RepeatCount("repeat", cl::init(1), cl::value_desc("runs"),
                                     cl::desc("Re-run the operation multiple times."), cl::cat(ExerciserFlags));
static cl::opt<unsigned> DropBestCount("drop-best", cl::init(0), cl::value_desc("runs"),
                                       cl::desc("Drop the best timing(s) from the average over multiple runs."),
                                       cl::cat(ExerciserFlags));
static cl::opt<unsigned> DropWorstCount("drop-worst", cl::init(0), cl::value_desc("runs"),
                                        cl::desc("Drop the worst timing(s) from the average over multiple runs."),
                                        cl::cat(ExerciserFlags));

static cl::opt<bool> QuietMode("q", cl::desc("Suppress output, set the return code only."), cl::cat(ExerciserFlags));

static cl::opt<bool> ReportTiming("timing", cl::desc("Report pipeline compilation and kernel execution time."),
                                  cl::init(false), cl::cat(ExerciserFlags));

int main(int argc, char *argv[]) {
    // The argument parsing for the operations is kind of janky. Another possibly better way is to register each
    // different one as a subcommand -- then they'd be able to get their own positional options in a way that LLVM
    // CommandLine would understand natively, and they'd maybe-get better-formatted help.

    // I didn't know that subcommands existed in the library when I started so I didn't make allowances for it. Thems
    // the breaks

    string overview =
        "Exercise an IDISA operation implementation, by comparing to a scalar reference or by timing runs.";
    codegen::ParseCommandLineOptions(argc, argv, {&ExerciserFlags, codegen::codegen_flags()}, overview);

    // Disable caching, if we're doing testing it's almost certainly a hazard -- also, it breaks our method for figuring
    // out which builder was used for compilation
    codegen::EnableObjectCache = false;
    codegen::EnablePipelineObjectCache = false;

    unique_ptr<OperationConfig> operationConfig;

    // Find opName in configurator list
    if (makeOperationConfig(OperationName, OperationFieldWidth, QuietMode, operationConfig)) {
        return 2;
    }
    if (operationConfig->isStdinGrabbed() && ((WarmupCount != 0) || (RepeatCount != 1))) {
        OperationArgs.error("Input can only come from STDIN if repeat count is 1, with no warmup.");
        return 2;
    }
    // I don't have a better dummy sink than file output right now, so I guess we permit this
    // if (!OperationOutputFile.empty() && (WarmupCount != 0)) {
    //     WarmupCount.error("Output can only be recorded if repeat count is 1, with no warmup.");
    //     return 2;
    // }
    // if (!OperationOutputFile.empty() && (RepeatCount != 1)) {
    //     RepeatCount.error("Output can only be recorded if repeat count is 1, with no warmup.");
    //     return 2;
    // }
    if (DropBestCount + DropWorstCount >= RepeatCount) {
        RepeatCount.error("Dropping more run timings than will be captured.");
        return 2;
    }

    if (ReportTiming && !DisableChecks) {
        outs() << "timing: warning: checks are enabled, parallelization of the test and check kernels can result in "
                  "timing which only depends on the slowest kernel\n";
    }
    CPUDriver driver("idisa_exerciser");
    operationConfig->constructPipeline(driver);
    if (operationConfig->configurePipelineFromArgs(OperationArgs, !DisableChecks)) {
        return 2;
    }
    if (!OperationOutputFile.empty()) {
        StreamSet *convertedOutput = operationConfig->getOutputStream();
        PipelineBuilder &p = operationConfig->getPipelineBuilder();
        if (OperationOutputHex) {
            StreamSet *outputAsHex = p.CreateStreamSet(1, 8);
            p.CreateKernelCall<BinaryToHex>(convertedOutput, outputAsHex);
            convertedOutput = outputAsHex;
        }
        Scalar *outputFilename = p.getInputScalar(OperationConfig::outputFilenameIdent);
        p.CreateKernelCall<FileSink>(outputFilename, convertedOutput);
        operationConfig->setOutputFilename(OperationOutputFile);
    }

    chrono::steady_clock::time_point compileStart;
    if (ReportTiming) {
        compileStart = std::chrono::steady_clock::now();
    }
    operationConfig->compilePipeline();
    if (ReportTiming) {
        auto compileTimeUs = chrono::duration_cast<chrono::microseconds>(chrono::steady_clock::now() - compileStart);
        outs() << "timing: compile: " << operationConfig->getKernelBuilderID() << " took " << compileTimeUs.count() << " us\n";
    }
    if (!QuietMode && !DisableChecks) {
        outs() << "test: built with " << operationConfig->getKernelBuilderID() << "\n";
    }

    vector<chrono::microseconds> execTimesUs;
    for (unsigned i = 0; i < WarmupCount; ++i) {
        size_t failureCount = operationConfig->executePipeline();
        if (!DisableChecks && (failureCount > 0)) {
            if (!QuietMode) {
                outs() << "test: fail: " << operationConfig->getDescription() << ": " << failureCount
                       << " incorrect blocks during warmup #" << i << "\n";
            }
            if (ReportTiming) {
                outs() << "timing: aborting due to test failure during warmup #" << i << "\n";
            }
            return 1;
        }
    }
    for (unsigned i = 0; i < RepeatCount; ++i) {
        chrono::steady_clock::time_point execStart;
        if (ReportTiming) {
            execStart = std::chrono::steady_clock::now();
        }
        size_t failureCount = operationConfig->executePipeline();
        if (ReportTiming) {
            execTimesUs.push_back(
                chrono::duration_cast<chrono::microseconds>(std::chrono::steady_clock::now() - execStart));
        }
        if (failureCount > 0) {
            if (!QuietMode) {
                outs() << "test: fail: " << operationConfig->getDescription() << ": " << failureCount
                       << " incorrect blocks during run #" << i << "\n";
            }
            if (ReportTiming) {
                outs() << "timing: aborting due to test failure during run #" << i << "\n";
            }
            return 1;
        }
    }

    if (!QuietMode) {
        if (DisableChecks) {
            outs() << "test: disabled\n";
        } else {
            outs() << "test: pass: " << operationConfig->getDescription() << "\n";
        }
    }

    if (ReportTiming) {
        assert(execTimesUs.size() > 0);
        if (execTimesUs.size() == 0) {
            outs() << "timing: kernel execution: no data!\n";
        } else if (execTimesUs.size() == 1) {
            outs() << "timing: kernel execution: " << execTimesUs[0].count() << " us\n";
        } else {
            if (!QuietMode) {
                outs() << "timing: kernel execution: raw data us (including dropped elements): ";
                unsigned i = 0;
                for (auto const &t : execTimesUs) {
                    if ((i >= 1000) && (execTimesUs.size() > 1000)) {
                        outs() << " // Data table too large, truncating to 1000 elements";
                        break;
                    }
                    if (i > 0)
                        outs() << ", ";
                    outs() << t.count();
                    ++i;
                }
                outs() << "\n";
            }
            std::sort(execTimesUs.begin(), execTimesUs.end());
            execTimesUs.erase(execTimesUs.end() - DropWorstCount, execTimesUs.end());
            execTimesUs.erase(execTimesUs.begin(), execTimesUs.begin() + DropBestCount);
            double sumExecUs = 0.;
            for (auto const &t : execTimesUs) {
                sumExecUs += t.count();
            }
            double meanExecUs = sumExecUs / execTimesUs.size();
            double devExecUs2 = 0.;
            for (auto const &t : execTimesUs) {
                double dExecUs = t.count() - meanExecUs;
                devExecUs2 += dExecUs * dExecUs;
            }
            // -1 from size for sample deviation
            double stdevExecUs = sqrt(devExecUs2 / (execTimesUs.size() - 1));
            outs() << "timing: kernel execution: " << static_cast<int64_t>(round(meanExecUs)) << " us mean, "
                   << static_cast<int64_t>(round(stdevExecUs)) << " us sample dev, " << execTimesUs.size()
                   << " runs counted";
            if (DropWorstCount || DropBestCount) {
                outs() << "; ";
                if (DropWorstCount) {
                    outs() << DropWorstCount << " worst";
                    if (DropBestCount) {
                        outs() << ", ";
                    }
                }
                if (DropBestCount) {
                    outs() << DropWorstCount << " best";
                }
                outs() << " not counted";
            }
            outs() << "\n";
        }
    }
    return 0;
}
