#!/usr/bin/python3

import argparse, logging, math, os, re, subprocess, sys, yaml
from typing import Callable
from dataclasses import dataclass
from functools import reduce, partial
from operator import xor
from pathlib import Path

mlog = logging.Logger(__name__)

cmake_configure_args = ["-DCMAKE_BUILD_TYPE=Debug"]
cmake_build_args = ["-j", 12]


def default_immediate_order(imm_range: int):
    bits = imm_range.bit_length()
    basis = [(1 << (4 - i)) - 1 for i in range(bits)]
    swizzle_basis = [
        basis[bits - (i >> 1)] if i & 1 else basis[i >> 1] for i in range(bits)
    ]
    imms = [
        reduce(xor, (swizzle_basis[j] * ((i >> j) & 1) for j in range(4)))
        for i in range(1 << bits)
    ]
    return [imm for imm in imms if imm < imm_range]


@dataclass(frozen=True)
class OperationMetadata:
    num_operands: int
    bitblock: bool = False
    bit_immed: bool = False
    field_immed: bool = False
    fixed_immed: int | None = None

    @property
    def takes_immediate(self) -> bool:
        assert self.bit_immed + self.field_immed + (self.fixed_immed
                                                    is not None) <= 1
        return self.bit_immed or self.field_immed or (self.fixed_immed
                                                      is not None)

    def immediate_range(self, bitblock_width: int, field_width: int):
        if self.fixed_immed is not None:
            assert not self.bit_immed and not self.field_immed
            return self.fixed_immed
        if self.bitblock:
            assert not self.field_immed
            if self.bit_immed:
                return bitblock_width
            else:
                return 0
        elif self.bit_immed:
            assert not self.field_immed
            assert field_width != 0
            return field_width
        elif self.field_immed:
            assert field_width != 0
            assert bitblock_width > field_width
            assert (bitblock_width % field_width) == 0
            return bitblock_width // field_width
        else:
            return 0


unary = OperationMetadata(1)
unary_bitblock = OperationMetadata(1, bitblock=True)
unary_bitblock_immed = OperationMetadata(1, bitblock=True, bit_immed=True)
unary_bit_immed = OperationMetadata(1, bit_immed=True)
unary_field_immed = OperationMetadata(1, field_immed=True)
binary = OperationMetadata(2)
binary_bitblock = OperationMetadata(2, bitblock=True)
binary_bitblock_immed = OperationMetadata(2, bitblock=True, bit_immed=True)
binary_field_immed = OperationMetadata(2, field_immed=True)
ternary = OperationMetadata(3)
ternary_bitblock_immed = OperationMetadata(3, bitblock=True, bit_immed=True)

all_ops: dict[str, OperationMetadata] = {
    "simd_select_hi": unary,
    "simd_select_lo": unary,
    "simd_fill": unary,
    "esimd_bitspread": unary,
    "bitblock_any": unary,
    "bitblock_popcount": unary,
    "simd_not": unary_bitblock,
    "bitblock_mask_from": unary_bitblock_immed,
    "bitblock_mask_to": unary_bitblock_immed,
    "bitblock_set_bit": unary_bitblock_immed,
    "simd_slli": unary_bit_immed,
    "simd_srli": unary_bit_immed,
    "simd_srai": unary_bit_immed,
    "mvmd_extract": unary_field_immed,
    "mvmd_slli": unary_field_immed,
    "mvmd_srli": unary_field_immed,
    "simd_add": binary,
    "simd_sub": binary,
    "simd_mult": binary,
    "simd_eq": binary,
    "simd_ne": binary,
    "simd_gt": binary,
    "simd_ugt": binary,
    "simd_ge": binary,
    "simd_uge": binary,
    "simd_lt": binary,
    "simd_le": binary,
    "simd_ult": binary,
    "simd_ule": binary,
    "simd_max": binary,
    "simd_min": binary,
    "simd_umax": binary,
    "simd_umin": binary,
    "simd_sllv": binary,
    "simd_srlv": binary,
    "simd_rotl": binary,
    "simd_rotr": binary,
    "simd_pext": binary,
    "simd_pdep": binary,
    "esimd_mergeh": binary,
    "esimd_mergel": binary,
    "hsimd_packh": binary,
    "hsimd_packl": binary,
    "hsimd_packus": binary,
    "hsimd_packss": binary,
    "mvmd_sll": binary,
    "mvmd_srl": binary,
    "mvmd_shuffle": binary,
    "mvmd_shuffle:over": binary,
    "mvmd_shuffle:highbit": binary,
    "mvmd_compress": binary,
    "mvmd_expand": binary,
    "simd_and": binary_bitblock,
    "simd_or": binary_bitblock,
    "simd_xor": binary_bitblock,
    "simd_binary": OperationMetadata(2, bitblock=True, fixed_immed=(1 << 2)),
    "mvmd_insert": binary_field_immed,
    "mvmd_dslli": binary_field_immed,
    "bitblock_advance.shiftout": binary_bitblock_immed,
    "bitblock_advance.shifted": binary_bitblock_immed,
    "simd_if": ternary,
    "bitblock_add_with_carry.sum": ternary,
    "bitblock_add_with_carry.carry": ternary,
    "bitblock_subtract_with_borrow.diff": ternary,
    "bitblock_subtract_with_borrow.borrow": ternary,
    "simd_ternary": OperationMetadata(3, bitblock=True, fixed_immed=(1 << 3)),
    "bitblock_indexed_advance.shiftout": ternary_bitblock_immed,
    "bitblock_indexed_advance.shifted": ternary_bitblock_immed,
}


@dataclass(frozen=True)
class OperationConfig:
    operation: str
    field_width: int
    immediate: int | None = None


@dataclass(frozen=True)
class ExerciserProcedure:
    name: str
    exerciser_path: Path | str
    extra_args: list[Path | str]
    block_width: int
    default_test_operand_vectors: list[Path | str]
    default_timing_operand_vectors: list[Path | str]
    specialized_test_operand_vectors: dict[str, list[Path | str]]
    specialized_timing_operand_vectors: dict[str, list[Path | str]]
    timing_warmup: int = 2
    timing_repeat: int = 500
    timing_drop_worst: int = 0
    timing_drop_best: int = 0
    timing_capture_log_path: Callable[..., Path | str
                                      | None] = lambda proc, cfg: None
    test_capture_ir_path: Callable[...,
                                   Path | str | None] = lambda proc, cfg: None
    test_capture_asm_path: Callable[...,
                                    Path | str | None] = lambda proc, cfg: None
    test_capture_output_path: Callable[..., Path | str
                                       | None] = lambda proc, cfg: None
    test_capture_log_path: Callable[..., Path | str
                                    | None] = lambda proc, cfg: None

    def prep_test_run(self, cfg: OperationConfig) -> Callable:
        global project_dir
        meta = all_ops[cfg.operation]
        operands = self.specialized_test_operand_vectors.get(
            cfg.operation, self.default_test_operand_vectors)

        ir_path = self.prep_file(cfg, self.test_capture_ir_path)
        asm_path = self.prep_file(cfg, self.test_capture_asm_path)
        output_path = self.prep_file(cfg, self.test_capture_output_path)
        log_path = self.prep_file(cfg, self.test_capture_log_path)
        return partial(run_idisa_exerciser,
                       project_dir / self.exerciser_path,
                       self.extra_args,
                       cfg.operation,
                       cfg.field_width,
                       operands[:meta.num_operands],
                       output_path=output_path,
                       ir_path=ir_path,
                       asm_path=asm_path,
                       log_path=log_path)

    def prep_timing_run(self, cfg: OperationConfig) -> Callable:
        global project_dir
        meta = all_ops[cfg.operation]
        operands = self.specialized_timing_operand_vectors.get(
            cfg.operation, self.default_timing_operand_vectors)
        log_path = self.prep_file(cfg, self.test_capture_log_path)
        return partial(run_idisa_exerciser,
                       project_dir / self.exerciser_path,
                       self.extra_args,
                       cfg.operation,
                       cfg.field_width,
                       operands[:meta.num_operands],
                       warmup=self.timing_warmup,
                       repeat=self.timing_repeat,
                       drop_worst=self.timing_drop_worst,
                       drop_best=self.timing_drop_best,
                       log_path=log_path)

    def prep_file(self, cfg, path_f):
        path = path_f(self, cfg)
        if path is not None:
            path = project_dir / path
            path.parent.mkdir(parents=True, exist_ok=True)
        return path


def make_configs(
    proc: ExerciserProcedure,
    operations: str | list[str],
    field_widths: int | list[int],
    immediates: None | int | float | list[int] = None,
):
    # regularize parameters
    if isinstance(operations, str):
        operation = [operations]
    if isinstance(field_widths, int):
        field_widths = [field_widths]

    # generate combinations
    for operation in operations:
        assert operation in all_ops
        meta = all_ops[operation]
        for fw in (field_widths if not meta.bitblock else [None]):
            assert type(fw) == int
            this_imms = [None]
            if meta.takes_immediate:
                imm_range = meta.immediate_range(proc.block_width, fw)
                if isinstance(immediates, list):
                    this_imms = [imm for imm in immediates if imm < imm_range]
                    this_imms = default_immediate_order(imm_range)
                    if immediates is None:
                        this_imms = this_imms[:3]
                    elif type(immediates) == int:
                        this_imms = this_imms[:immediates]
                    elif immediates > 0 and immediates < 1:
                        this_imms = this_imms[:math.ceil(
                            len(this_imms) * immediates)]
                for imm in this_imms:
                    yield OperationConfig(operation, fw, imm)
            else:
                yield OperationConfig(operation, fw)


def find_project_dir(start_path):
    cur = start_path
    explored = set()
    while True:
        # print(f"Checking {cur}..")
        if (cur / "scripts").is_dir() and (cur / "scripts" / "idisa_suite" /
                                           "idisa_exerciser.py").is_file():
            return cur
        explored.add(cur)
        parent = cur.parent
        if parent in explored:
            raise Exception(
                "Unable to find project root (reached system root or a cycle while looking)"
            )
        cur = parent


project_dir: Path
idisa_suite_dir: Path
build_dir: Path
idisa_exerciser_path: Path


def macro_subs(s: str, macros: dict[str, str]):
    parts = []
    cur = len(s)
    for m in reversed(list(re.finditer("%[-_0-9A-Za-z]*%", s))):
        macro = m.group()[1:-1]
        b, e = m.span()
        parts.append(s[e:cur])
        if macro == '':
            parts.append('%')
        else:
            parts.append(macros.get(macro, '<BAD MACRO>'))
        cur = b
    parts.append(s[:cur])
    parts.reverse()
    return ''.join(parts)


def init_paths():
    global project_dir
    global idisa_suite_dir

    project_dir = find_project_dir(Path.cwd())
    idisa_suite_dir = project_dir / "scripts" / "idisa_suite"


# def guess_config():
#     global idisa_exerciser_path
#     for d in ["build", "build-vscode"]:
#         build_dir = project_dir / d
#         if build_dir.is_dir() and (build_dir / "CMakeCache.txt").is_file():
#             break
#     else:
#         build_dir = None
#     if build_dir is not None:
#         idisa_exerciser_path = build_dir / "bin" / "idisa_exerciser"


def set_build_dir(rel_dir):
    global build_dir
    build_dir = project_dir / rel_dir


def run_project_cmake(*args):
    was_cur_dir: Path = Path.cwd()
    os.chdir(project_dir)
    try:
        return subprocess.call(["cmake"] + args)
    finally:
        try:
            os.chdir(was_cur_dir)
        except:
            pass


def build_idisa_exerciser(build_dir: Path | str | None = None):
    global idisa_exerciser_path
    if build_dir is None:
        build_dir = project_dir / "build"
    else:
        build_dir = Path(build_dir)
    if not build_dir.is_dir():
        mlog.info("Build dir not found, reconfiguring project")
        result = run_project_cmake("-B",
                                   str(build_dir.relative_to(project_dir)),
                                   "-S", ".", "-G", "Ninja")
        if result != 0:
            raise Exception("cmake failed during configure")
    result = run_project_cmake("--build",
                               str(build_dir.relative_to(project_dir)),
                               *cmake_build_args)
    if result != 0:
        raise Exception("cmake failed during build")
    if not idisa_exerciser_path.is_file:
        raise Exception("cmake finished but idisa_exerciser not found?")


def run_idisa_exerciser(exerciser_path,
                        extra_args,
                        operation,
                        field_width,
                        op_args,
                        disable_checks=False,
                        warmup=None,
                        repeat=None,
                        drop_worst=None,
                        drop_best=None,
                        output_path=None,
                        ir_path=None,
                        unopt_ir_path=None,
                        asm_path=None,
                        log_path=None):
    cmd = [str(exerciser_path)]
    if disable_checks:
        cmd.append("--disable-checks")
    if warmup is not None:
        cmd += ["--warmup", str(warmup)]
    if drop_best is not None:
        cmd += ["--drop-best", str(drop_best)]
    if drop_worst is not None:
        cmd += ["--drop-worst", str(drop_worst)]
    if repeat is not None:
        cmd += ["--repeat", str(repeat)]
    if output_path is not None:
        cmd.append(f"--output={str(output_path)}")
    if ir_path is not None:
        cmd.append(f"--ShowIR={str(ir_path)}")
    if unopt_ir_path is not None:
        cmd.append(f"--ShowUnoptIR={str(unopt_ir_path)}")
    if asm_path is not None:
        cmd.append(f"--ShowASM={str(asm_path)}")
    cmd += [str(arg) for arg in extra_args]
    cmd += [str(operation), str(field_width)]
    cmd += [str(arg) for arg in op_args]
    mlog.debug("Running %s", " ".join(cmd))
    log_file = None
    if log_path is not None:
        log_file = os.open(log_path, os.O_WRONLY | os.O_APPEND | os.O_CREAT)
    result = subprocess.call(cmd, stdout=log_file, stderr=log_file)
    if log_file is not None:
        os.close(log_file)
    return result, log_path


def main():
    global mlog

    logging.basicConfig(level=logging.INFO)
    mlog = logging.root

    def macros_from(proc, opcfg):
        return {
            'mode':
            str(proc.name),
            'bw':
            str(proc.block_width),
            'bw2':
            '%02d' % (proc.block_width, ),
            'bw3':
            '%03d' % (proc.block_width, ),
            'bw4':
            '%04d' % (proc.block_width, ),
            'op':
            opcfg.operation,
            'fw':
            str(opcfg.field_width),
            'fw2':
            '%02d' % (opcfg.field_width, ),
            'fw3':
            '%03d' % (opcfg.field_width, ),
            'imm-':
            '-' if opcfg.immediate is not None else '',
            'imm_':
            '-' if opcfg.immediate is not None else '',
            'imm':
            str(opcfg.immediate) if opcfg.immediate is not None else '',
            'imm2':
            ('%02d' %
             (opcfg.immediate, )) if opcfg.immediate is not None else '',
            'imm3':
            ('%03d' %
             (opcfg.immediate, )) if opcfg.immediate is not None else '',
        }

    init_paths()
    config = yaml.safe_load(open(idisa_suite_dir / "config.yaml"))

    # print(repr(config))

    test_config = config["test"]
    timing_config = config["timing"]
    test_vectors = list(map(lambda p: project_dir / p, test_config["vectors"]))
    timing_vectors = list(
        map(lambda p: project_dir / p, timing_config["vectors"]))

    procedures = {}
    for mode in config["modes"]:

        def extract_count(cfg, name, default):
            if name not in cfg:
                return default
            return int(cfg[name])

        def extract_path_f(cfg, name):
            if (name not in cfg) or (not cfg[name]):
                return (lambda proc, opcfg: None)
            return (lambda proc, opcfg: macro_subs(str(cfg[name]),
                                                   macros_from(proc, opcfg)))

        procedures[mode["name"]] = ExerciserProcedure(
            mode["name"],
            mode["exerciser-path"],
            mode.get("extra-args", []),
            int(mode["block-size"]),
            test_vectors,
            timing_vectors,
            {},
            {},
            timing_warmup=extract_count(timing_config, "warmup", None),
            timing_repeat=extract_count(timing_config, "repeat", 1),
            timing_drop_worst=extract_count(timing_config, "drop-worst", 0),
            timing_drop_best=extract_count(timing_config, "drop-best", 0),
            timing_capture_log_path=extract_path_f(timing_config, "log-path"),
            test_capture_ir_path=extract_path_f(test_config, "ir-path"),
            test_capture_asm_path=extract_path_f(test_config, "asm-path"),
            test_capture_output_path=extract_path_f(test_config,
                                                    "output-path"),
            test_capture_log_path=extract_path_f(test_config, "log-path"),
        )

    parser = argparse.ArgumentParser(
        sys.argv[0],
        description=
        "Run the idisa_exerciser for different combinations of parameters, "
        "storing outputs to the standard locations",
    )
    parser.add_argument("modes",
                        type=str,
                        nargs="?",
                        default=config["default"].get("modes"))
    parser.add_argument("operations",
                        type=str,
                        nargs="?",
                        default=config["default"].get("operations"))
    parser.add_argument("field_widths",
                        type=str,
                        nargs="?",
                        default=config["default"].get("field-widths"))
    parser.add_argument("--time", action="store_true", default=False)
    args: argparse.Namespace = parser.parse_args()
    do_modes = [m for m in args.modes.split(",") if m]
    do_ops = [o for o in args.operations.split(",") if o]
    do_fws = [int(s) for s in args.field_widths.split(",") if s]

    for m in do_modes:
        if m not in procedures:
            raise Exception(
                f"mode {m} not recognized (configured options: {', '.join(procedures.keys())})"
            )

    if args.time:
        for m in do_modes:
            p = procedures[m]
            for c in make_configs(p, do_ops, do_fws):
                fn = p.prep_timing_run(c)
                fn()
    else:
        mlog.info("mode,bw,op,fw,imm,status,log")
        for m in do_modes:
            p = procedures[m]
            for c in make_configs(p, do_ops, do_fws):
                fn = p.prep_test_run(c)
                res, logf = fn()
                mlog.info(
                    f"{p.name},{p.block_width},{c.operation},{c.field_width},"
                    f"{c.immediate if c.immediate is not None else ''},"
                    f"{'pass' if res == 0 else 'FAIL'},{logf}")


if __name__ == "__main__":
    main()
