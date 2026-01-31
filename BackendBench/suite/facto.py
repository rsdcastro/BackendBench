# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import logging
from collections import defaultdict

import torch
from torch.utils._python_dispatch import TorchDispatchMode

import traceback

try:
    from facto.inputgen.argtuple.gen import ArgumentTupleGenerator
    from facto.inputgen.utils.config import TensorConfig
    from facto.specdb.db import SpecDictDB
except ImportError:
    ArgumentTupleGenerator = None
    TensorConfig = None
    SpecDictDB = None


from BackendBench.eval import allclose
from BackendBench.op_categories import (
    RANDOM_OPS,
    TENSOR_CREATION_AND_MANIPULATION_OPS,
    UNSUPPORTED_OPERATORS,
)
from BackendBench.opregistry import get_operator

from .base import OpTest, TestSuite

logger = logging.getLogger(__name__)


class FactoTest:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


class FactoOpTest(OpTest):
    def __init__(self, op, correctness_tests):
        self.op = op
        self._correctness_tests = correctness_tests
        self.performance_tests = []

    @property
    def correctness_tests(self):
        for test in self._correctness_tests:
            yield FactoTest(*test.args, **test.kwargs)


class OpTracerMode(TorchDispatchMode):
    def __init__(self):
        self.ops = []
        self.args = []
        self.kwargs = []

    def __torch_dispatch__(self, fn, types, args=(), kwargs={}):
        self.ops.append(fn)
        self.args.append(args)
        self.kwargs.append(kwargs)
        return fn(*args, **kwargs)


def build_facto_op_tests(device, dtype, filter=None, num_runs=10, empty=False, probability=1.0):
    facto_op_tests = []
    failed = []

    orig_dtype = dtype

    logger.info(f"Building {num_runs} FACTO tests for {device} {dtype} with filter {filter}")
    for spec_name in SpecDictDB:
        try:
            # logger.info(f"Building FACTO tests for {spec_name}")
            if filter and spec_name not in filter:
                # logger.info(f"Skipping {spec_name}: not in filter")
                continue
            if (
                spec_name
                in UNSUPPORTED_OPERATORS + RANDOM_OPS + TENSOR_CREATION_AND_MANIPULATION_OPS
            ):
                # logger.info(f"Skipping {spec_name}: unsupported operator")
                continue

            # Get canonical operator from registry
            op = get_operator(spec_name)
            if op is None:
                logger.info(f"Skipping {spec_name}: operator resolution failed")
                continue

            logger.info(f"Building FACTO tests for {spec_name} with op {op} ({spec_name})")

            dtype = orig_dtype
            if spec_name == "svd.default" or spec_name == "triangular_solve.default" or spec_name == "special_zeta.default":
                dtype = torch.float32

            config = TensorConfig(
                empty=empty,
                # half_precision=True,
                device='cuda' if torch.cuda.is_available() else 'cpu',
            ).set_probability(probability)

            spec = SpecDictDB[spec_name]
            generator = ArgumentTupleGenerator(spec, config)

            op_tests = defaultdict(list)

            for idx, (posargs, inkwargs, outargs) in enumerate(generator.gen()):
                logger.info(f"Generated {idx}th test for {spec_name}")
                if idx >= num_runs:
                    break

                # Filter arguments to target device/dtype
                filtered_posargs = []
                for arg in posargs:
                    # logger.info(f"posargs: {type(arg)} {arg}")
                    if isinstance(arg, torch.Tensor):
                        logger.info(f"posargs dim: {arg.shape}")
                        # HACK HACK: to address batch size
                        if spec_name == "lstm.data" and arg.shape == torch.Size([1]):
                            arg = arg.to(device='cpu', dtype=torch.int64)
                        # arg = arg.to(device=device, dtype=dtype)
                    if isinstance(arg, list):
                        logger.info(f"tensor list: {len(arg)}")
                        for t in arg:
                            logger.info(f"posargs (list object): {type(arg)}")
                            if isinstance(t, torch.Tensor):
                                logger.info(f"posargs dim: {t.shape}")
                    if isinstance(arg, int) or isinstance(arg, float) or isinstance(arg, bool) or isinstance(arg, str):
                        logger.info(f"posargs: {arg}")
                    filtered_posargs.append(arg)

                filtered_inkwargs = {}
                for k, v in inkwargs.items():
                    if isinstance(v, torch.Tensor):
                        logger.info(f"inkwards dim: {v.shape}")
                        # v = v.to(device=device, dtype=dtype)                        
                    if isinstance(v, list):
                        logger.info(f"tensor list: {len(v)}")
                        for t in v:
                            if isinstance(t, torch.Tensor):
                                logger.info(f"inkwargs dim: {t.shape}")
                    if isinstance(arg, int) or isinstance(arg, float) or isinstance(arg, bool) or isinstance(arg, str):
                        logger.info(f"inkwargs: {arg}")
                    filtered_inkwargs[k] = v

                filtered_outargs = {}
                for k, v in outargs.items():
                    if isinstance(v, torch.Tensor):
                        logger.info(f"outargs dim: {v.shape}")
                        # v = v.to(device=device, dtype=dtype)
                    if isinstance(v, list):
                        logger.info(f"tensor list: {len(v)}")
                        for t in v:
                            if isinstance(t, torch.Tensor):
                                logger.info(f"outargs dim: {t.shape}")
                    if isinstance(arg, int) or isinstance(arg, float) or isinstance(arg, bool) or isinstance(arg, str):
                        logger.info(f"outargs: {arg}")
                    filtered_outargs[k] = v

                all_kwargs = {**filtered_inkwargs, **filtered_outargs}

                try:
                    # Trace execution to find underlying PyTorch ops
                    with OpTracerMode() as tracer:
                        # logger.info(f"args: {filtered_posargs} kwargs: {all_kwargs}")
                        print(f"op being executed: {op}")
                        ref = op(*filtered_posargs, **all_kwargs)
                except Exception as e:
                    logger.info(f"FACTO spec {spec_name} couldn't run underlying op {op} due to {e}")
                    # logger.info(f"args: {filtered_posargs} kwargs: {all_kwargs}")
                    traceback.print_exc()
                    continue

                logger.info(f"Traced {len(tracer.ops)} ops")
                # Check if we captured exactly one op (clean mapping)
                if len(tracer.ops) == 1:
                    try:
                        # Verify the traced op produces the same result
                        # logger.info(f"args: {filtered_posargs} kwargs: {all_kwargs}")
                        res = tracer.ops[0](*filtered_posargs, **all_kwargs)
                        if allclose(ref, res):
                            op_tests[tracer.ops[0]].append(
                                FactoTest(*filtered_posargs, **all_kwargs)
                            )
                    except Exception as e:
                        logger.info(
                            f"FACTO spec {spec_name} couldn't run underlying op {tracer.ops[0]} due to {e}"
                        )
                        traceback.print_exc()
                else:
                    logger.info(f"FACTO spec {spec_name} has {len(tracer.ops)} ops")
                    for traced_op in tracer.ops:
                        logger.info(f"op: {traced_op}")

            logger.info(f"DONE with FACTO generation after idx {idx}")

            for traced_op, tests in op_tests.items():
                if len(tests) > 0:
                    facto_op_tests.append(FactoOpTest(traced_op, tests))
        except Exception as e:
            logger.info(f"FACTO spec {spec_name} failed due to {e}")
            failed.append(spec_name)
            traceback.print_exc()

    logger.info(f"Failed specs: {failed}")

    return facto_op_tests


class FactoTestSuite(TestSuite):
    def __init__(self, name, device, dtype, filter=None, num_runs=10, empty=False, probability=1.0):
        super().__init__(
            name, build_facto_op_tests(device, dtype, filter, num_runs, empty, probability)
        )
