#########################################################################################
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries. All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc. and/or its subsidiaries.
#########################################################################################

"""Session-wide setup for the max-context suite.

Holds the three things that must happen exactly once per session regardless of which
test files are selected: the platform gate, the RNG seed, and the summary/teardown.
"""

import os
import sys


def _device_id_from_argv() -> str | None:
    """--device-id's value, read straight out of sys.argv.

    This has to happen before the `from vllm import platforms` import a few lines
    down, and pytest's own option machinery (pytest_addoption/config.getoption) is
    too late for that: conftest.py's module body -- including that import -- runs
    while pytest is *importing this file* to discover its hooks, which happens
    before argv is parsed against any option this file registers. Whatever
    QAIC_VISIBLE_DEVICES is at the moment vllm/torch_qaic first gets imported is
    what determines which physical devices the process can see; setting it later,
    in pytest_configure (as this used to do), was already too late -- confirmed by
    a live run where every job ended up on the same physical devices regardless of
    its assigned --device-id slice. A raw argv scan is the only thing early enough.
    """
    argv = sys.argv[1:]
    for index, arg in enumerate(argv):
        if arg == "--device-id" and index + 1 < len(argv):
            return argv[index + 1]
        if arg.startswith("--device-id="):
            return arg.split("=", 1)[1]
    return None


_device_id = _device_id_from_argv()
if _device_id:
    os.environ["QAIC_VISIBLE_DEVICES"] = _device_id

import pytest
import torch
from vllm import platforms

from ctx_config import RNG_SEED, SUMMARY_FILE
from engine_pool import release_engine
from run_metrics import SUMMARY_LEGEND, append_summary_workbook, render_summary

# Import-time gate, matching the other hw_models suites. Move this into an autouse
# fixture on the device tests if you want test_token_math_selftest.py -- which needs
# no device -- to be collectable off a QAIC host.
assert (
    platforms.current_platform.device_type == "qaic"
), "vLLM could not detect qaic plugin"

# Fixed seed so that token counts and generated text stay deterministic across runs.
torch.manual_seed(RNG_SEED)


def pytest_addoption(parser: pytest.Parser) -> None:
    """--device-id: registers the flag so pytest recognises it (--help, no
    "unrecognized arguments" error) and so scheduler.py's invocation is valid
    pytest usage. The value itself is already applied above, by _device_id_from_argv()
    -- config.getoption("device_id") below is for anything that wants to read it
    back (logging, assertions), not for applying it; that ship has already sailed
    by the time pytest_configure runs.
    """
    parser.addoption(
        "--device-id",
        action="store",
        default=None,
        help=(
            "Comma-separated physical QAIC device IDs this process is restricted "
            "to (e.g. 48,49,50,51,52,53,54,55). Sets QAIC_VISIBLE_DEVICES, read "
            "from sys.argv before any vllm import (see _device_id_from_argv() at "
            "the top of this file). Normally set by scheduler.py, not by hand."
        ),
    )


@pytest.fixture(scope="session", autouse=True)
def max_context_summary():
    """Print and append the run summary, then release the engine."""
    yield
    rendered = render_summary()
    if rendered:
        print("\n\n=== VLM Max-Context Summary ===")
        print(rendered)
        print(SUMMARY_LEGEND)
        append_summary_workbook(SUMMARY_FILE)
    else:
        print("No VLM max-context data collected.")
    release_engine()
