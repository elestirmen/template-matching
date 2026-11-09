"""Detect sticky CUDA faults in the output of a GPU job, so the job is stopped instead of retried.

After one of these errors the process's CUDA context is unusable. Starting the job again on the same GPU is not a
clean retry. On 2026-11-09 the replay (TensorFlow 2.10 / CUDA 11 on an RTX 5060 Ti) hit 'an illegal memory access'
six times overnight and was restarted after each one. The sixth restart ended in a GPU reset (TDR), a hung NVIDIA
driver (LiveKernelEvent 0x1CC in nvlddmkm) and a frozen machine that even remote desktop could not reach.
replay_flights.py's supervisor and run_all.py therefore stop at the first such line, and the work is resumed after
a reboot.
"""
import re

# Exit code for "stopped after a sticky CUDA fault": the supervisor returns it, and run_all.py exits with it.
# run_replay.bat stops on any non-zero code.
GPU_FAULT_RC = 86

# The CUDA errors that leave the context unusable (cudaError 700, 714-719), as TensorFlow/CUDA print them,
# and the driver's message for a GPU that is no longer on the PCIe bus.
_PATTERN = re.compile(
    r'illegal memory access|CUDA_ERROR_ILLEGAL_ADDRESS'
    r'|illegal instruction was encountered|CUDA_ERROR_ILLEGAL_INSTRUCTION'
    r'|misaligned address was encountered|CUDA_ERROR_MISALIGNED_ADDRESS'
    r'|CUDA_ERROR_INVALID_ADDRESS_SPACE|CUDA_ERROR_INVALID_PC|CUDA_ERROR_HARDWARE_STACK_ERROR'
    r'|unspecified launch failure|CUDA_ERROR_LAUNCH_FAILED'
    r'|GPU has fallen off the bus',
    re.IGNORECASE)


def is_gpu_fault(line):
    """True if a line of job output reports a sticky CUDA fault."""
    return _PATTERN.search(line) is not None
