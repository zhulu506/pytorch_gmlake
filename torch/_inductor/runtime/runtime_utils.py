# mypy: allow-untyped-defs
from __future__ import annotations

import functools
import getpass
import inspect
import operator
import os
import re
import tempfile
import time

import torch


def conditional_product(*args):
    return functools.reduce(operator.mul, [x for x in args if x])


def ceildiv(numer: int, denom: int) -> int:
    return -(numer // -denom)


def is_power_of_2(n: int) -> bool:
    """Returns whether n = 2 ** m for some integer m."""
    return n > 0 and n & n - 1 == 0


def next_power_of_2(n: int) -> int:
    """Return the smallest power of 2 greater than or equal to n"""
    n -= 1
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    n |= n >> 32
    n += 1
    return n


def get_num_bytes(*args: torch.Tensor, num_in_out_args: int = 0) -> int:
    """
    Return the total number of bytes the arguments of tensor type takes.

    For in/out args, tensor sizes are counted twice: once for reading and
    once for writing.

    The first num_in_out_args arguments are in out tensors.
    """
    return sum(
        arg.numel() * arg.element_size() * (1 + int(i < num_in_out_args))
        for i, arg in enumerate(args)
        if isinstance(arg, torch.Tensor)
    )


def triton_config_to_hashable(cfg):
    """
    Convert triton config to a tuple that can uniquely identify it. We can use
    the return value as a dictionary key.
    """
    items = sorted(cfg.kwargs.items())
    items.append(("num_warps", cfg.num_warps))
    items.append(("num_stages", cfg.num_stages))
    return tuple(items)


def create_bandwidth_info_str(ms, num_gb, gb_per_s, prefix="", suffix="", color=True):
    info_str = f"{prefix}{ms:.3f}ms    \t{num_gb:.3f} GB \t {gb_per_s:7.2f}GB/s{suffix}"
    slow = ms > 0.012 and gb_per_s < 650
    return red_text(info_str) if color and slow else info_str


def get_max_y_grid():
    return 65535


def do_bench(fn, fn_args, fn_kwargs, **kwargs):
    from torch._inductor.utils import is_cpu_device

    args = list(fn_args)
    args.extend(fn_kwargs.values())
    if is_cpu_device(args):
        return do_bench_cpu(lambda: fn(*fn_args, **fn_kwargs), **kwargs)
    else:
        return do_bench_gpu(lambda: fn(*fn_args, **fn_kwargs), **kwargs)


def do_bench_gpu(*args, **kwargs):
    @functools.lru_cache(None)
    def load_triton():
        try:
            # NB: Lazily load triton, as importing triton is slow
            # see https://github.com/openai/triton/issues/1599
            from triton.testing import do_bench as triton_do_bench
        except ImportError as exc:
            raise NotImplementedError("requires Triton") from exc

        # triton PR https://github.com/openai/triton/pull/1513 change the
        # quantile fields name from 'percentiles' to 'quantiles'
        # and change the default value from (0.5, 0.2, 0.8) to None.
        # This may break inductor since a caller expects a tuple may get a item.
        #
        # Add a wrapper to maintain the same behavior for inductor.
        # Maybe we should have own implementation of this function?
        return triton_do_bench, (
            "quantiles"
            if inspect.signature(triton_do_bench).parameters.get("quantiles")
            is not None
            else "percentiles"
        )

    triton_do_bench, quantile_field_name = load_triton()

    benchmark_stream = torch.cuda.Stream()

    with torch.cuda.stream(benchmark_stream):
        timing = do_bench_cudagraph(*args, **kwargs)[0]
    
    return timing

    if quantile_field_name not in kwargs:
        kwargs[quantile_field_name] = (0.5, 0.2, 0.8)
    return triton_do_bench(*args, **kwargs)[0]


def do_bench_cpu(fn, warmup=5, times=20):
    assert times > 0
    for _ in range(warmup):
        fn()
    durations = []
    for _ in range(times):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        durations.append((t1 - t0) * 1000)
    # return the median time
    sorted_durations = sorted(durations)
    if times % 2 == 0:
        return (sorted_durations[times // 2 - 1] + sorted_durations[times // 2]) / 2
    else:
        return sorted_durations[times // 2]


def _summarize_statistics(times, quantiles, return_mode):
    if quantiles is not None:
        ret = torch.quantile(times, torch.tensor(quantiles, dtype=torch.float)).tolist()
        if len(ret) == 1:
            ret = ret[0]
        return ret
    return getattr(torch, return_mode)(times).item()


def do_bench_cudagraph(fn, rep=20, grad_to_none=None, quantiles=None, return_mode="mean", **kwargs):
    """
    Benchmark the runtime of the provided function.

    :param fn: Function to benchmark
    :type fn: Callable
    :param rep: Repetition time (in ms)
    :type rep: int
    :param grad_to_none: Reset the gradient of the provided tensor to None
    :type grad_to_none: torch.tensor, optional
    :param return_mode: The statistical measure to return. Options are "min", "max", "mean", or "median". Default is "mean".
    :type return_mode: str
    """
    assert return_mode in ["min", "max", "mean", "median"]

    if torch.cuda.current_stream() == torch.cuda.default_stream():
        raise RuntimeError("Cannot capture graph in default stream. Please use side stream in benchmark code.")
    # warmup
    fn()
    # step 1 - we estimate the amount of time the kernel call takes
    # NOTE: this estimate isn't super accurate because the GPU isn't warmed up at this point
    #       but it is probably good enough
    if grad_to_none is not None:
        for x in grad_to_none:
            x.detach_()
            x.requires_grad_(True)
            x.grad = None
    event_pairs = [
        (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True)
        )
        for _ in range(5)
    ]
    torch.cuda.synchronize()
    for start_event, end_event in event_pairs:
        start_event.record()
        fn()
        end_event.record()
    torch.cuda.synchronize()
    tt = 0
    for start_event, end_event in event_pairs:
        tt += start_event.elapsed_time(end_event)
    estimate_ms = tt / 5
    n_repeat = max(1, int(rep / estimate_ms))
    # step 2 - construct a cuda graph with `n_repeat` unrolled function calls to minimize
    # host overhead
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(n_repeat):
            if grad_to_none is not None:
                for x in grad_to_none:
                    x.grad = None
            fn()
    torch.cuda.synchronize()
    # measure time and return
    ret = []
    n_retries = 10
    for _ in range(n_retries):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        g.replay()
        end_event.record()
        torch.cuda.synchronize()
        ret += [start_event.elapsed_time(end_event) / n_repeat]
    return _summarize_statistics(torch.tensor(ret), quantiles, return_mode)


def cache_dir() -> str:
    cache_dir = os.environ.get("TORCHINDUCTOR_CACHE_DIR")
    if cache_dir is None:
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = cache_dir = default_cache_dir()
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def default_cache_dir():
    sanitized_username = re.sub(r'[\\/:*?"<>|]', "_", getpass.getuser())
    return os.path.join(
        tempfile.gettempdir(),
        "torchinductor_" + sanitized_username,
    )


try:
    import colorama

    HAS_COLORAMA = True
except ModuleNotFoundError:
    HAS_COLORAMA = False
    colorama = None  # type: ignore[assignment]


def _color_text(msg, color):
    if not HAS_COLORAMA:
        return msg

    return getattr(colorama.Fore, color.upper()) + msg + colorama.Fore.RESET


def green_text(msg):
    return _color_text(msg, "green")


def yellow_text(msg):
    return _color_text(msg, "yellow")


def red_text(msg):
    return _color_text(msg, "red")


def blue_text(msg):
    return _color_text(msg, "blue")


def get_first_attr(obj, *attrs):
    """
    Return the first available attribute or throw an exception if none is present.
    """
    for attr in attrs:
        if hasattr(obj, attr):
            return getattr(obj, attr)

    raise AssertionError(f"{obj} does not has any of the attributes: {attrs}")


try:
    dynamo_timed = torch._dynamo.utils.dynamo_timed
except AttributeError:  # Compile workers only have a mock version of torch

    def dynamo_timed(original_function=None, phase_name=None, fwd_only=True):
        if original_function:
            return original_function
        return dynamo_timed
