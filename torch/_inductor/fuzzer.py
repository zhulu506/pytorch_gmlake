import itertools
import random
import string
import traceback
from enum import Enum
from functools import partial, wraps
from typing import (
    Any,
    Callable,
    Dict,
    get_args,
    get_origin,
    List,
    Literal,
    Optional,
    Tuple,
    Type,
    Union,
)

import torch
from torch._inductor.custom_graph_pass import CustomGraphPass
from torch._inductor.scheduler import BaseSchedulerNode
from torch.utils._config_module import _ConfigEntry, ConfigModule


"""
# Known failures on inductor config:
cpp_wrapper, triton_debug_sync_graph
cpp_wrapper, triton_debug_sync_kernel
cpp_wrapper, disable_cpp_codegen
combo_kernels, benchmark_combo_kernel, profile_bandwidth, profile_bandwidth_regex

# Example usage:
import torch._inductor.config as cfg

fuzzer = ConfigFuzzer(cfg, create_simple_test_model_gpu, seed=2)

# Test every pair of configs:
results = fuzzer.fuzz_n_tuple(n, max_combinations=10000000)

visualize_results(n, results)

# Test random configs with bisection:
ret = fuzzer.fuzz_random_with_bisect(num_attempts=10)

# reproduce a failing config
fuzzer.reproduce([{"triton.autotune_pointwise": ..., "coordinate_descent_tuning": ...}])
"""


def is_optional_type(type_hint) -> bool:  # type: ignore[no-untyped-def]
    origin = get_origin(type_hint)

    if origin is Union:
        args = get_args(type_hint)
        return type(None) in args

    return False


# callable types are messed up
def is_callable_type(type_hint) -> bool:  # type: ignore[no-untyped-def]
    return type_hint.__name__ == "Callable"


def is_type(type_hint, comp_type) -> bool:  # type: ignore[no-untyped-def]
    return type_hint is comp_type or get_origin(type_hint) is comp_type


class DummyPass(CustomGraphPass):
    def __call__(self, graph: torch.fx.graph.Graph) -> None:
        """
        Implementation of the custom pass.
        """
        return None

    def uuid(self) -> Optional[Any]:
        """
        Return an ID to uniquely identify your custom pass implementation. Return None
        to skip inductor code caching entirely.
        """
        return None


TYPE_EXEMPLARS: dict[str, Any] = {
    CustomGraphPass.__name__: DummyPass(),
    torch.fx.graph.Graph.__name__: torch.fx.graph.Graph(),
    BaseSchedulerNode.__name__: BaseSchedulerNode(None),  # type: ignore[arg-type]
}


class Status(Enum):
    SKIPPED = "skipped"
    PASSED = "passed"
    FAILED_RUN_EXCEPTION = "failed_run_exception"
    FAILED_RUN_RETURN = "failed_run_return"
    FAILED_COMPILE = "failed_compile"

    def failing(self) -> bool:
        return self == Status.FAILED_RUN or self == Status.FAILED_COMPILE


class SamplingMethod(Enum):
    """
    This class handles sampling values of a type assign to configs.
    """

    TOGGLE = "TOGGLE"  # toggle to the opposite value
    RANDOM = "RANDOM"  # randomly choose an option

    @staticmethod
    def _generate_value_for_type(
        random_sample: bool, type_hint: Type[Any], default: Any
    ) -> Any:
        """this setting will use randomness too, but if there's a sensible 'toggle', it will use that"""
        if type_hint == bool:
            return random.choice([True, False]) if random_sample else not default
        elif type_hint == int:
            # NOTE initially tried to use negation of the value, but it doesn't work because most types are ints
            # when they should be natural numbers + zero. Python types to cover these values aren't super convenient.
            return random.randint(0, 1000)
        elif type_hint == float:
            return random.uniform(0, 1000)
        elif type_hint == str:
            characters = string.ascii_letters + string.digits + string.punctuation
            return "".join(
                random.choice(characters) for _ in range(random.randint(1, 20))
            )
        elif is_type(type_hint, list):
            elem_type = getattr(
                type_hint,
                "__args__",
                [type(default[0])] if len(default) else [type(None)],
            )[0]
            new_default = default[0] if len(default) > 0 else None
            return [
                SamplingMethod._generate_value_for_type(
                    random_sample, elem_type, new_default
                )
                for _ in range(random.randint(1, 3))
            ]
        elif is_type(type_hint, set):
            indexable = list(default)
            elem_type = getattr(
                type_hint,
                "__args__",
                [type(indexable[0])] if len(default) else [type(None)],
            )[0]
            new_default = indexable[0] if len(default) > 0 else None
            return {
                SamplingMethod._generate_value_for_type(
                    random_sample, elem_type, new_default
                )
                for _ in range(random.randint(1, 3))
            }
        elif is_type(type_hint, dict):
            key_type, value_type = getattr(
                type_hint,
                "__args__",
                map(type, next(iter(default.items())))
                if len(default)
                else (type(None), type(None)),
            )
            items = list(default.items())
            if len(items) > 0:
                default_key, default_val = items[0]
            else:
                default_key, default_val = None, None
            return {
                SamplingMethod._generate_value_for_type(
                    random_sample, key_type, default_key
                ): SamplingMethod._generate_value_for_type(
                    random_sample, value_type, default_val
                )
                for _ in range(random.randint(0, 3))
            }
        elif is_type(type_hint, Union):
            # do whatever is not the type of default
            try:
                assert len(type_hint.__args__) > 1
            except AttributeError as err:
                raise ValueError("Union type with no args") from err
            if random_sample:
                new_type = random.choice(type_hint.__args__)
            else:
                new_type = random.choice(
                    [t for t in type_hint.__args__ if t != type(default)]
                )
            try:
                new_default = new_type()
            except Exception:  # noqa: E722
                # if default constructor doesn't work, try None
                new_default = None

            return SamplingMethod._generate_value_for_type(
                random_sample, new_type, new_default
            )
        elif is_type(type_hint, tuple):
            args = getattr(
                type_hint,
                "__args__",
                tuple(map(type, default)),
            )
            zipped = zip(args, default)
            return tuple(
                map(  # noqa: C417
                    lambda x: SamplingMethod._generate_value_for_type(
                        random_sample, x[0], x[1]
                    ),
                    zipped,
                )
            )
        elif is_type(type_hint, Literal):
            try:
                if random_sample:
                    return random.choice(type_hint.__args__)
                else:
                    return random.choice(
                        [t for t in type_hint.__args__ if t != default]
                    )
            except AttributeError as err:
                raise ValueError("Literal type with no args") from err
        elif is_optional_type(type_hint):
            try:
                elem_type = type_hint.__args__[0]
            except AttributeError as err:
                raise ValueError("Optional type with no args") from err
            if random_sample:
                return random.choice(
                    [
                        None,
                        SamplingMethod._generate_value_for_type(
                            random_sample, elem_type, default
                        ),
                    ]
                )
            else:
                if default is None:
                    return SamplingMethod._generate_value_for_type(
                        random_sample, elem_type, None
                    )
                else:
                    return None
        elif type_hint is type(None):
            return None
        elif is_callable_type(type_hint):
            try:
                input_args, return_type = (
                    list(type_hint.__args__)[:-1],
                    list(type_hint.__args__)[-1],
                )
            except AttributeError as err:
                raise ValueError("Callable type with no args") from err

            @wraps(lambda *args, **kwargs: None)
            def dummy_function(*args, **kwargs):  # type: ignore[no-untyped-def]
                return SamplingMethod._generate_value_for_type(
                    random_sample, return_type, None
                )

            return dummy_function
        elif type_hint.__name__ in TYPE_EXEMPLARS:
            return TYPE_EXEMPLARS[type_hint.__name__]
        elif type_hint == Any:
            return 1 if not default == 1 else 2
        else:
            raise ValueError(f"Unable to process type {type_hint}. PRs welcome :)")

    @staticmethod
    def dispatch(sm: "SamplingMethod") -> Callable[[Type[Any], Any], Any]:
        if sm == SamplingMethod.RANDOM:
            return partial(SamplingMethod._generate_value_for_type, True)
        elif sm == SamplingMethod.TOGGLE:
            return partial(SamplingMethod._generate_value_for_type, False)
        else:
            raise ValueError(f"malformed sampling method: {sm}")


class Default:
    pass


DEFAULT = Default()

ComboType = Tuple[str, ...]
ResultType = Dict[ComboType, Status]
ConfigType = Dict[str, Any]
FactoryOutputType = Callable[[], bool | Tuple[Any]]
FactoryType = Callable[[], FactoryOutputType]


class ConfigFuzzer:
    sample: Callable[[Type[Any], Any], Any]

    def __init__(
        self,
        config_module: ConfigModule,
        test_model_fn_factory: FactoryType,
        seed: int,
        default: Optional[ConfigType] = None,
        sm: SamplingMethod = SamplingMethod.TOGGLE,
    ):
        """
        Args:
            config_module: The module containing the configs to fuzz
            test_model_fn_factory: Function that returns a test model, which runs and returns True if successful, or the outputs if they should be compared with eager
            seed: Randomness seed.
            default: Default values for the config. Inductor has preset based on know failures.
            sm: How type value samples are generated, default TOGGLE.
        """
        self.seed = seed
        self.config_module = config_module
        self.test_model_fn_factory = test_model_fn_factory
        self.fields: Dict[str, _ConfigEntry] = self.config_module._config
        self.sample = SamplingMethod.dispatch(sm)

        if default is None:
            if self.config_module.__name__ == "torch._inductor.config":
                self.default = {
                    "force_disable_caches": True,
                    "cpp.cxx": DEFAULT,
                    "TYPE_CHECKING": DEFAULT,
                    "max_autotune_pointwise": DEFAULT,
                    "max_autotune_gemm": DEFAULT,
                    "max_autotune_gemm_backends": DEFAULT,
                    "max_autotune_conv_backends": DEFAULT,
                    "max_autotune_gemm_search_space": DEFAULT,
                    "max_autotune_subproc_result_timeout_seconds": DEFAULT,
                    "max_autotune_subproc_graceful_timeout_seconds": DEFAULT,
                    "max_autotune_subproc_terminate_timeout_seconds": DEFAULT,
                    "autoheuristic_collect": DEFAULT,
                    "autoheuristic_use": DEFAULT,
                    "aot_inductor.presets": DEFAULT,
                    "cuda.arch": DEFAULT,
                    "cuda.version": DEFAULT,
                    "cuda.cutlass_dir": DEFAULT,
                    "cuda.cuda_cxx": DEFAULT,
                    "rocm.arch": DEFAULT,
                    "rocm.ck_supported_arch": DEFAULT,
                    "rocm.ck_dir": DEFAULT,
                    "rocm.rocm_home": DEFAULT,
                    "check_stack_no_cycles_TESTING_ONLY": DEFAULT,
                    "reorder_for_compute_comm_overlap": DEFAULT,
                    "enabled_metric_tables": DEFAULT,  # disabled due to lack of typing
                    "triton.debug_sync_graph": DEFAULT,  # disabled due to known failure
                    "triton.debug_sync_kernel": DEFAULT,  # disabled due to known failure
                    "profile_bandwidth_regex": DEFAULT,  # disabled due to know failure
                    "disable_cpp_codegen": DEFAULT,  # disabled due to know failure
                }
            else:
                raise ValueError("No default passed to ConfigFuzzer.")
        else:
            self.default = default

    def __repr__(self) -> str:
        return (
            f"ConfigFuzzer(config_module={self.config_module}, "
            f"test_model_fn_factor={self.test_model_fn_factory}, seed={self.seed}, default={self.default})"
        )

    def _set_config(self, field_name: str, value: Any) -> None:
        """Set a config value in the module."""
        setattr(self.config_module, field_name, value)

    def _reset_configs(self) -> None:
        """Reset all configs to their default values."""
        for field_name, field_obj in self.fields.items():
            self._set_config(field_name, field_obj.default)

    def _set_status(
        self, results: ResultType, combo: ComboType, status: Status
    ) -> None:
        combo = tuple(sorted(combo))
        results[combo] = status

    def _lookup_status(self, results: ResultType, combo: ComboType) -> Optional[Status]:
        combo = tuple(sorted(combo))
        return results[combo] if combo in results else None

    def new_config(self) -> ConfigType:
        """creates a new config from the default"""
        ret = {
            name: val if val != DEFAULT else self.fields[name].default
            for name, val in self.default.items()
        }
        return ret

    def _combo_run_common(self, results: ResultType, combo: ComboType) -> None:
        print(combo)
        if self._lookup_status(results, combo):
            # we already processed this config
            return

        config = self.new_config()

        skip = False
        for field_name in combo:
            if field_name in config:
                # don't break here because we need to build the config dict
                skip = True
            if field_name.startswith("_"):
                skip = True
            field = self.fields[field_name]
            value = self.sample(field.value_type, field.default)
            config[field_name] = value
        if skip:
            self._set_status(results, combo, Status.SKIPPED)
            return

        self.test_config(results, config)

    def reproduce(self, configs: List[ConfigType]) -> ResultType:
        """entrypoint to reproduce any failure"""
        results: ResultType = {}
        for conf in configs:
            print(f"Starting repro of {conf}")
            new_config = self.new_config()
            new_config.update(conf)
            self.test_config(results, new_config)
        return results

    def fuzz_n_tuple(self, n: int, max_combinations: int = 1000) -> ResultType:
        """
        Test every combination of n configs.

        returns a dict of this shape: {(config-1, config-2... config-n): status}
        """
        results: ResultType = {}
        print(f"Starting {n}-tuple testing with seed {self.seed}")
        random.seed(self.seed)

        for combo in itertools.combinations(self.fields, n):
            self._combo_run_common(results, combo)
            max_combinations -= 1
            if max_combinations <= 0:
                print("Reached maximum combinations limit")
                break

        return results

    def test_config(self, results: ResultType, config: ConfigType) -> Status:
        """
        Tests a config
        """
        print(f"Testing config {config}")
        config_tuple = tuple(config.keys())
        if ret := self._lookup_status(results, config_tuple):
            return ret
        torch._dynamo.reset()
        test_model_fn = self.test_model_fn_factory()
        def set_config():
            for name, value in config.items():
                self._set_config(name, value)

        def compile_with_options(test_fn):
            if self.config_module.__name__ == "torch._inductor.config":
                return torch.compile(options=config)(test_model_fn)
            self._reset_configs()
            set_config()
            comp = torch.compile()(test_model_fn)
        def run_eager(test_fn):
            if self.config_module.__name__ == "torch._inductor.config":
                # we didn't set config earlier for compile in inductor
                set_config()
            return test_fn()
        def print_config():
            for field, value in config.items():
                print(f"{field} = {value}")
        def handle_return(message, return_status, print_traceback):
            print(f"{message} with config combination:")
            print_config(config)
            if print_traceback:
                traceback.print_exc()
            self._set_status(results, config_tuple, return_status)
            return return_status

        # try compilation
        try:
            comp = compile_with_options(test_model_fn)
        except Exception:  # noqa: E722
            return handle_return("Exception compiling", Status.FAILED_COMPILE, True)

        # try running compiled
        try:
            success = comp()
        except Exception:  # noqa: E722
            return handle_return("Exception running", Status.FAILED_RUN_EXCEPTION, True)

        # bool return value means don't compare with eager
        if type(success) is bool:
            if not success:
                return handle_return("Failure returned bool", Status.FAILED_RUN_RETURN, False)
            else:
                ret = Status.PASSED
                self._set_status(results, config_tuple, ret)
                return ret
        # try running in eager
        elif type(success) is tuple:
            try:
                eager_results = test_model_fn()
            except Exception:  # noqa: E722
                return handle_return("Eager exception", Status.FAILED_RUN_EXCEPTION, True)
            for er, cr in zip(eager_results, success):
                if not torch.isclose(er, cr):

        else:
            raise ValueError(f"Unable to process return type of test function: {type(success)}")

    def fuzz_with_bisect(
        self, num_attempts: int = 100, p: float = 0.5
    ) -> List[ConfigType]:
        """
        Test configs and bisect to minimal failing configuration.
        """
        print(f"Starting random testing with bisection, seed {self.seed}, and p {p}")
        random.seed(self.seed)
        self._reset_configs()
        results: ResultType = {}
        ret: List[ConfigType] = []

        for attempt in range(num_attempts):
            print(f"Random attempt {attempt + 1}/{num_attempts}")

            config = self.new_config()

            for field_name, config_entry in self.fields.items():
                if (
                    field_name not in config
                    and not field_name.startswith("_")
                    and random.random() < p
                ):
                    value = self.sample(config_entry.value_type, config_entry.default)
                    config[field_name] = value

            status = self.test_config(results, config)
            if status not in {Status.PASSED, Status.SKIPPED}:
                if minimal_failing_config := self._bisect_failing_config(
                    results, config
                ):
                    print(f"Minimum failing config: {minimal_failing_config}")
                    ret.append(minimal_failing_config)

        return ret

    def _bisect_failing_config(
        self, results: ResultType, failing_config: ConfigType
    ) -> Optional[ConfigType]:
        return self._bisect_failing_config_helper(results, list(failing_config.items()))

    def _bisect_failing_config_helper(
        self, results: ResultType, failing_config: List[Tuple[str, Any]]
    ) -> Optional[ConfigType]:
        """
        Bisect a failing configuration to find minimal set of configs that cause failure.

        Splits it into halves, then fourths, then tries dropping configs one-by-one.
        """
        print(f"bisecting config: {failing_config}")

        if not failing_config:
            return None

        def test(x: List[Tuple[str, Any]]) -> Status:
            d = dict(x)
            result = self.test_config(results, d)
            return result

        if len(failing_config) <= 1:
            return dict(failing_config) if test(failing_config).failing() else None

        # Shuffling helps the worst case
        random.shuffle(failing_config)

        mid = len(failing_config) // 2
        first_half = failing_config[:mid]
        second_half = failing_config[mid:]
        if test(first_half).failing():
            return self._bisect_failing_config_helper(results, first_half)
        if test(second_half).failing():
            return self._bisect_failing_config_helper(results, second_half)

        if len(failing_config) >= 8:
            low = len(failing_config) // 4
            high = mid + low
            quart1 = failing_config[low:]
            if test(quart1).failing():
                return self._bisect_failing_config_helper(results, quart1)
            quart2 = failing_config[:low] + second_half
            if test(quart2).failing():
                return self._bisect_failing_config_helper(results, quart2)
            quart3 = first_half + failing_config[:high]
            if test(quart3).failing():
                return self._bisect_failing_config_helper(results, quart3)
            quart4 = failing_config[high:]
            if test(quart4).failing():
                return self._bisect_failing_config_helper(results, quart4)
        # try dropping one value at a time
        for i in range(len(failing_config)):
            new_list = [x for j, x in enumerate(failing_config) if j != i]
            if test(new_list).failing():
                return self._bisect_failing_config_helper(results, new_list)
        # we have the minimal set
        return dict(failing_config)


def create_simple_test_model_cpu() -> FactoryOutputType:
    def test_fn() -> bool:
        model = torch.nn.Sequential(
            torch.nn.Linear(10, 10), torch.nn.ReLU(), torch.nn.Linear(10, 1)
        )

        x = torch.randn(32, 10)
        y = model(x)
        return True

    return test_fn


def create_simple_test_model_gpu() -> FactoryOutputType:
    batch_size = 32
    seq_length = 50
    hidden_size = 768

    def test_fn() -> bool:
        inp = torch.randn(batch_size, seq_length, hidden_size, device="cuda")
        weight = torch.randn(hidden_size, hidden_size, device="cuda")
        matmul_output = inp @ weight
        final_output = torch.nn.LayerNorm(hidden_size, device="cuda")(matmul_output)
        return True

    return test_fn


def visualize_results(
    n: int, status: ResultType, filename: str = "results.html"
) -> None:
    # TODO Support more dimensions
    assert n == 2
    assert len(status) > 0

    # Create a dictionary for quick lookup of status
    input_set = set({})
    for key in status.keys():
        input_set.add(key[0])
        input_set.add(key[1])
    input_list = sorted(input_set)

    # Start the HTML content
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title> Fuzzer Visualization</title>
        <style>
            table {
                border-collapse: collapse;
                width: 50%;
                margin: 20px auto;
            }
            th, td {
                border: 1px solid #ddd;
                padding: 8px;
                text-align: center;
            }
            th {
                background-color: #f2f2f2;
            }
            .skipped {
                background-color: yellow;
            }
            .passed {
                background-color: green;
                color: white;
            }
            .failed {
                background-color: red;
                color: white;
            }
        </style>
    </head>
    <body>
        <h2 style="text-align: center;">Fuzzer Visualization</h2>
        <table>
        <thead>
    """

    html_content += "<tr><th>\\</th>"
    for i, col_name in enumerate(input_list):
        col = "<br>".join(col_name)
        html_content += f"<th>{col}</th>"
    html_content += "</tr></thead><tbody>"

    # Add table rows
    for i, row_name in enumerate(input_list):
        html_content += f"<tr><th>{row_name}</th>"
        for j, col_name in enumerate(input_list):
            # Determine the status class for the cell
            status_class = ""
            status_val = ""
            if (row_name, col_name) in status:
                status_enum = status[(row_name, col_name)]
                if status_enum == Status.SKIPPED:
                    status_class = "skipped"
                    status_val = "-"
                elif status_enum == Status.PASSED:
                    status_class = "passed"
                    status_val = "O"
                elif status_enum == Status.FAILED_RUN_EXCEPTION:
                    status_class = "failed"
                    status_val = "E"
                elif status_enum == Status.FAILED_RUN_RETURN:
                    status_class = "failed"
                    status_val = "R"
                elif status_enum == Status.FAILED_COMPILE:
                    status_class = "failed"
                    status_val = "C"

            html_content += f'<td class="{status_class}">{status_val}</td>'
        html_content += "</tr>"

    html_content += """
        </tbody>
        </table>
    </body>
    </html>
    """

    with open(filename, "w") as file:
        file.write(html_content)

    print(f"HTML file '{filename}' has been generated.")
