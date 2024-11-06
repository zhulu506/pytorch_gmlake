# mypy: allow-untyped-defs
import contextlib
from typing import Any, Callable, Dict, List, Optional, Tuple
from unittest.mock import patch

import sympy

from .. import ir, lowering as L
from ..select_algorithm import PartialRender
from ..virtualized import V
from .cpp_gemm_template import (
    CppGemmTemplate,
    GEMM_TEMPLATE,
    get_padded_n,
    MICROKERNEL_DEF,
)
from .cpp_micro_gemm import LayoutType
from .cpp_template_kernel import CppTemplateKernel
from .cpp_utils import DTYPE_TO_CPP, GemmBlocking


GEMM_SINGLE_THREAD_MM_STUB = r"""
{{kernel.def_kernel(
    inputs={"X": X, "W": W},
    outputs={"Y": Y},
    aliases=aliases,
    function_name="single_thread_mm",
    symbols=[b_index],
    placeholder="<SINGLE_THREAD_DEF_MM_FOR_BMM>")}}"""

GEMM_THREADED_MM_STUB = r"""
{{kernel.def_kernel(
    inputs={"X": X, "W": W},
    outputs={"Y": Y},
    aliases=aliases,
    function_name="threaded_mm",
    symbols=[b_index],
    placeholder="<THREADED_MM_DEF_FOR_BMM>")}}"""

BMM_WRAPPER = r"""
extern "C"
{{kernel.def_kernel(inputs={"X": BX, "W": BW}, outputs={"Y": BY}, aliases=aliases)}}
{
    const int64_t B = {{kernel.size(BY_2d, 0)}};
    {%- if num_threads > 1 %}
    constexpr int64_t num_threads = {{num_threads}};
    int64_t B_single_thread_block = (B / num_threads) * num_threads;

    #pragma omp parallel for num_threads({{num_threads}})
    {%- else %}
    int64_t B_single_thread_block = B;
    {%- endif %}
    for (int64_t b_start = 0; b_start < B_single_thread_block; ++b_start) {
        {{template.get_gemm_function_call(
            kernel,
            "single_thread_mm",
            "<SINGLE_THREAD_CALL_FOR_BMM>",
            b_index="b_start",
        )}}
    }
    for (int64_t b_start = B_single_thread_block; b_start < B; ++b_start) {
        {{template.get_gemm_function_call(
            kernel,
            "threaded_mm",
            "<THREADED_MM_CALL_FOR_BMM>",
            b_index="b_start",
        )}}
    }
}
"""


class CppBmmTemplate(CppGemmTemplate):
    def __init__(
        self,
        input_nodes,
        layout: ir.Layout,
        num_threads: int,
        register_blocking: GemmBlocking,
        beta=1,
        alpha=1,
        has_bias=False,
        epilogue_creator: Optional[Callable[[ir.Buffer], ir.Pointwise]] = None,
        name="bmm",
    ):
        super().__init__(
            input_nodes,
            layout,
            num_threads,
            register_blocking,
            beta=beta,
            alpha=alpha,
            has_bias=has_bias,
            epilogue_creator=epilogue_creator,
            name=name,
        )
        # Value may be changed after micro_gemm is instantiated if using VNNI layout
        self.should_block_weights = False
        self.b_index = sympy.Symbol("s_b_index", integer=True, nonnegative=True)

    @staticmethod
    def get_padded_size(n, block_n, k, block_weight):
        padded_n = get_padded_n(n, block_n)
        if block_weight:
            new_size = [-1, padded_n // block_n, k, block_n]
        else:
            new_size = [-1, k, padded_n]
        return new_size, padded_n

    @staticmethod
    def block_weight_irnode(W, new_size, padding):
        assert isinstance(W, ir.IRNode)
        if not isinstance(W, ir.TensorBox):
            W = ir.TensorBox(W)
        permuted_size = list(new_size)
        permuted_size[-2], permuted_size[-3] = permuted_size[-3], permuted_size[-2]
        blocked_w = L.constant_pad_nd(W, (0, padding))
        blocked_w = L.permute(
            L.view(blocked_w, permuted_size),
            [0, 2, 1, 3],
        )
        return blocked_w

    @staticmethod
    def pack_vnni_weight_irnode(W, micro_gemm, new_size):
        # new_size = [-1, padded_n // block_n, k, block_n]
        # TODO: (frost-intel): For non-constant packed weights, do this VNNI conversion at microkernel level
        k = new_size[-2]
        if not isinstance(W, ir.TensorBox):
            W = ir.TensorBox(W)
        if micro_gemm.get_b_layout() != LayoutType.NORMAL:
            vnni_size = 4 if micro_gemm.get_b_layout() == LayoutType.VNNI4 else 2
            vnni_view_size = list(new_size)
            vnni_view_size[-2] = k // vnni_size
            vnni_view_size.insert(-1, vnni_size)
            W = L.view(
                L.permute(L.view(W, vnni_view_size), [0, 1, 2, 4, 3]),
                new_size,
            )
            W = CppBmmTemplate.realize_permuted_irnode(W)
        return W

    def get_gemm_function_call(
        self,
        kernel: CppTemplateKernel,
        function_name: str,
        placeholder: str,
        b_index: int,
    ) -> str:
        """
        Similar to 'def_kernel' in cpp_template_kernel, but instead of generating a function definition,
        generate a function call for the GEMM kernel.
        Args:
            placeholder: The string to replace the function call with
            b_index: The index for slicing the 3D batch tensors
            nodes: The 3D batch tensors
        """

        def hook():
            call_args, buffer_names, _, _ = kernel.args.python_argdefs()
            for i, buf in enumerate(buffer_names):
                if buf == self.b_index:
                    call_args[i] = b_index
            call = f"{function_name}({', '.join(call_args)});"
            return call

        assert placeholder not in kernel.render_hooks
        kernel.render_hooks[placeholder] = hook
        return placeholder

    def get_default_reindexers(self, epilogue_nodes):
        def reindexer(args):
            # if epilogue nodes exist, they have 3D ranges but args are 2D, so add 0 index
            if len(epilogue_nodes) == 0:
                return args
            return [self.b_index] + args

        return [reindexer]

    def get_options(
        self,
        kernel: CppTemplateKernel,
        template_buffer_node: Optional[ir.CppTemplateBuffer] = None,
        flag_template_buffer_has_other_users: Optional[bool] = None,
        epilogue_nodes: Optional[List[ir.IRNode]] = None,
        **kwargs,
    ) -> Tuple[Dict[str, Any], List[ir.Buffer]]:
        options, fake_buffers = super().get_options(
            kernel=kernel,
            template_buffer_node=template_buffer_node,
            flag_template_buffer_has_other_users=flag_template_buffer_has_other_users,
            epilogue_nodes=epilogue_nodes,
            **kwargs,
        )
        if options["micro_gemm"].get_b_layout() != LayoutType.NORMAL:
            self.should_block_weights = True

        BX, BW, BY = options["X"], options["W"], options["Y"]
        options["BX"], options["BW"], options["BY"] = BX, BW, BY
        options["BY_2d"] = options["Y_2d"]
        for kword in ["X", "W", "Y", "GemmOut", "Y_2d"]:
            options[kword] = kernel.select(options[kword], 0, self.b_index)
        for kword in ["X", "W", "Y"]:
            options[kword + "_dtype"] = DTYPE_TO_CPP[options[kword].dtype]
        options["b_index"] = self.b_index

        return options, fake_buffers

    def render(  # type: ignore[override, return]
        self,
        kernel: CppTemplateKernel,
        template_buffer_node: Optional[ir.CppTemplateBuffer] = None,
        flag_template_buffer_has_other_users: Optional[bool] = None,
        epilogue_nodes: Optional[List[ir.IRNode]] = None,
        **kwargs,
    ) -> str:
        options, fake_buffers = self.get_options(
            kernel=kernel,
            template_buffer_node=template_buffer_node,
            flag_template_buffer_has_other_users=flag_template_buffer_has_other_users,
            epilogue_nodes=epilogue_nodes,
            **kwargs,
        )

        with contextlib.ExitStack() as stack:
            for buf in fake_buffers:
                stack.enter_context(
                    patch.object(V.graph, "get_dtype", self._fake_get_dtype(buf))
                )
            result = self._template_from_string(MICROKERNEL_DEF).render(**options)
            result += self._template_from_string(
                GEMM_THREADED_MM_STUB + GEMM_TEMPLATE
            ).render(**options)
            result += self._template_from_string(
                GEMM_SINGLE_THREAD_MM_STUB + GEMM_TEMPLATE
            ).render(**{**options, "num_threads": 1})
            result += self._template_from_string(BMM_WRAPPER).render(**options)

            # Finalize the function definitions for the gemm routines
            sub_mm_hooks = {
                name: hook
                for name, hook in kernel.render_hooks.items()
                if "FOR_BMM" in name
            }
            result = PartialRender(result, sub_mm_hooks).finalize_all()
            for name in sub_mm_hooks:
                del kernel.render_hooks[name]
            del kernel.args.sizevars[options["b_index"]]
            return result
