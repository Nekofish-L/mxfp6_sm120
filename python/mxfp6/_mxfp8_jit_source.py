"""Source and dependency identity shared by the optional JIT path and tooling."""
import hashlib
import re

from ._mxfp8_config import kernel_expression


def device_source(root, config, k):
    included = set()

    def inline_helpers(text):
        def replace(match):
            name = match[1]
            if name.startswith(('mxfp8_gemm/', 'mxfp_common/')):
                if name in included:
                    return ''
                included.add(name)
                return inline_helpers((root/'csrc/include'/name).read_text())
            return '#include <' + name + '>'
        return re.sub(r'#include "([^"]+)"', replace, text)

    source = (root/'csrc/include/mxfp8_gemm/kernels'/f"{config['header']}.cuh").read_text()
    # The pinned DeepJIT parser rejects quoted includes. Hash the dependency
    # trees separately, including patched CUTLASS, instead of losing identity.
    return inline_helpers(source) + '\nstatic void instantiate() {\n' + (
        f"  auto ptr = reinterpret_cast<void*>(&{kernel_expression(config, k)});\n"
    ) + '}\n'


def dependency_signature(root):
    digest = hashlib.sha256()
    for relative in ('third_party/cutlass/include', 'third_party/deepjit/include', 'csrc/include'):
        for path in sorted((root/relative).rglob('*')):
            if path.is_file():
                digest.update(str(path.relative_to(root)).encode())
                digest.update(path.read_bytes())
    digest.update((root/'csrc/mxfp8_jit.cpp').read_bytes())
    return digest.hexdigest()
