"""Portable scope and unchanged-kernel checks; GPU qualification is separate."""
import ast
import copy
import hashlib
from pathlib import Path
from types import SimpleNamespace
import unittest

SOURCE = Path(__file__).parents[1] / 'python/mxfp6/gdn.py'


class TestM4ProducerOrder(unittest.TestCase):
    def setUp(self):
        self.tree = ast.parse(SOURCE.read_text())

    def function(self, name):
        return next(n for n in self.tree.body
                    if isinstance(n, ast.FunctionDef) and n.name == name)

    def test_other_keys_keep_original_complete_kernel_ast(self):
        kernel = copy.deepcopy(self.function('_producer'))
        self.assertEqual(kernel.args.args[-1].arg, 'M4_ORDER')
        self.assertIs(kernel.args.defaults[-1].value, False)
        kernel.args.args.pop()
        kernel.args.defaults.pop()
        index = next(i for i, n in enumerate(kernel.body)
                     if isinstance(n, ast.If) and isinstance(n.test, ast.Name)
                     and n.test.id == 'M4_ORDER')
        kernel.body[index:index + 1] = kernel.body[index].orelse
        self.assertEqual(hashlib.sha256(ast.dump(kernel).encode()).hexdigest(),
                         '0b915e1232a6ae768f26874b1fadbafa467b806174dc50d5ec41945b27539557')

    def test_local4_helpers_match_qualified_prototype_ast(self):
        expected = {
            '_rn_mul': '23697220800a625a0be3182c79c33f450939d1a6c95b3d8b0db407b1febf8104',
            '_rn_fma': 'fdff6e4092f66216015cb8f9e9db4e078b4150f457f57192373b9699f5020fd4',
            '_m4_square_sum': '5550dcb252550144d9d3471c1961c7fd5296a8fe77b42a89ee0859a44d8e7684',
        }
        for name, digest in expected.items():
            with self.subTest(helper=name):
                self.assertEqual(hashlib.sha256(ast.dump(self.function(name)).encode()).hexdigest(), digest)

    def test_real_frontend_selector_is_finite(self):
        value = next(n.value for n in ast.walk(self.function('gated_norm_mxfp8'))
                     if isinstance(n, ast.keyword) and n.arg == 'M4_ORDER')
        expr = compile(ast.Expression(value), str(SOURCE), 'eval')
        bf16 = object()
        base = dict(m=4, r=1, z=SimpleNamespace(stride=lambda axis: 8192),
                    w=SimpleNamespace(dtype=bf16), eps=1.e-6, rounded_out=None,
                    torch=SimpleNamespace(bfloat16=bf16))
        self.assertTrue(eval(expr, base.copy()))
        self.assertTrue(eval(expr, {**base, 'rounded_out': object()}))
        negatives = [('m', v) for v in (1, 2, 8, 16, 24, 32)] + [
            ('r', 2), ('z', SimpleNamespace(stride=lambda axis: 3072)),
            ('w', SimpleNamespace(dtype=object())), ('eps', 1.e-5)]
        for key, value in negatives:
            with self.subTest(key=key, value=value):
                self.assertFalse(eval(expr, {**base, key: value}))


if __name__ == '__main__':
    unittest.main()
