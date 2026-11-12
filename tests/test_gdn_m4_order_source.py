"""Portable scope and unchanged-kernel checks; GPU qualification is separate."""
import ast
import copy
import hashlib
from pathlib import Path
from types import SimpleNamespace
import unittest

SOURCE = Path(__file__).parents[1] / 'python/mxfp6/gdn.py'


def canonical_ast_sha(node):
    """Ignore only Python 3.12's empty type-parameter schema extension."""
    node = copy.deepcopy(node)
    for child in ast.walk(node):
        if getattr(child, 'type_params', None) == []:
            child._fields = tuple(f for f in child._fields if f != 'type_params')
    return hashlib.sha256(ast.dump(node).encode()).hexdigest()


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
        self.assertEqual(canonical_ast_sha(kernel),
                         '6332e52cd9ad1b88d521282f8d7505fde9bc5bc69fe54dfb2d95c7c9f411c221')

    def test_local4_helpers_match_qualified_prototype_ast(self):
        expected = {
            '_rn_mul': '4dda3bad9e40583e7fddfe2a44f936d7c5e1a8522749e1cec86a3cd249abe1ff',
            '_rn_fma': '3da2bf47519aeb61342baa223e1726c15666e915f6dbc5ba40a40b18041ae7fa',
            '_m4_square_sum': '09521dfba0b137722f0664f41a3524b45807b09d6927aa7f9bb55cdef6ec42a6',
        }
        for name, digest in expected.items():
            with self.subTest(helper=name):
                self.assertEqual(canonical_ast_sha(self.function(name)), digest)

    def test_only_empty_schema_field_is_ignored(self):
        plain = ast.parse('def f(): return 1').body[0]
        extended = copy.deepcopy(plain)
        if 'type_params' not in extended._fields:
            extended._fields += ('type_params',)
        extended.type_params = []
        self.assertEqual(canonical_ast_sha(plain), canonical_ast_sha(extended))
        extended.type_params = [ast.Name(id='T', ctx=ast.Load())]
        self.assertNotEqual(canonical_ast_sha(plain), canonical_ast_sha(extended))
        changed = ast.parse('def f(): return 2').body[0]
        self.assertNotEqual(canonical_ast_sha(plain), canonical_ast_sha(changed))

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
