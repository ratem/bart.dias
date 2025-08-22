# tests/test_master_worker_e2e.py
import unittest, types, pathlib, ast, textwrap
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bdias_parser import BDiasParser
from bdias_pattern_analyzer import BDiasPatternAnalyzer
from bdias_pattern_codegen import generate_parallel_code

def _load_ns_from_code(src: str):
    ns = {}
    exec(compile(src, "<mem>", "exec"), ns, ns)
    return ns

def _pick_first_function(ns):
    for k, v in ns.items():
        if isinstance(v, types.FunctionType):
            return k, v
    raise RuntimeError("Nenhuma função encontrada")

def _build_bottleneck(fn_src: str, fn_name: str):
    # monta um "bottleneck" no formato que generate_parallel_code espera
    # (o seu codegen usa 'source' de uma função isolada)
    # garanta que o src começa com "def fn(...):"
    src = textwrap.dedent(fn_src)
    return {
        'type': 'function',
        'name': fn_name,
        'source': src,
        'lineno': 1,
        'end_lineno': src.count("\n") + 1,
    }

class TestMasterWorkerE2E(unittest.TestCase):
    def test_simple_master_worker(self):
        # Função sequencial com um for que produz 1 saída por item
        seq_src = """
def process_tasks(tasks):
    results = []
    for t in tasks:
        results.append((t*2 + 10))
    return results
"""
        # 1) Detecção
        parser = BDiasParser()
        analyzer = BDiasPatternAnalyzer(parser)
        structured = parser.parse(seq_src)
        identified = analyzer._identify_patterns(structured, seq_src)

        self.assertIn("master_worker", identified)
        self.assertGreater(len(identified["master_worker"]), 0)

        # 2) Codegen
        ns_seq = _load_ns_from_code(seq_src)
        func_name, _ = _pick_first_function(ns_seq)
        bottleneck = _build_bottleneck(seq_src, func_name)

        _orig, transformed, ctx = generate_parallel_code(
            bottleneck, pattern="master_worker", partitioning_strategy=["default"]
        )

        ns_par = {}
        exec(compile(transformed, "<gen>", "exec"), ns_par, ns_par)
        par_fn = ns_par[f"{func_name}_parallel"]

        # 3) Execução: entradas e paridade de saída
        data = [1,2,3,4,5]
        self.assertEqual(ns_seq[func_name](data), par_fn(data))

    def test_master_worker_with_predicate(self):
        # inclui um if de filtro no corpo do for
        seq_src = """
def filter_and_square(nums):
    out = []
    for x in nums:
        if x % 2 == 0:
            out.append(x*x)
    return out
"""
        parser = BDiasParser()
        analyzer = BDiasPatternAnalyzer(parser)
        structured = parser.parse(seq_src)
        identified = analyzer._identify_patterns(structured, seq_src)

        self.assertIn("master_worker", identified)
        self.assertGreater(len(identified["master_worker"]), 0)

        ns_seq = _load_ns_from_code(seq_src)
        fn_name, _ = _pick_first_function(ns_seq)
        bottleneck = _build_bottleneck(seq_src, fn_name)

        _o, transformed, ctx = generate_parallel_code(
            bottleneck, pattern="master_worker", partitioning_strategy=["default"]
        )
        ns_par = {}
        exec(compile(transformed, "<gen>", "exec"), ns_par, ns_par)
        par_fn = ns_par[f"{fn_name}_parallel"]

        data = [1,2,3,4,5,6]
        self.assertEqual(ns_seq[fn_name](data), par_fn(data))

if __name__ == "__main__":
    unittest.main()
