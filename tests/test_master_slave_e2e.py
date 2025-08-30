# tests/test_master_slave_e2e.py
import unittest, types, pathlib, textwrap, sys
import multiprocessing as mp

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

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
    src = textwrap.dedent(fn_src)
    return {
        'type': 'function',
        'name': fn_name,
        'source': src,
        'lineno': 1,
        'end_lineno': src.count("\n") + 1,
    }

@unittest.skipIf(mp.get_start_method(allow_none=True) == "spawn",
                 "Os testes gerados via exec com Process podem falhar sob 'spawn'; use Linux/fork.")
class TestMasterSlaveE2E(unittest.TestCase):
    def test_ms_simple(self):
        seq_src = """
def process_tasks(tasks):
    out = []
    for t in tasks:
        out.append((t*2 + 10))
    return out
"""
        ns_seq = _load_ns_from_code(seq_src)
        func_name, _ = _pick_first_function(ns_seq)
        bottleneck = _build_bottleneck(seq_src, func_name)

        _orig, transformed, ctx = generate_parallel_code(
            bottleneck, pattern="master_slave", partitioning_strategy=["default"]
        )
        ns_par = {}
        exec(compile(transformed, "<gen>", "exec"), ns_par, ns_par)
        par_fn = ns_par[f"{func_name}_parallel"]

        data = [1, 2, 3, 4, 5]
        self.assertEqual(ns_seq[func_name](data), par_fn(data))

    def test_ms_with_predicate(self):
        seq_src = """
def filter_and_square(nums):
    out = []
    for x in nums:
        if x % 2 == 0:
            out.append(x*x)
    return out
"""
        ns_seq = _load_ns_from_code(seq_src)
        fn_name, _ = _pick_first_function(ns_seq)
        bottleneck = _build_bottleneck(seq_src, fn_name)

        _o, transformed, ctx = generate_parallel_code(
            bottleneck, pattern="master_slave", partitioning_strategy=["default"]
        )
        ns_par = {}
        exec(compile(transformed, "<gen>", "exec"), ns_par, ns_par)
        par_fn = ns_par[f"{fn_name}_parallel"]

        data = [1,2,3,4,5,6]
        self.assertEqual(ns_seq[fn_name](data), par_fn(data))

    def test_ms_complex_dicts(self):
        seq_src = """
def transform_records(records):
    out = []
    for rec in records:
        if isinstance(rec, dict) and rec.get("enabled", True):
            out.append(rec.get("value", 0) * rec.get("scale", 1) + 5)
    return out
"""
        ns_seq = _load_ns_from_code(seq_src)
        fn_name, _ = _pick_first_function(ns_seq)
        bottleneck = _build_bottleneck(seq_src, fn_name)

        _o, transformed, ctx = generate_parallel_code(
            bottleneck, pattern="master_slave", partitioning_strategy=["default"]
        )
        ns_par = {}
        exec(compile(transformed, "<gen>", "exec"), ns_par, ns_par)
        par_fn = ns_par[f"{fn_name}_parallel"]

        data = [
            {"value": 2, "scale": 3, "enabled": True},   # 11
            {"value": 5,               "enabled": True}, # 10
            42,                                          # filtrado
            {"value": 1, "scale": 4,   "enabled": False},# filtrado
            {"other": 7,               "enabled": True}, # 5
        ]
        self.assertEqual(ns_seq[fn_name](data), par_fn(data))

    def test_ms_empty_input(self):
        seq_src = """
def f(xs):
    out = []
    for x in xs:
        out.append(x+1)
    return out
"""
        ns_seq = _load_ns_from_code(seq_src)
        fn_name, _ = _pick_first_function(ns_seq)
        bottleneck = _build_bottleneck(seq_src, fn_name)

        _o, transformed, ctx = generate_parallel_code(
            bottleneck, pattern="master_slave", partitioning_strategy=["default"]
        )
        ns_par = {}
        exec(compile(transformed, "<gen>", "exec"), ns_par, ns_par)
        par_fn = ns_par[f"{fn_name}_parallel"]

        self.assertEqual(ns_seq[fn_name]([]), par_fn([]))

    def test_ms_generator_input_and_nslaves(self):
        seq_src = """
def inc(xs):
    out = []
    for x in xs:
        out.append(x+1)
    return out
"""
        ns_seq = _load_ns_from_code(seq_src)
        fn_name, _ = _pick_first_function(ns_seq)
        bottleneck = _build_bottleneck(seq_src, fn_name)

        _o, transformed, ctx = generate_parallel_code(
            bottleneck, pattern="master_slave", partitioning_strategy=["default"]
        )
        ns_par = {}
        exec(compile(transformed, "<gen>", "exec"), ns_par, ns_par)
        par_fn = ns_par[f"{fn_name}_parallel"]

        # entrada como gerador — o template deve fazer snapshot list(...)
        gen = (i for i in range(10))
        self.assertEqual(ns_seq[fn_name](list(range(10))), par_fn(gen, nslaves=2))


if __name__ == "__main__":
    unittest.main()
