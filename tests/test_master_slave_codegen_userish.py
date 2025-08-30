# tests/test_master_slave_codegen_userish.py
import unittest, types, pathlib, textwrap, sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bdias_parser import BDiasParser
from bdias_pattern_analyzer import BDiasPatternAnalyzer
from bdias_pattern_codegen import generate_parallel_code


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

class TestMasterSlaveCodegenUserish(unittest.TestCase):
    def test_userish_halt_sentinel_and_filter(self):
        """
        Worker com nome diferente, usa sentinela 'HALT' e filtra itens (índices ímpares).
        Sem task_done(). Deve cair no ramo com sentinela.
        """
        user_src = """
# código hipotético vindo do usuário
from queue import Queue

HALT = ("__STOP__",)

def consumer(in_q, out_q):
    while True:
        item = in_q.get()
        if item == HALT:
            break
        i, x = item
        if i % 2 == 1:
            continue
        out_q.put((i, (x or 0) * 2))

if __name__ == "__main__":
    print("eu não devo rodar no codegen")
"""
        # detecção
        parser = BDiasParser()
        analyzer = BDiasPatternAnalyzer(parser)
        structured = parser.parse(user_src)
        identified = analyzer._identify_patterns(structured, user_src)
        self.assertGreater(len(identified.get("master_slave", [])), 0)

        # pega a função do worker
        fns = parser.get_all_functions(user_src)
        fn_name = fns[0]["name"]
        bottleneck = _build_bottleneck(fns[0]["source"], fn_name)

        # codegen
        _o, transformed, ctx = generate_parallel_code(
            bottleneck, pattern="master_slave", partitioning_strategy=["default"]
        )
        ns = {}
        exec(compile(transformed, "<gen>", "exec"), ns, ns)
        par_fn = ns[f"{fn_name}_parallel"]

        data = [5, None, 7, 10, 0, 3]   # só pares: (0,5)->10 ; (2,7) pula ; (3,10) pula ; (4,0)->0
        out = par_fn(data, nslaves=4)
        self.assertEqual(out, [10, 14, 0])

    def test_userish_task_done_join_no_sentinel(self):
        """
        Worker com task_done()/q.join() (sem sentinela). O wrapper deve detectar isso,
        NÃO enviar sentinela, usar threads daemon e sincronizar via in_q.join().
        """
        user_src = """
# código hipotético vindo do usuário
from queue import Queue
from threading import Thread

def handle(q: Queue, out: Queue):
    while True:
        idx, rec = q.get()
        if isinstance(rec, dict) and rec.get("ok", True):
            out.put((idx, rec.get("v", 0) * 3 + 1))
        q.task_done()

if __name__ == "__main__":
    # script do usuário que não deve rodar aqui
    q, out = Queue(), Queue()
    t = Thread(target=handle, args=(q, out), daemon=True)
    t.start()
"""
        parser = BDiasParser()
        analyzer = BDiasPatternAnalyzer(parser)
        structured = parser.parse(user_src)
        identified = analyzer._identify_patterns(structured, user_src)
        self.assertGreater(len(identified.get("master_slave", [])), 0)

        fns = parser.get_all_functions(user_src)
        fn_name = fns[0]["name"]  # 'handle'
        bottleneck = _build_bottleneck(fns[0]["source"], fn_name)

        _o, transformed, ctx = generate_parallel_code(
            bottleneck, pattern="master_slave", partitioning_strategy=["default"]
        )
        ns = {}
        exec(compile(transformed, "<gen>", "exec"), ns, ns)
        par_fn = ns[f"{fn_name}_parallel"]

        data = [
            {"v": 2, "ok": True},   # 2*3+1 = 7
            {"v": 1, "ok": False},  # filtrado
            123,                    # filtrado
            {"ok": True},           # 0*3+1 = 1
        ]
        out = par_fn(data, nslaves=3)
        self.assertEqual(out, [7, 1])

    def test_userish_annotations_alias_and_stop_default(self):
        """
        Worker com anotações 'exóticas' e STOP não definido no snippet injetado.
        O template publica STOP default e envia a mesma instância como sentinela.
        """
        user_src = """
# código hipotético vindo do usuário
from multiprocessing import Queue as MPQ  # usado só na anotação
from queue import Queue

def proc(q: MPQ, out: "Queue"):
    while True:
        item = q.get()
        if item == STOP:  # STOP não estará no snippet; wrapper deve criá-lo
            break
        idx, x = item
        out.put((idx, (x or 0) + 1))

if __name__ == "__main__":
    print("main do usuário; não deve rodar no codegen")
"""
        parser = BDiasParser()
        analyzer = BDiasPatternAnalyzer(parser)
        structured = parser.parse(user_src)
        identified = analyzer._identify_patterns(structured, user_src)
        self.assertGreater(len(identified.get("master_slave", [])), 0)

        fns = parser.get_all_functions(user_src)
        fn_name = fns[0]["name"]  # 'proc'
        bottleneck = _build_bottleneck(fns[0]["source"], fn_name)

        _o, transformed, ctx = generate_parallel_code(
            bottleneck, pattern="master_slave", partitioning_strategy=["default"]
        )
        ns = {}
        exec(compile(transformed, "<gen>", "exec"), ns, ns)
        par_fn = ns[f"{fn_name}_parallel"]

        data = [0, None, 3]
        out = par_fn(data, nslaves=2)
        # esperado: (0->1), (None->1), (3->4) → ordenado pelos índices e sem None
        self.assertEqual(out, [1, 1, 4])


if __name__ == "__main__":
    unittest.main()
