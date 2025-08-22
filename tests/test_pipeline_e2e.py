import os
import ast
import sys
import types
import unittest
from pathlib import Path

# --- garante imports do projeto e CWD no raiz p/ templates ---
ROOT = Path(__file__).resolve().parents[1]  # .../bart.dias
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

# módulos do seu projeto
from bdias_parser import BDiasParser
from bdias_pattern_analyzer import BDiasPatternAnalyzer
from bdias_pattern_codegen import generate_parallel_code


EXAMPLES = ROOT / "tests" / "examples" / "pipeline"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _exec_code(src: str) -> dict:
    ns: dict = {}
    exec(compile(src, "<exec>", "exec"), ns, ns)
    return ns


def _top_func_name(src: str) -> str:
    mod = ast.parse(src)
    for node in mod.body:
        if isinstance(node, ast.FunctionDef):
            return node.name
    raise AssertionError("Nenhuma função top-level encontrada no exemplo.")


def _slice_func_source(src: str, lineno: int, end_lineno: int) -> str:
    lines = src.splitlines()
    # linhas em Python são 1-based
    return "\n".join(lines[lineno-1:end_lineno]) + "\n"


def _pick_pipeline_instance_for_func(identified: dict, func_name: str) -> dict:
    """Escolhe a instância de pipeline cujo 'details.name' bate com func_name
    (ou primeira disponível, se não houver name)."""
    candidates = identified.get("pipeline", [])
    if not candidates:
        raise AssertionError("Nenhuma instância de pipeline detectada.")
    # tenta casar por name
    for inst in candidates:
        det = inst.get("details", {})
        if det.get("name") == func_name:
            return inst
    # fallback: primeira
    return candidates[0]


def _build_parallel_callable(bottleneck: dict, pattern="pipeline", strategy="default"):
    """Gera o código paralelo e retorna a função paralela <func>_parallel."""
    _orig, transformed, ctx = generate_parallel_code(
        bottleneck=bottleneck, pattern=pattern, partitioning_strategy=[strategy]
    )
    ns = _exec_code(transformed)
    func_name = bottleneck["name"]
    # Nome padrão: <func>_parallel; senão, qualquer <func>_*
    cand = f"{func_name}_parallel"
    if cand in ns and isinstance(ns[cand], types.FunctionType):
        return ns[cand], transformed, ctx
    for k, v in ns.items():
        if isinstance(v, types.FunctionType) and k.startswith(func_name + "_"):
            return v, transformed, ctx
    raise AssertionError("Função paralela não encontrada no código gerado.")


class TestPipelineE2E(unittest.TestCase):
    """Fluxo real: caminho -> analyzer detecta pipeline -> codegen -> equivalência."""

    def _assert_detects_pipeline_and_codegen(self, example_file: str, build_input):
        """
        build_input: callable que recebe o namespace sequencial e retorna (args, kwargs)
                     para chamar a função sequencial e a paralela.
        """
        path = EXAMPLES / example_file
        self.assertTrue(path.exists(), f"Exemplo não encontrado: {path}")
        src = _read(path)
        func_name = _top_func_name(src)

        # --- 1) análise (tem que detectar PIPELINE) ---
        parser = BDiasParser()
        analyzer = BDiasPatternAnalyzer(parser)
        analysis = analyzer.analyze(src)  # dict com identified_patterns etc.
        identified = analysis.get("identified_patterns", {})
        self.assertIn("pipeline", identified, "Chave 'pipeline' não presente na análise.")
        self.assertGreater(len(identified["pipeline"]), 0, "Nenhuma instância de pipeline identificada.")

        inst = _pick_pipeline_instance_for_func(identified, func_name)
        det = inst.get("details", {})
        # alguns paths trazem 'source' no details; se não, fatiamos do arquivo
        if det.get("source"):
            func_src = det["source"]
        else:
            lineno = det.get("lineno")
            end_lineno = det.get("end_lineno")
            self.assertIsNotNone(lineno, "lineno ausente no details.")
            self.assertIsNotNone(end_lineno, "end_lineno ausente no details.")
            func_src = _slice_func_source(src, lineno, end_lineno)

        # Sanity: o nome tem que bater
        self.assertEqual(det.get("name", func_name), func_name, "Nome da função não bate com o top-level.")

        # --- 2) codegen (opção 3 do assistente) ---
        bottleneck = {
            "type": "function",
            "name": func_name,
            "source": func_src,
            "lineno": det.get("lineno", 1),
        }
        par_fn, transformed, ctx = _build_parallel_callable(bottleneck, pattern="pipeline", strategy="default")

        # --- 3) executar sequencial e paralelo e comparar ---
        ns_seq = _exec_code(src)
        seq_out_args, seq_out_kwargs = build_input(ns_seq)
        seq_val = ns_seq[func_name](*seq_out_args, **seq_out_kwargs)
        par_val = par_fn(*seq_out_args, **seq_out_kwargs)

        self.assertEqual(par_val, seq_val, f"Divergência no exemplo {example_file}")

        # --- 4) reforço: recomendado deve conter TDP/TIP para pipeline ---
        rec = analysis.get("recommended_partitioning", {}).get("pipeline", {})
        strategies = rec.get("strategies", [])
        self.assertTrue(any(s in strategies for s in ("TDP", "TIP")),
                        "Estratégias recomendadas para pipeline não incluem TDP/TIP.")

    # ============ casos concretos ============
    def test_function_pipeline_e2e(self):
        """LCs encadeadas: deve detectar pipeline e gerar igual."""
        def build_input(ns):
            return ([[1, 2, 3, 4, 5]], {})

        self._assert_detects_pipeline_and_codegen("function_pipeline.py", build_input)

    def test_nested_pipeline_e2e(self):
        """for+append com temporários e estágio final: pipeline."""
        def build_input(ns):
            return ([[1, 2, 3, 4, 5]], {})

        self._assert_detects_pipeline_and_codegen("nested_pipeline.py", build_input)

    def test_function_tip_pipeline_e2e(self):
        """Pipeline com predicado (filtro)."""
        def build_input(ns):
            packets = [
                {'valid': True,  'size': 1500, 'protocol': 'TCP'},
                {'valid': False, 'size': 1200, 'protocol': 'UDP'},
                {'valid': True,  'size': 800,  'protocol': 'ICMP'},
                {'valid': True,  'size': 2000, 'protocol': 'TCP'},
            ]
            return ([packets], {})

        self._assert_detects_pipeline_and_codegen("function_tip_pipeline.py", build_input)

    # --- opcionais: só rodam se existirem no repo ---
    def test_tdp_pipeline_if_present(self):
        path = EXAMPLES / "tdp_pipeline.py"
        if not path.exists():
            self.skipTest("tdp_pipeline.py não encontrado; pulando.")
        def build_input(ns):
            # heurística simples: se a função tiver 1 arg, usa [1..5]
            fn = next(v for k, v in ns.items() if isinstance(v, types.FunctionType))
            if fn.__code__.co_argcount == 1:
                return ([[1, 2, 3, 4, 5]], {})
            self.skipTest("Assinatura não suportada automaticamente.")
        self._assert_detects_pipeline_and_codegen("tdp_pipeline.py", build_input)

    # def test_tip_pipeline_if_present(self):
    #     path = EXAMPLES / "tip_pipeline.py"
    #     if not path.exists():
    #         self.skipTest("tip_pipeline.py não encontrado; pulando.")
    #     def build_input(ns):
    #         fn = next(v for k, v in ns.items() if isinstance(v, types.FunctionType))
    #         if fn.__code__.co_argcount == 1:
    #             return ([[1, 2, 3, 4, 5]], {})
    #         self.skipTest("Assinatura não suportada automaticamente.")
    #     self._assert_detects_pipeline_and_codegen("tip_pipeline.py", build_input)


if __name__ == "__main__":
    unittest.main(verbosity=2)
