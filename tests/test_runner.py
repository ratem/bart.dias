import unittest
import sys, pathlib, argparse

ROOT = pathlib.Path(__file__).resolve().parents[1]  # .../bartDias/bart.dias
sys.path.insert(0, str(ROOT))  # deixa os bdias_*.py importáveis

def build_suite(names=None, pattern=None, start_dir='tests'):
    loader = unittest.TestLoader()
    if names:  # nomes pontilhados
        return loader.loadTestsFromNames(names)
    # discover por padrão de arquivo
    return loader.discover(start_dir=start_dir,
                           pattern=pattern or "test_*.py",
                           top_level_dir=str(ROOT))

def _iter_tests(suite):
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            yield from _iter_tests(item)
        else:
            yield item

def filter_suite_by_keyword(suite, keyword: str):
    if not keyword:
        return suite
    selected = unittest.TestSuite(t for t in _iter_tests(suite) if keyword in t.id())
    return selected

def main(argv=None):
    parser = argparse.ArgumentParser(description="Runner de testes (unittest) com seleção")
    parser.add_argument("names", nargs="*", help="Nomes pontilhados (ex: tests.test_mod.TestClass.test_method)")
    parser.add_argument("-p", "--pattern", default=None,
                        help='Padrão de arquivos (glob), ex: "test_pipeline*.py"')
    parser.add_argument("-k", "--keyword", default=None,
                        help="Roda apenas testes cujo id contenha a substring (ex: pipeline)")
    parser.add_argument("-v", "--verbose", action="count", default=1, help="Aumenta verbosidade")
    args = parser.parse_args(argv)

    suite = build_suite(names=args.names, pattern=args.pattern)
    suite = filter_suite_by_keyword(suite, args.keyword)

    runner = unittest.TextTestRunner(verbosity=args.verbose)
    result = runner.run(suite)
    # exit code útil para CI
    sys.exit(0 if result.wasSuccessful() else 1)

if __name__ == "__main__":
    main()
