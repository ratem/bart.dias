# tests/test_master_slave_detection.py
import unittest
import pathlib, sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bdias_parser import BDiasParser
from bdias_pattern_analyzer import BDiasPatternAnalyzer


class TestMasterSlaveDetection(unittest.TestCase):
    def _identify(self, code: str):
        parser = BDiasParser()
        analyzer = BDiasPatternAnalyzer(parser)
        structured = parser.parse(code)
        return analyzer._identify_patterns(structured, code)

    def test_ms_queue_sentinel_none(self):
        code = """
from multiprocessing import Process, Queue
SENTINEL = None

def worker(task_q: Queue, result_q: Queue):
    while True:
        item = task_q.get()
        if item is SENTINEL:
            break
        idx, x = item
        result_q.put((idx, x*x + 1))
"""
        identified = self._identify(code)
        self.assertIn("master_slave", identified)
        self.assertGreater(len(identified["master_slave"]), 0)

    def test_ms_queue_event_timeout(self):
        code = """
from multiprocessing import Process, Queue, Event
from queue import Empty

def worker(task_q: Queue, result_q: Queue, stop: Event):
    while True:
        try:
            idx, x = task_q.get(timeout=0.05)
        except Empty:
            if stop.is_set():
                break
            continue
        result_q.put((idx, (x+10)*3))
"""
        identified = self._identify(code)
        self.assertIn("master_slave", identified)
        self.assertGreater(len(identified["master_slave"]), 0)

    def test_ms_pipe_eof(self):
        code = """
from multiprocessing import Process, Pipe

def worker(conn):
    try:
        while True:
            idx, x = conn.recv()   # EOF -> EOFError
            conn.send((idx, (x*2)+1))
    except EOFError:
        pass
"""
        identified = self._identify(code)
        self.assertIn("master_slave", identified)
        self.assertGreater(len(identified["master_slave"]), 0)

    def test_ms_thread_q_join(self):
        code = """
from queue import Queue
from threading import Thread

def worker(q: Queue, out: Queue):
    while True:                 # daemon no processo real
        idx, rec = q.get()
        if isinstance(rec, dict) and rec.get("enabled", True):
            out.put((idx, rec.get("value", 0) * rec.get("scale", 1) + 5))
        q.task_done()
"""
        identified = self._identify(code)
        self.assertIn("master_slave", identified)
        self.assertGreater(len(identified["master_slave"]), 0)

    def test_negative_pool_is_not_master_slave(self):
        code = """
import multiprocessing as mp

def _w(x):
    return x*x

def run(xs):
    with mp.Pool(processes=4) as pool:
        return pool.map(_w, xs)
"""
        identified = self._identify(code)
        # Garantia de que pool não é classificado como master-slave
        self.assertEqual(len(identified.get("master_slave", [])), 0)
        # (opcional) se seu analyzer marca pool_worker aqui, você pode validar:
        # self.assertGreater(len(identified.get("pool_worker", [])), 0)


if __name__ == "__main__":
    unittest.main()
