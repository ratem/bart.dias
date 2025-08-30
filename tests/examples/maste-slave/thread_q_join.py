# examples/master-slave/thread_q_join.py
from queue import Queue
from threading import Thread

def worker(q: Queue, out: Queue):
    while True:                        # thread daemon roda até o processo morrer
        idx, rec = q.get()
        if isinstance(rec, dict) and rec.get("enabled", True):
            out.put((idx, rec.get("value", 0) * rec.get("scale", 1) + 5))
        q.task_done()

if __name__ == "__main__":
    q, out = Queue(), Queue()
    t = Thread(target=worker, args=(q, out), daemon=True)
    t.start()

    data = [{"value": 2, "scale": 3}, {"value": 1, "enabled": False}, 42, {}]
    for i, rec in enumerate(data):
        q.put((i, rec))

    q.join()  # espera processar tudo (sem sentinela)
    got = []
    while not out.empty():
        got.append(out.get())
    got.sort()
    print([y for _, y in got])  # [11, 5]
