# examples/master-slave/qs_event_timeout.py
from multiprocessing import Process, Queue, Event
from queue import Empty  # exceção usada por mp.Queue.get(timeout=...)

def worker(task_q: Queue, result_q: Queue, stop: Event):
    while True:
        try:
            idx, x = task_q.get(timeout=0.05)
        except Empty:
            if stop.is_set():
                break
            continue
        result_q.put((idx, x * x + 1))

if __name__ == "__main__":
    task_q, result_q, stop = Queue(), Queue(), Event()
    p = Process(target=worker, args=(task_q, result_q, stop))
    p.start()

    data = [3, 5, 7, 11]
    for i, x in enumerate(data):
        task_q.put((i, x))
    stop.set()  # sinaliza fim (sem sentinela no queue)

    out = [result_q.get() for _ in range(len(data))]
    out.sort()
    print([y for _, y in out])  # [10, 26, 50, 122]
    p.join()
