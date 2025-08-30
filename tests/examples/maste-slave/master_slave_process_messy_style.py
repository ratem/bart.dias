from multiprocessing import Process, Queue

END = None

def w(in_q, out_q):
    while True:
        it = in_q.get()
        if it is END:
            break
        i, val = it
        if i % 2 == 1:  # ignora índices ímpares
            continue
        out_q.put((i, (val or 0) * 2))

if __name__ == "__main__":
    q = Queue()
    r = Queue()
    p = Process(target=w, args=(q, r))
    p.start()

    items = [5, None, 7, 10, 0, 3]  # mistura de valores
    for i, v in enumerate(items):
        q.put((i, v))
    q.put(END)

    got = []
    # apenas metade (índices pares) gera resultado
    for _ in range((len(items) + 1) // 2):
        got.append(r.get())
    got.sort()
    print([y for _, y in got])  # esperado (pares): [10, 14, 0]

    p.join()
