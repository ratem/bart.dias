# examples/master-slave/pipe_eof.py
from multiprocessing import Process, Pipe

def worker(conn):
    try:
        while True:
            idx, x = conn.recv()       # EOF → levanta EOFError
            conn.send((idx, (x*2)+1))
    except EOFError:
        pass

if __name__ == "__main__":
    parent, child = Pipe(duplex=True)
    p = Process(target=worker, args=(child,))
    p.start()

    data = [4, 9, 16]
    out = []
    for i, x in enumerate(data):
        parent.send((i, x))
        out.append(parent.recv())
    parent.close()  # fecha → worker recebe EOFError e encerra

    out.sort()
    print([y for _, y in out])  # [9, 19, 33]
    p.join()
