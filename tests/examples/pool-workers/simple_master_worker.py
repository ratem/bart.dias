def process_tasks(tasks):
    # Um resultado por item (sem dependências entre itens)
    results = []
    for t in tasks:
        results.append((t * 2 + 10) * 3)
    return results


if __name__ == "__main__":
    data = [1, 2, 3, 4, 5]
    print(process_tasks(data))  # esperado: [36, 42, 48, 54, 60]
