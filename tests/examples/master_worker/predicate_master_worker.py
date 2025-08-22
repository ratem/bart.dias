def filter_and_square(nums):
    # Mesmo padrão, mas com filtro (predicado) no corpo do loop
    out = []
    for x in nums:
        if x % 2 == 0:
            out.append(x * x)
    return out


if __name__ == "__main__":
    data = [1, 2, 3, 4, 5, 6]
    print(filter_and_square(data))  # esperado: [4, 16, 36]
