import numpy as np
import multiprocessing as mp


# Função para calcular o Fibonacci de um número
def fibonacci(n):
    if n <= 1:
        return n
    else:
        return fibonacci(n - 1) + fibonacci(n - 2)


# Função para calcular o fatorial de um número
def fatorial(n):
    if n == 0:
        return 1
    else:
        return n * fatorial(n - 1)


# Função para executar diversos cálculos sobre um vetor NumPy
def calcular_estatisticas(vetor):
    media = np.mean(vetor)
    desvio_padrao = np.std(vetor)
    variancia = np.var(vetor)
    return media, desvio_padrao, variancia


# Função com dependência global
global_var = 0


def funcao_com_global():
    global global_var
    global_var += 1
    return global_var


# Função com loop aninhado e dependência
def nested_loop_function(n):
    result = 0
    for i in range(n):
        for j in range(n):
            result += i * j
    return result


# Função com chamada recursiva e loop
def recursive_loop_function(n):
    if n <= 0:
        return 0
    result = 0
    for i in range(n):
        result += recursive_loop_function(n - 1)
    return result


# Função para teste de paralelização de loop
def loop_function(start, end):
    return sum(range(start, end))


# Função para teste de paralelização de função
def parallelizable_function(x):
    return x * x


# Função que contém um loop
def function_with_loop(n):
    result = []
    for i in range(n):
        result.append(i * i)
    return result


if __name__ == "__main__":
    print("Teste de Fibonacci:")
    for i in range(10):
        print(f"Fibonacci({i}) = {fibonacci(i)}")

    print("\nTeste de Fatorial:")
    for i in range(5):
        print(f"{i}! = {fatorial(i)}")

    print("\nTeste de Loop For:")
    for i in range(1, 6):
        print(f"Número: {i}")

    print("\nTeste de Loop While:")
    i = 1
    while i <= 5:
        print(f"Número: {i}")
        i += 1

    print("\nTeste de List Comprehension:")
    lista = [x ** 2 for x in range(1, 11)]
    print(f"Lista de quadrados: {lista}")

    print("\nTeste de Cálculo de Estatísticas:")
    vetor = np.random.rand(100)
    media, desvio_padrao, variancia = calcular_estatisticas(vetor)
    print(f"Média: {media}, Desvio Padrão: {desvio_padrao}, Variância: {variancia}")

    print("\nTeste de Loop For com Chamada de Função:")
    for i in range(5):
        vetor = np.random.rand(100)
        media, desvio_padrao, variancia = calcular_estatisticas(vetor)
        print(f"Iteração {i + 1}: Média: {media}, Desvio Padrão: {desvio_padrao}, Variância: {variancia}")

    print("\nTeste de Loop While com Chamada de Função:")
    i = 1
    while i <= 5:
        vetor = np.random.rand(100)
        media, desvio_padrao, variancia = calcular_estatisticas(vetor)
        print(f"Iteração {i}: Média: {media}, Desvio Padrão: {desvio_padrao}, Variância: {variancia}")
        i += 1

    print("\nTeste de Loop For com Cálculos em Iterables:")
    numeros = [1, 2, 3, 4, 5]
    soma = 0
    for numero in numeros:
        soma += numero * fibonacci(numero)
    print(f"Soma dos produtos: {soma}")

    print("\nTeste de Nested Loops com Cálculos:")
    matriz = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    soma_diagonal = 0
    for i in range(len(matriz)):
        for j in range(len(matriz[i])):
            if i == j:
                soma_diagonal += matriz[i][j] * fatorial(i + j)
    print(f"Soma da diagonal com fatorial: {soma_diagonal}")

    print("\nTeste de Loop While Mais Complexo com Cálculos:")
    i = 1
    soma_pares = 0
    while i <= 10:
        if i % 2 == 0:
            soma_pares += i ** 2
        i += 1
    print(f"Soma dos quadrados dos números pares: {soma_pares}")

    print("\nTeste de Função com Dependência Global:")
    for _ in range(5):
        print(f"Valor global: {funcao_com_global()}")

    print("\nTeste de Função com Loop Aninhado:")
    print(f"Resultado do loop aninhado: {nested_loop_function(5)}")

    print("\nTeste de Função Recursiva com Loop:")
    print(f"Resultado da função recursiva com loop: {recursive_loop_function(3)}")

    # NOVOS TESTES PARA COMBOS

    print("\n--- TESTES DE COMBOS ---")

    print("\nTeste de While Loop Contendo For Loop:")
    counter = 0
    while counter < 3:
        print(f"Iteração while {counter}:")
        for i in range(3):
            print(f"  For loop interno: {i}")
        counter += 1

    print("\nTeste de For Loop com Chamada de Função Recursiva:")
    for i in range(1, 5):
        result = fibonacci(i)
        print(f"Fibonacci({i}) = {result}")

    print("\nTeste de Nested Loops com Diferentes Profundidades:")
    for i in range(3):
        print(f"Nível 1: {i}")
        for j in range(2):
            print(f"  Nível 2: {j}")
            for k in range(2):
                print(f"    Nível 3: {k}")

    print("\nTeste de Loop Contendo Função que Contém Loop:")
    for i in range(3):
        result = function_with_loop(i + 2)
        print(f"Resultado para i={i}: {result}")

    print("\nTeste de Combinação de While, For e Recursão:")
    i = 0
    while i < 3:
        print(f"While loop iteração {i}:")
        for j in range(2):
            result = fibonacci(i + j)
            print(f"  For loop j={j}, Fibonacci({i + j}) = {result}")
        i += 1

    print("\nTeste de Paralelização com Multiprocessing:")
    with mp.Pool(processes=4) as pool:
        results = pool.map(parallelizable_function, range(10))
    print(f"Resultados paralelos: {results}")

    print("\nTeste de For Loop Dentro de While com Função que Contém Loop:")
    counter = 0
    while counter < 2:
        print(f"While iteração {counter}:")
        for i in range(3):
            result = function_with_loop(i + 1)
            print(f"  For iteração {i}, resultado: {result}")
        counter += 1

    print("\nTeste de Recursão Múltipla com Loops:")


    def recursive_with_multiple_loops(n):
        if n <= 0:
            return 0
        result = 0
        # For loop dentro da função recursiva
        for i in range(n):
            result += i
        # While loop dentro da função recursiva
        j = 0
        while j < n:
            result += j
            j += 1
        # Chamada recursiva
        return result + recursive_with_multiple_loops(n - 1)


    print(f"Resultado da recursão múltipla: {recursive_with_multiple_loops(3)}")
