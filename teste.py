import numpy as np

# Função para calcular o Fibonacci de um número
def fibonacci(n):
  """
  Calcula o n-ésimo número da sequência de Fibonacci.

  Args:
    n: O número inteiro para o qual calcular o Fibonacci.

  Returns:
    O n-ésimo número da sequência de Fibonacci.
  """
  if n <= 1:
    return n
  else:
    return fibonacci(n-1) + fibonacci(n-2)

# Função para calcular o fatorial de um número
def fatorial(n):
  """
  Calcula o fatorial de um número inteiro.

  Args:
    n: O número inteiro para o qual calcular o fatorial.

  Returns:
    O fatorial do número.
  """
  if n == 0:
    return 1
  else:
    return n * fatorial(n-1)

# Função para executar diversos cálculos sobre um vetor NumPy
def calcular_estatisticas(vetor):
  """
  Calcula a média, o desvio padrão e a variância de um vetor NumPy.

  Args:
    vetor: O vetor NumPy para o qual calcular as estatísticas.

  Returns:
    Uma tupla contendo a média, o desvio padrão e a variância do vetor.
  """
  media = np.mean(vetor)
  desvio_padrao = np.std(vetor)
  variancia = np.var(vetor)
  return media, desvio_padrao, variancia

# Exemplos de uso

# Fibonacci
for i in range(10):
  print(f"Fibonacci({i}) = {fibonacci(i)}")

# Fatorial
for i in range(5):
  print(f"{i}! = {fatorial(i)}")

# Loop for
for i in range(1, 6):
  print(f"Número: {i}")

# Loop while
i = 1
while i <= 5:
  print(f"Número: {i}")
  i += 1

# List comprehension
lista = [x**2 for x in range(1, 11)]
print(f"Lista de quadrados: {lista}")

# Chamada à função calcular_estatisticas
vetor = np.random.rand(100)
media, desvio_padrao, variancia = calcular_estatisticas(vetor)
print(f"Média: {media}, Desvio Padrão: {desvio_padrao}, Variância: {variancia}")

# Loop for chamando a função calcular_estatisticas
for i in range(5):
  vetor = np.random.rand(100)
  media, desvio_padrao, variancia = calcular_estatisticas(vetor)
  print(f"Iteração {i+1}: Média: {media}, Desvio Padrão: {desvio_padrao}, Variância: {variancia}")

# Loop while chamando a função calcular_estatisticas
i = 1
while i <= 5:
  vetor = np.random.rand(100)
  media, desvio_padrao, variancia = calcular_estatisticas(vetor)
  print(f"Iteração {i}: Média: {media}, Desvio Padrão: {desvio_padrao}, Variância: {variancia}")
  i += 1
  
# Loop for com cálculos em iterables
numeros = [1, 2, 3, 4, 5]
soma = 0
for numero in numeros:
  soma += numero * fibonacci(numero)
print(f"Soma dos produtos: {soma}")

# Nested loops com cálculos
matriz = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
soma_diagonal = 0
for i in range(len(matriz)):
  for j in range(len(matriz[i])):
    if i == j:
      soma_diagonal += matriz[i][j] * fatorial(i+j)
print(f"Soma da diagonal com fatorial: {soma_diagonal}")

# Loop while mais complexo com cálculos
i = 1
soma_pares = 0
while i <= 10:
  if i % 2 == 0:
    soma_pares += i ** 2
  i += 1
print(f"Soma dos quadrados dos números pares: {soma_pares}")

