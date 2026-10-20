import sys
import pandas as pd
import matplotlib.pyplot as plt

# Verifica se o usuário passou o caminho do arquivo
if len(sys.argv) < 2:
    print("Uso: python plot.py /caminho/para/arquivo.csv")
    sys.exit(1)

arquivo = sys.argv[1]

# Ler CSV
dados = pd.read_csv(arquivo)

# Plot
plt.figure(figsize=(10, 5))

plt.plot(dados["t"], dados["v"], label="Velocidade medida")
plt.plot(dados["t"], dados["vref"], "--", label="Velocidade de referência")

plt.xlabel("Tempo [s]")
plt.ylabel("Velocidade [m/s]")
plt.title("Velocidade do veículo")

plt.grid(True)
plt.legend()
plt.tight_layout()

plt.show()
