# Resultados dos ensaios de banguela

Serie: 15 ensaios, 0.5 a 2.5 m/s, amostragem 50 ms.

Perdas: dv/dt = -(0.5943 + 0.0000*v + 0.0000*v^2).

Ganho normalizado do motor: 0.780. Ganho incremental para set_u: 0.9828.

Melhor tau da grade: 20.0 ms. Atraso resolvido: nao.

Planta incremental recomendada: 0.9828/s.

A aproximacao reduzida com tau=0 nao comprova ausencia de atraso fisico. O modelo descreve variacoes de velocidade em movimento para frente no plano; o atrito constante deve ser tratado como perturbacao/compensacao. A partida, a parada, re e rampas ficam fora desta validacao.

| Velocidade inicial desejada [m/s] | RMSE do melhor ajuste [m/s] | RMSE do integrador reduzido [m/s] |
|---:|---:|---:|
| 0.5 | 0.0201 | 0.0225 |
| 1.0 | 0.0216 | 0.0400 |
| 1.5 | 0.0090 | 0.0271 |
| 2.0 | 0.0156 | 0.0114 |
| 2.5 | 0.0164 | 0.0211 |

A ultima repeticao de cada velocidade foi reservada para validacao. As velocidades reais de soltura estao em serie.json. Os CSVs preservam as entradas aplicadas e todas as fases do ensaio.
