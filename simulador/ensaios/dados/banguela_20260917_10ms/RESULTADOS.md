# Resultados dos ensaios de banguela

Serie: 15 ensaios, 0.5 a 2.5 m/s, amostragem 10 ms.

Perdas: dv/dt = -(0.5881 + 0.0000*v + 0.0000*v^2).

Ganho normalizado do motor: 0.770. Ganho incremental para set_u: 0.9702.

Melhor tau da grade: 10.0 ms. Atraso resolvido: nao.

Planta incremental recomendada: 0.9702/s.

A aproximacao reduzida com tau=0 nao comprova ausencia de atraso fisico. O modelo descreve variacoes de velocidade em movimento para frente no plano; o atrito constante deve ser tratado como perturbacao/compensacao. A partida, a parada, re e rampas ficam fora desta validacao.

| Velocidade inicial desejada [m/s] | RMSE do melhor ajuste [m/s] | RMSE do integrador reduzido [m/s] |
|---:|---:|---:|
| 0.5 | 0.0169 | 0.0188 |
| 1.0 | 0.0110 | 0.0176 |
| 1.5 | 0.0127 | 0.0176 |
| 2.0 | 0.0349 | 0.0418 |
| 2.5 | 0.0322 | 0.0241 |

A ultima repeticao de cada velocidade foi reservada para validacao. As velocidades reais de soltura estao em serie.json. Os CSVs preservam as entradas aplicadas e todas as fases do ensaio.
