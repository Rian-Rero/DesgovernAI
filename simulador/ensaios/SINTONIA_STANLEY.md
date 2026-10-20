# Sintonia lateral a partir dos logs - 2026-10-20

## Diagnostico

Os logs mais recentes do usuario, em especial `20261020_200906_732955`,
mostraram RMS visual de 4.5 cm nas curvas, pico de 16.8 cm e saturacao
do esterçamento em apenas 0.4% das amostras de curva. O ganho efetivamente
aplicado era 1.5 1/s. O limite de 20 graus nao era a causa dominante da
correcao lenta: havia margem para aumentar a realimentacao lateral.

## Comparacao controlada

Foram comparados cinco ganhos no mesmo trecho de `simulador_cones.ttt`:

- pose inicial mundial: x=11.025 m, y=-9.124 m, heading=0 graus;
- referencia de cruzeiro: 1.5 m/s, com o mesmo limitador nas curvas;
- camera a 30 graus, periodo de 0.05 s, duracao de 60 s;
- avaliacao a partir de 5 s; RMS visual de curvas calculado com
  `line_curvature_inv_m > 0.12` e observacao valida;
- referencia independente obtida da textura e UV de `/Pista`.

| Ganho k (1/s) | RMS visual nas curvas (cm) | RMS independente do centro do carro (cm) | Quadros perdidos depois de 5 s |
| ---: | ---: | ---: | ---: |
| 1.5 | 4.41 | 2.97 | 0 |
| 2.5 | 3.81 | 1.73 | 0 |
| 3.5 | 3.63 | 1.17 | 0 |
| **5.0** | **3.32** | **0.93** | **0** |
| 7.0 | 3.11 | 7.29 | 345 |

O ganho 7 foi descartado: perdeu a linha na transicao em torno de 43 s
e parou. Seu RMS visual aparentemente menor considera apenas observacoes
validas e nao representa sucesso de rastreamento.

Entre os ajustes avaliados sem perda de linha, o ganho 5 apresentou:

- reducao de 25% no RMS visual nas curvas;
- reducao de 69% no RMS independente relativo a linha;
- pico independente de 5.36 cm, contra 10.84 cm no ganho original;
- menor variacao dos comandos: RMS da taxa de esterçamento de 12.96 graus/s,
  contra 22.59 graus/s, uma reducao de 43%;
- velocidade media de 1.355 m/s, contra 1.352 m/s no caso original.

Depois do primeiro pico da curva, o primeiro retorno a faixa de +/-2 cm
visuais, sustentado por pelo menos 0.5 s, ocorreu em 1.95 s com k=5,
contra 13.55 s com k=1.5. Esse indicador descreve esse transiente do
percurso; nao e um tempo de acomodacao universal do sistema.

A textura tem resolucao espacial de 1.8 cm/pixel. Os valores subpixel do
RMS independente sao indicadores comparativos aproximados, nao uma
garantia de acuracia absoluta milimetrica. O erro visual usa o eixo
dianteiro; a avaliacao independente usa o centro do corpo. Os dois
referenciais diferem durante curvas e nao devem ser confundidos.

## Registros e repeticao

Os dados locais foram preservados em:

- `logs/stanley_tuning_20261020_201320/`: ganhos 1.5, 2.5 e 3.5;
- `logs/stanley_tuning_20261020_201617/`: ganhos 5 e 7;
- `logs/stanley_tuning_20261020_201320/final_comparison.json`: tabela consolidada;
- `logs/stanley_tuning_20261020_201320/comparison.png`: respostas e comparacao.

Para repetir, com a mesma cena aberta e simulacao parada:

```bash
venv/bin/python simulador/ensaios/sintonizar_stanley.py \
  --ganhos 1.5 2.5 3.5 5 7 --vref 1.5 --ts 60
```

Adicionar `--port 23010` para uma instancia de teste nessa porta. O script
restaura a pose do carro e a montagem da camera entre ensaios e registra
perdas de linha, velocidade e taxa de esterçamento, alem dos erros.

Os novos ensaios tambem salvam `controller_config.json` e as parcelas
angular e lateral do Stanley no `car.csv`. Assim, nao e necessario inferir
o ganho ou atribuir picos a saturacao sem verificar o comando nominal.

Os limites de esterçamento, a parada por perda da linha e a reducao de
velocidade nas curvas foram mantidos. Persistem picos nas transicoes
abruptas da linha; nao se espera erro identicamente nulo nessas quinas.

## Validacao longa do ganho escolhido

Depois da comparacao, o ganho 5 foi testado na pose original da cena,
com referencia de 1.5 m/s, por 260 s:

| Medida | Resultado |
| --- | ---: |
| Distancia percorrida | 356.26 m |
| Retorno pela reta inicial apos uma volta | 181.4 s |
| Observacoes validas depois de 5 s | 100% |
| Paradas por perda da linha depois de 5 s | 0 |
| RMS visual no eixo dianteiro | 2.72 cm |
| Pico visual no eixo dianteiro | 26.57 cm |
| RMS independente do centro do corpo | 0.91 cm |
| Percentil 95 independente | 1.84 cm |
| Pico independente | 6.34 cm |
| Saturacao do esterçamento depois de 5 s | 0.12% |

Registro local: `logs/20261020_201822_463435/`, com dados, configuracao,
relatorio e sobreposicao da trajetoria na pista. A volta e identificada
pelo cruzamento de x=-24 m para leste, com -9 < y < -7 m, apos 60 s.
Essa verificacao usa apenas as posicoes de avaliacao do log, nao a
realimentacao do controlador.

Esse ensaio confirma uma volta continua para a sintonia escolhida. Nao
substitui repeticoes em outras condicoes nem a validacao da camera e do
servo do carrinho real.
