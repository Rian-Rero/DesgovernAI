# ADAS de frenagem frontal

O `main.py` chama `car.set_adas_vel(MAIN_VEL, dist, valid)` em cada ciclo.
O ADAS usa a maior velocidade entre encoder bruto e filtrado para nao
subestimar a distancia de parada durante a aceleracao.

## Decisao

Com velocidade `v`, desaceleracao de coast-down `a_coast`, margem `d_min`
e atraso `tau`, a distancia estimada para parar em roda livre e:

```text
d_coast = d_min + v * tau + v^2 / (2 * a_coast)
d_ativacao = d_coast + v * anticipation_time
```

- `CRUISE (0)`: distancia acima de `d_ativacao`; usa o PI de velocidade
  com referencia limitada pelo espaco disponivel.
- `COAST (1)`: distancia entre `d_coast` e `d_ativacao`; neutro e bip.
- `BRAKE (2)`: distancia menor que `d_coast`; PI com referencia zero e
  saturacao negativa, aplicado como comando reverso de frenagem no ESC.
- `HOLD (3)`: parado ou com inversao de movimento; neutro. A retomada exige
  espaco para uma referencia adaptada maior que `stop_speed`, margem
  adicional e 0.5 s consecutivos com sensores validos e caminho livre.
- `FAULT (4)`: encoder invalido, ou ultrassom invalido com o carro parado;
  neutro e bip, sem retomar enquanto os sensores nao forem validos.

Uma intervencao permanece ativa ate parar. Durante coast-down, uma reducao
da distancia disponivel pode promover `COAST` para `BRAKE`. Depois de
entrar em `BRAKE`, o ADAS mantem a frenagem ate parar. Ultrassom invalido
com velocidade valida positiva solicita frenagem preventiva. Encoder
invalido corta a tracao: sem velocidade confiavel, nao e possivel garantir
a retirada do reverso no instante certo.

## Velocidade adaptativa

`MAIN_VEL` e um teto, nao uma velocidade obrigatoria. O ADAS inverte a
equacao da distancia de parada para limitar a referencia do PI em cada
ciclo, usando tambem a antecipacao e a margem de liberacao:

```text
espaco = max(0, distancia - clearance - release_margin)
atraso = reaction_time + anticipation_time
v_limite = 2 * espaco / (sqrt(atraso^2 + 2 * espaco / a_coast) + atraso)
v_alvo = min(MAIN_VEL, v_limite)
```

Sem espaco disponivel, `v_limite` e zero. O limite considera a leitura
atual; nao descarta distancias curtas como ruido. Com os parametros
iniciais, 2.49 m permitem referencia de aproximadamente 0.79 m/s, em vez
de bloquear a retomada por nao haver os 3.60 m exigidos para 1 m/s.
Conforme o obstaculo se aproxima, a referencia diminui. A selecao entre
roda livre e reverso continua baseada na velocidade realmente medida,
nao na referencia reduzida; nao basta baixar a referencia para presumir
que o carro ja desacelerou.

O freio reutiliza `KP_VEL`, `KI_VEL` e o anti-windup do PI existente. A
saida `u` e negativa durante a frenagem, sem referencia negativa de
velocidade. `Servos.set_brake()` aplica esse PWM de imediato, sem usar o
procedimento normal de marcha re, que espera 5 s e depois envia pulsos.
O estado `gear` continua representando a marcha de conducao. Quando o
encoder bruto indica velocidade <= `stop_speed`, o comando volta a neutro.

## Configuracao e calibracao

Os parametros ficam no dicionario `parameters['adas']` do `main.py`; os
valores restantes sao definidos em `BrakingConfig`.

| Parametro | Inicial | Unidade |
| --- | ---: | --- |
| `coast_deceleration` | `0.5 * CAR['MI'] * CAR['GRAV']` = 0.1962 | m/s^2 |
| `clearance` | 0.20 | m |
| `reaction_time` | 0.40 | s |
| `anticipation_time` | 0.30 | s |
| `stop_speed` | 0.05 | m/s |
| `release_margin` | 0.15 | m |
| `release_time` | 0.50 | s |
| `max_brake` | 1.0 | fracao do throttle reverso calibrado |

`coast_deceleration` e uma estimativa inicial do modelo, nao uma calibracao
dos carrinhos. Obtenha a desaceleracao com motor neutro em diferentes
velocidades e use um limite inferior observado, incluindo variacao de piso
e carga. A 1 m/s, o valor inicial estima `d_coast` em aproximadamente
3.15 m, incluindo a margem de 0.20 m e o atraso de 0.40 s.

`reaction_time` inclui o prazo de validade de 0.30 s dos sensores, atraso
do filtro de velocidade, ciclo de controle e atualizacao do PWM. Aumente-o
se os atrasos medidos forem maiores. A margem e medida a partir do sensor;
inclua a distancia do sensor ate a frente do carro na escolha de `clearance`.

Antes de ensaio no chao, confirme em bancada o sinal do encoder e que o
ESC responde ao primeiro comando abaixo do neutro com frenagem/reverso
contrario ao movimento. O modelo/configuracao do ESC nao esta registrado
no repositorio; o codigo nao presume que os pulsos de armar a marcha re
sejam necessarios para frear. Calibre tambem `max_brake` e verifique a
distancia real de frenagem: o ADAS nao garante parada em uma distancia
fisicamente insuficiente, nem quando o sensor nao detecta o obstaculo.

O mecanismo e frontal e recebe referencias >= 0; nao protege marcha re.

## Ultrassom e registros

O eco e capturado por bordas, sem o loop Python que ficava consultando o
pino continuamente. Na Raspberry Pi 5, `lgpio` fornece timestamps das
bordas; o atraso de entrega do callback nao entra no calculo da largura
do pulso. Na Raspberry Pi 3/4, `RPi.GPIO` registra eventos e os callbacks
usam o relogio monotonic do processo, com menor precisao temporal.
As APIs seguem a [implementacao oficial do GPIO Zero para lgpio](https://gpiozero.readthedocs.io/en/latest/_modules/gpiozero/pins/lgpio.html)
e a [documentacao do RPi.GPIO](https://sourceforge.net/p/raspberry-gpio-python/wiki/Inputs/).

O prazo de validade permanece em 0.30 s. Sem eco nao se presume caminho
livre. Erros do GPIO sao registrados e o ciclo seguinte tenta nova
leitura. Os diagnosticos distinguem `sem_leitura`, `leitura_expirada`,
`timeout_subida_echo`, `timeout_descida_echo`, `echo_alto_antes_trigger`,
`eco_fora_faixa` e `erro_gpio`.

O terminal/GUI exibe linhas `ADAS,...` nas mudancas de estado ou validade
dos sensores e a cada segundo, inclusive enquanto mantem um estado. O
log inclui tempo, validade, `v_target`, `d_release`, idade da ultima
medida valida, erro da ultima tentativa, falhas consecutivas, largura do
ultimo pulso e backend GPIO. Uma distancia numerica pode ser uma leitura
antiga; observe sempre `d_valid` e `us_age`.

O CSV mantem `u` com sinal e acrescenta `v_raw`,
`velocity_valid`, `adas_mode`, `obstacle_distance`, `obstacle_valid`,
`coast_stop_distance`, `adas_required_deceleration`, `adas_target_speed`,
`adas_release_distance`, `ultrasonic_age` e `ultrasonic_failures`. Decisao e comando
sao atualizados na mesma amostra de sensores, sem duplicar o tempo.

## Testes sem hardware

```sh
python3 -m unittest discover -s veiculo_real/tests -v
```

Os testes precisam de NumPy. Os drivers de I2C/serial sao substituidos
apenas na importacao; a decisao ADAS, o PI e a atuacao PWM sao reais.
