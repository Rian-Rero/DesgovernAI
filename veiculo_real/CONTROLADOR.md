# PI de velocidade no carrinho real

O `Car.set_vel()` usa agora o mesmo PI do simulador, com saturacao em
`[-1, 1] m/s^2`, anti-windup por integracao condicional, filtro exponencial
de 30 ms na velocidade do encoder e filtro de 50 ms na acao de controle. O
controle roda a cada 50 ms, como no projeto do simulador. A sintonia atual e
`kp = 4` e `ki = 2`; ela ainda deve ser confirmada no veiculo real.

No replay do log, a mudanca de `kp = 8`, `ki = 4` para `kp = 4`, `ki = 2`,
com o filtro de acao, reduziu em aproximadamente 74% a variacao RMS da acao
entre amostras e em 65% seu desvio-padrao. No modelo nominal, o degrau de
1 m/s passou de `Ta = 1,20 s` para aproximadamente `Ta = 1,30 s`, mantendo um
unico cruzamento. Assim, aceita-se cerca de 0,10 s a mais para reduzir bastante
a oscilacao do atuador. Esses numeros devem ser confirmados pelos novos logs.

O comando zero reduz o throttle enquanto o carrinho ainda se move. Abaixo de
0,03 m/s, o ESC vai para neutro e o integrador do PI e reiniciado. Trocas de
marcha tambem reiniciam o integrador.

## Primeiro ensaio

1. Apoie o carrinho de forma que as rodas possam girar sem tocar o chao e
   confirme o sinal do encoder.
2. Coloque-o em uma reta livre, mantenha acesso ao botao de emergencia e use
   inicialmente uma referencia baixa em `MAIN_VEL`.
3. Na Raspberry Pi, a partir da raiz do repositorio, execute:

   ```bash
   python3 veiculo_real/main.py
   ```

4. Interrompa com `Ctrl+C` se necessario. O programa salva `car.csv` mesmo
   quando interrompido e depois coloca o carrinho em estado de parada.

O CSV inclui `v`, `vref`, `u`, `control_percent`, `motor_pwm_percent` e
`pi_integral`. Esses sinais permitem verificar seguimento, saturacao e
windup antes de aumentar a referencia. Para mudar os ganhos sem alterar a
classe, edite `speed_kp`, `speed_ki` e `control_filter_time` no dicionario
`parameters` de `veiculo_real/main.py`.

Os testes que nao exigem hardware podem ser executados com:

```bash
python3 -m unittest discover -s veiculo_real/tests -v
```
