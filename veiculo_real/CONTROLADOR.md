# PI de velocidade no carrinho real

O `Car.set_vel()` usa agora o mesmo PI do simulador, com saturacao em
`[-1, 1] m/s^2`, anti-windup por integracao condicional e filtro exponencial
de 30 ms na velocidade do encoder. O controle roda a cada 50 ms, como no
projeto do simulador. Os valores iniciais sao `kp = 8` e `ki = 4`; eles ainda
precisam ser validados e, se necessario, sintonizados no veiculo real.

O ganho proporcional anterior, `kp = 12`, foi reduzido porque amplificava mais
o ruido e produzia tres cruzamentos da referencia no modelo. Com `kp = 8`, o
degrau nominal de 1 m/s manteve `Ta` em aproximadamente 1,20 s, com um unico
cruzamento e sobressinal calculado de aproximadamente 0,38%. Esses numeros sao
previsoes do modelo e devem ser confirmados pelos logs do carrinho.

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
classe, edite `speed_kp` e `speed_ki` no dicionario `parameters` de
`veiculo_real/main.py`.

Os testes que nao exigem hardware podem ser executados com:

```bash
python3 -m unittest discover -s veiculo_real/tests -v
```
