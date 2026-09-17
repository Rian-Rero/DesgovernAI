# Validacao do PI de velocidade

O controlador aplicado e `C(s) = (4*s + 4)/s`, com referencia de 1.5 m/s,
amostragem de 50 ms, filtro de velocidade de 30 ms e integracao condicional
durante a saturacao. Os ganhos e periodos ficam explicitos em
`simulador/fva_car/speed_controller.py`. O MATLAB le esses parametros do Python.

Os CSVs registram a velocidade longitudinal sem filtro, a velocidade filtrada,
o comando e a posicao em 10 segundos de simulacao. A validacao usa somente o
trecho reto e plano de cada cena; a cena de rampa nao implica ensaio na inclinacao.
`resultados.json` registra as metricas e `validacao.png` mostra a resposta.

O ajuste anterior usava `kp = 10.980038` e `ki = 38.801744` com velocidade sem
filtro. Antes do reinicio, a reproducao apresentou RMS de 0.02853 m/s e desvio
padrao do comando de 0.3180 m/s^2 entre 4 e 6 segundos. O erro medio pequeno
nao demonstrava acomodacao: o controle amplificava as variacoes da medicao.

O programa antigo atualizava camera e reconstruia os graficos a cada 10 ms de
tempo simulado. Uma atualizacao completa levou cerca de 75 ms no teste, alem
do controle e da pausa do Matplotlib. Em modo sincronizado, o proximo passo
espera o processamento do Python. Agora o painel atualiza ate 10 vezes por
segundo real, os objetos dos graficos sao reutilizados e o sensor de visao
e renderizado somente quando solicitado. As chamadas remotas por passo de
controle diminuiram de 12 para 3. O periodo de 50 ms manteve a velocidade
estavel e permitiu acompanhar o tempo real no teste com o painel.

O ritmo exibido no painel e tempo simulado dividido pelo tempo real. O Python
espera somente se estiver adiantado; os calculos do PI usam sempre o intervalo
de simulacao. O CSV inclui `wall_time_s` e e salvo tambem ao interromper a missao.

Para executar, abra uma das cenas no CoppeliaSim e, na raiz do repositorio, use:

```sh
venv/bin/python simulador/main.py
```

Os scripts da API agrupada sao criados temporariamente na cena e removidos
ao fechar a missao. Os ensaios de identificacao a 10 ms permanecem preservados.
