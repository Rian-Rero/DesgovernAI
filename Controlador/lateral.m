% Modelo lateral cinematico, sem identificacao experimental dos pneus/servo.
% Estados [e_y; e_psi], erros carro - caminho, pequenos angulos, pista reta.
L = 0.302; % m
v0 = 1.5; % m/s
A = [0 v0; 0 0];
B = [0; v0/L];
% B usa delta GEOMETRICO positivo para esquerda. Na cena Car, st positivo
% vira para direita: delta = steering_sign*st (confirmado no log).
steering_sign = -1;
B_comando_cena = steering_sign*B;
C = eye(2);
D = zeros(2,1);
planta_lateral = ss(A,B,C,D);
planta_comando_cena = ss(A,B_comando_cena,C,D);
G_guinada = tf(v0/L,1);  % r/delta
G_orientacao = tf(v0/L,[1 0]);
G_lateral = tf(v0^2/L,[1 0 0]);
% Stanley implementado em Python:
% delta = -e_psi + atan2(-k*e_y_front, abs(v)+v_soft)
% e_y_front ~= e_y + L*e_psi (estado e_y medido no eixo traseiro).
% A percepcao Python usa linha - carro, logo ambos sinais sao invertidos.
k = 1.5;
v_soft = 0.3;
K = [k/(v0+v_soft), 1+k*L/(v0+v_soft)];
polos_malha_fechada = eig(A-B*K)
