clc;
n_data = length(data_lora);
pl_d = data_lora;

disp('fminunc');
[n, fval] = fminunc(@ldpltod, 5)
disp('fminbnd');
[n, fval] = fminbnd(@ldpltod, 1,10)
disp('lsqnonlin');
[n, fval] = lsqnonlin(@ldpltod, 5)
disp('fminsearch');
[n, fval] = fminsearch(@ldpltod, 5)