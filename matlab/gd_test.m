clc; clear all; close all;

df = @(x)dfx(x);
l = @(dx,gamma)loss(dx, gamma);

initial_guess = 6;
gamma = 0.01;
precision = 0.00001;
cur_x = initial_guess;
max_iter = 1000;

[l_cx, min_x, iter] = gradient_descent(df, l, initial_guess, gamma, max_iter, precision); 

disp('The local minimum occurs at: ');
disp(min_x);


x = -10:0.02:10;
y = fx(x);

plot(x,y); grid on; grid minor; hold on;
plot(l_cx, fx(l_cx), '*-');