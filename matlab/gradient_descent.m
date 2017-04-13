function [l_cx, min_x, iter] = gradient_descent(df, l, initial_guess, gamma, max_iter, precision)
% Poor man's gradient descent implementation
    iter = 1;
    previous_step_size = precision+eps;
    cur_x = initial_guess;
    l_cx = [initial_guess];
    while previous_step_size>precision && iter<max_iter
        prev_x = cur_x;
        cur_x = cur_x + l(df(prev_x),gamma);
        previous_step_size = abs(cur_x-prev_x);
        l_cx = [l_cx, cur_x];
        iter = iter+1;
    end
    min_x = cur_x;
end

