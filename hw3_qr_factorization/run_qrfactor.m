%%
% run_qrfactor.m
% 
% Homework 3, AMATH 584, Fall 2020
% Author: Jacqueline Nugent
% Last Modified: November 2, 2020
%
% NOTE: first load in the four matrices from file
% random_matrices.mat
% they are ordered as square, overdet, overdet, ill-cond
%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% run the script qrfactor.m
[Q0, R0] = qrfactor(m0);
[Q1, R1] = qrfactor(m1);
[Q2, R2] = qrfactor(m2);
[Q3, R3] = qrfactor(m3);

% reconstruct the original matrices
m0qr = Q0*R0;
m1qr = Q1*R1;
m2qr = Q2*R2;
m3qr = Q3*R3;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% check them as in the python script: 
% (NOTE: code to check for orthogonality modified
%  from https://stackoverflow.com/questions/35682635/
%  check-if-a-matrix-is-an-identity-matrix-in-matlab)

%% square matrix %%
[m, n] = size(m0);
if isequal(size(Q0), [m m]) && isequal(size(R0), [m, n])
    disp('Q and R are the correct size')
else
    disp('Q and R are NOT the correct size')
end
prod = Q0'*Q0;
if (all(~all(triu(prod,1)+tril(prod,-1)))) && (all(diag(prod)))
    if all(vecnorm(Q0))
       disp('Q is orthonormal')
    else
        disp('Q is NOT orthonormal')
    end
end
diff0 = norm(m0qr-m0);
disp(['norm of A-QR: ', num2str(diff0)]);
disp(['condition number of QR: ', num2str(cond(m0qr))]);
fprintf('\n')


%% tall + skinny #1 %%
[m, n] = size(m1);
if isequal(size(Q1), [m m]) && isequal(size(R1), [m, n])
    disp('Q and R are the correct size')
else
    disp('Q and R are NOT the correct size')
end
prod = Q1'*Q1;
if (all(~all(triu(prod,1)+tril(prod,-1)))) && (all(diag(prod)))
    if all(vecnorm(Q1))
       disp('Q is orthonormal')
    else
        disp('Q is NOT orthonormal')
    end
end
diff1 = norm(m1qr-m1);
disp(['norm of A-QR: ', num2str(diff1)]);
disp(['condition number of QR: ', num2str(cond(m1qr))]);
fprintf('\n')


%% tall + skinny #2 %%
[m, n] = size(m2);
if isequal(size(Q2), [m m]) && isequal(size(R2), [m, n])
    disp('Q and R are the correct size')
else
    disp('Q and R are NOT the correct size')
end
prod = Q2'*Q2;
if (all(~all(triu(prod,1)+tril(prod,-1)))) && (all(diag(prod)))
    if all(vecnorm(Q2))
       disp('Q is orthonormal')
    else
        disp('Q is NOT orthonormal')
    end
end
diff2 = norm(m2qr-m2);
disp(['norm of A-QR: ', num2str(diff2)]);
disp(['condition number of QR: ', num2str(cond(m2qr))]);
fprintf('\n')


%% ill-conditioned %%
[m, n] = size(m3);
if isequal(size(Q3), [m m]) && isequal(size(R3), [m, n])
    disp('Q and R are the correct size')
else
    disp('Q and R are NOT the correct size')
end
prod = Q3'*Q3;
if (all(~all(triu(prod,1)+tril(prod,-1)))) && (all(diag(prod)))
    if all(vecnorm(Q3))
       disp('Q is orthonormal')
    else
        disp('Q is NOT orthonormal')
    end
end
diff3 = norm(m3qr-m3);
disp(['norm of A-QR: ', num2str(diff3)]);
disp(['condition number of QR: ', num2str(cond(m3qr))]);
