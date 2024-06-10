function logkum = kummer_log(a,c,k,n)
N = length(k);
n_case = round(k/abs(k));
logkum = zeros(1,N);
logkum_old = ones(1,N);
tol = 10^(-10);
% M = zeros(1,N);
% Mold = 2*ones(1,N);
foo = zeros(1,N);
j = 1;
% figure,hold on
switch n_case
    case 1
        while (any(abs(logkum-logkum_old)>tol) && (j<n))
%             Mold = M;
            logkum_old = logkum;
            foo = foo + log((a+j-1)/(j*(c+j-1))*k);
%             M = foo;
            logkum = log_sum(logkum,foo,1);
%             if ismember(j,[1,10:10:50000]),plot(j,logkum,'k.'),shg,end
            j = j+1;
        end
    case -1
        a = c-a;
        while (any(abs(logkum-logkum_old)>tol) && (j<n))
%             Mold = M;
            logkum_old = logkum;
            foo = foo + log((a+j-1)/(j*(c+j-1))*abs(k));
%             M = foo;
            logkum = log_sum(logkum,foo,1);
            j = j+1;
        end
        logkum = k+logkum;
end
end

function s = log_sum(x,y,sign)
    s = x + log(1+sign*exp(y-x));
end