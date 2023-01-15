function y_l = kummer_log(a,c,k,n)
N = length(k);
n_case = round(k/abs(k));
y_l = zeros(1,N);
tol = 10^(-5);
M = zeros(1,N);
Mold = 2*ones(1,N);
foo = zeros(1,N);
j = 1;
% figure,hold on
switch n_case
    case 1
        while (any(abs(M-Mold)>tol) && (j<n))
            Mold = M;
            foo = foo + log((a+j-1)/(j*(c+j-1))*k);
            M = foo;
            y_l = log_sum(y_l,foo,1);
%             if ismember(j,[1,10:10:50000]),plot(j,y_l,'k.'),shg,end
            j = j+1;
        end
    case -1
        a = c-a;
        while (any(abs(M-Mold)>tol) && (j<n))
            Mold = M;
            foo = foo + log((a+j-1)/(j*(c+j-1))*abs(k));
            M = foo;
            y_l = log_sum(y_l,foo,1);
            j = j+1;
        end
        y_l = k+y_l;
end
end

function s = log_sum(x,y,sign)
    s = x + log(1+sign*exp(y-x));
end