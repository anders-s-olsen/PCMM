close all
point_dim = 3;
num_cols = 2;
num_points = 10000;
%%
S = eye(3);

X = zeros(num_points,point_dim,num_cols);
for n = 1:num_points
    Xi = mvnrnd(zeros(point_dim,1),S,num_cols)';
    Xsq = (Xi'*Xi)^(-0.5);
    tmp = Xi*Xsq;
    X(n,:,1) = tmp(:,1);
    X(n,:,2) = tmp(:,2);
end

pointsspherefig([squeeze(X(:,:,1));squeeze(X(:,:,2))],ones(num_points*2,1))

%%
S = [1,0,0;0,1e-12,0;0,0,1e-12];

X = zeros(num_points,point_dim,num_cols);
for n = 1:num_points
    Xi = mvnrnd(zeros(point_dim,1),S,num_cols)';
    Xsq = (Xi'*Xi)^(-0.5);
    tmp = Xi*Xsq;
    X(n,:,1) = tmp(:,1);
    X(n,:,2) = tmp(:,2);
end

pointsspherefig([squeeze(X(:,:,1));squeeze(X(:,:,2))],ones(num_points*2,1))