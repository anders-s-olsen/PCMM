%% Figure fit, Sphere, contour
function order = contourspherefig(type,params,viewval,likelihood_threshold,target,facetype)
if nargin<5
    target = [];
end

gridPoints = 200;
[XX,YY,ZZ] = sphere(gridPoints);
figure('units','normalized','outerposition',[0.5 0 .5 1]); clf;%'visible','off',
ax1 = axes;
sh(1) = surf(XX,YY,ZZ, 'FaceAlpha', .2, 'EdgeAlpha', .1,'EdgeColor','none','FaceColor','none');
hold on; axis equal;
xlabel('x'); ylabel('y'); zlabel('z');
view(viewval(1),viewval(2))

% smaller sphere to show lines on
[x2,y2,z2] = sphere(20); %30
sh(2) = surf(x2,y2,z2, 'EdgeAlpha', .5,'FaceColor','none','EdgeColor',[0,0,0]);
set(gca,'XColor', 'none','YColor','none','ZColor','none')
grid off
view(viewval(1),viewval(2))

T1 = zeros(size(XX));T2 = zeros(size(XX));

data = zeros(prod(size(XX)),3);
rows = zeros(prod(size(XX)),1);
cols = zeros(prod(size(XX)),1);
count = 1;
for i = 1:size(XX,1)
    for j = 1:size(XX,2)
        data(count,:) = [XX(i,j),YY(i,j),ZZ(i,j)];
        rows(count) = i;
        cols(count) = j;
        count = count + 1;
    end
end


% if strcmpi(type,'MACG'),
%     data2 = zeros(3,2,prod(size(XX)));
%     for i = 1:size(data,1)
%         for j = 1:size(data,1)
%             data2(:,:,count) = [data(i,:)',data(j,:)'];
%         end
%     end
% end

if strcmpi(type,'watson')
    M2 = kummer_log(0.5,1.5,params.kappa',50000);
    Cp = gammaln(1.5)-log(2)-1.5*log(pi)-M2';
    for i = 1:size(data,1)
        logpdf = log(params.pi) + Cp + params.kappa.*((params.mu'*data(i,:)').^2);
        T1(rows(i),cols(i)) = logpdf(1);
        T2(rows(i),cols(i)) = logpdf(2);
    end
elseif strcmpi(type,'ACG')
    Cp = gammaln(1.5)-log(2)-1.5*log(pi);
    logdeta1 = log(det(params.L(:,:,1)));
    logdeta2 = log(det(params.L(:,:,2)));
    L1_inv = inv(params.L(:,:,1));
    L2_inv = inv(params.L(:,:,2));

    for i = 1:size(data,1)
        B1 = data(i,:)*L1_inv*data(i,:)';
        B2 = data(i,:)*L2_inv*data(i,:)';

        T1(rows(i),cols(i)) = Cp-0.5*logdeta1-1.5*log(B1);
        T2(rows(i),cols(i)) = Cp-0.5*logdeta2-1.5*log(B2);

    end
elseif strcmpi(type,'MACG')
    loggamma_k = (0.5*log(pi)+sum(log(gamma(1.5-[0,0.5]))));
    Cp = loggamma_k - 2*log(2)-2*1.5*log(pi);
    logdeta1 = log(det(params.L(:,:,1)));
    logdeta2 = log(det(params.L(:,:,2)));
    L1_inv = inv(params.L(:,:,1));
    L2_inv = inv(params.L(:,:,2));
    T3 = zeros(size(XX));T4 = zeros(size(XX));
    theta = linspace(0,2*pi,100); 

    for i = 1:size(data,1)
        if abs(data(i,:)*[0.7,0.7,0]')>0.95;
            stophere=8;
        end
        if mod(i,10000)==0
        disp(i)
        end
        [v,w] = great_circle(data(i,:));
        for j = 1:length(theta)
            u = cos(theta(j))*v + sin(theta(j))*w;
            tmp = [data(i,:);u'];
            B1 = det(tmp*L1_inv*tmp');
            B1s(j) = B1;
            B2 = det(tmp*L2_inv*tmp');
            if B1>0
                T1(rows(i),cols(i)) = T1(rows(i),cols(i)) + Cp-logdeta1-1.5*log(B1);
            end
            if B2>0
                T2(rows(i),cols(i)) = T2(rows(i),cols(i)) + Cp-logdeta2-1.5*log(B2);
            end
        end
    end
end

if strcmp(facetype,'type1')
    T1 = (T1-min(T1(:)))./(max(T1(:))-min(T1(:)));
    T2 = (T2-min(T2(:)))./(max(T2(:))-min(T2(:)));
    T1 = T1.*(T1>likelihood_threshold);
    T2 = T2.*(T2>likelihood_threshold);
    T1(T1==0) = nan;
    T2(T2==0) = nan;
    
    ax2 = axes;
    sh(3) = surf(ax2,XX,YY,ZZ);
    set(sh(3),'EdgeColor','none');
    set(sh(3),'CData',T1);
    view(viewval(1),viewval(2))
    
    ax3 = axes;
    sh(4) = surf(ax3,XX,YY,ZZ);
    set(sh(4),'EdgeColor','none');
    set(sh(4),'CData',T2);
    view(viewval(1),viewval(2))
elseif strcmp(facetype,'type2')
%     T1 = (T1-min(T1(:)))./(max(T1(:))-min(T1(:)));
%     T2 = (T2-min(T2(:)))./(max(T2(:))-min(T2(:)));
    ax2 = axes;
    sh(3) = surf(ax2,XX,YY,ZZ);
    set(sh(3),'EdgeColor','none','FaceAlpha',0.5);
    set(sh(3),'CData',T1);
    view(viewval(1),viewval(2))
    
    T2 = nan(size(T1));
    ax3 = axes;
    sh(4) = surf(ax3,XX,YY,ZZ);
    set(sh(4),'EdgeColor','none','FaceAlpha',0.5);
    set(sh(4),'CData',T2);
    view(viewval(1),viewval(2))
end

hlink = linkprop([ax1,ax2,ax3],{'XLim','YLim','ZLim','CameraUpVector','CameraPosition','CameraTarget','CameraViewAngle'});
% linkaxes([ax1,ax2,ax3,ax4])
ax2.Visible = 'off';
ax2.XTick = [];
ax2.YTick = [];
ax3.Visible = 'off';
ax3.XTick = [];
ax3.YTick = [];

cmaps{1} = ([linspace(1,0,256)',linspace(1,0.5,256)',linspace(1,0.5,256)']);
cmaps{2} = ([linspace(1,0.5,256)',linspace(1,0,256)',linspace(1,0.5,256)']);

%
if ~isempty(target)
if strcmpi(type,'watson')
    [t1,~] = eigs(target(:,:,1),1);
    % [t2,~] = eigs(target(:,:,2),1);
    if (t1'*params.mu(:,1))^2<(t1'*params.mu(:,2))^2
        colormap(ax2,cmaps{2})
        colormap(ax3,cmaps{1})
        order = [2,1];
    else
        colormap(ax2,cmaps{1})
        colormap(ax3,cmaps{2})
        order = [1,2];
    end
elseif strcmpi(type,'ACG')
    if norm(target(:,:,1)-params.L(:,:,1),'fro')>norm(target(:,:,1)-params.L(:,:,2),'fro')
        colormap(ax2,cmaps{2})
        colormap(ax3,cmaps{1})
        order = [2,1];
    else
        colormap(ax2,cmaps{1})
        colormap(ax3,cmaps{2})
        order = [1,2];
    end
elseif strcmpi(type,'MACG')
    if norm(target(:,:,1)-params.L(:,:,1),'fro')>norm(target(:,:,1)-params.L(:,:,2),'fro')
        colormap(ax2,cmaps{2})
        colormap(ax3,cmaps{1})
%         colormap(ax4,cmaps{2})
%         colormap(ax5,cmaps{1})
        order = [2,1];
    else
        colormap(ax2,cmaps{1})
        colormap(ax3,cmaps{2})
%         colormap(ax4,cmaps{1})
%         colormap(ax5,cmaps{2})
        order = [1,2];
    end
end
end



end
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

function [v,w] = great_circle(x)
x = x/norm(x); % ensure on unit circle
v = randn(3,1); %draw random points probably not equal to x
v = v/norm(v);
if x==v
    v = randn(3,1);
    v = v/norm(v);
end
v = v-(x*v).*x'; %make v orthogonal to x
v = v/norm(v); %make v orthonormal to x
w = cross(x',v);
end