%% Figure fit, Sphere, contour
function order = contourspherefig(mu,kappa,L,target)
if nargin < 4
    target = [];
end

gridPoints = 1000;
[XX,YY,ZZ] = sphere(gridPoints);
figure('units','normalized','outerposition',[0.5 0 .5 1]); clf;%'visible','off',
ax1 = axes;
sh(1) = surf(XX,YY,ZZ, 'FaceAlpha', .2, 'EdgeAlpha', .1,'EdgeColor','none','FaceColor','none');
hold on; axis equal;
xlabel('x'); ylabel('y'); zlabel('z');
view(-29,-13)

% smaller sphere to show lines on
[x2,y2,z2] = sphere(20); %30
sh(2) = surf(x2,y2,z2, 'EdgeAlpha', .5,'FaceColor','none','EdgeColor',[0,0,0]);
set(gca,'XColor', 'none','YColor','none','ZColor','none')
grid off
view(-29,-13)

%
% mu1 = WMM_results.mu(:,1);mu2 = WMM_results.mu(:,2);
% kappa = WMM_results.kappa;

% m = [m1;m2];
% [~,mini] = min(kappa)
% multi = [1,1];multi(mini)=8;
% kappa = multi'.*kappa;

T1 = nan(size(XX));T2 = nan(size(XX));

varfactor = 0.5;
likelihood_threshold = [-1.2,-0.7];

if ~isempty(mu)&&~isempty(kappa)
    M2 = kummer_log(0.5,1.5,kappa',50000);
    Cp = gammaln(1.5)-log(2)-1.5*log(pi)-M2';
    %     kappa = kappa*10;
    for i = 1:size(XX,1)
        for j = 1:size(XX,2)
            tmp = [XX(i,j),YY(i,j),ZZ(i,j)];
            logpdf = Cp + kappa.*((mu'*tmp').^2);
%             if (tmp*mu(:,1)).^2>1-varfactor./kappa(1)
%                 T1(i,j) = (tmp*mu(:,1)).^2;
%             elseif (tmp*mu(:,2)).^2>1-varfactor./kappa(2)
%                 T2(i,j) = (tmp*mu(:,2)).^2;
%             end
            if max(logpdf)>likelihood_threshold(1)
                T1(i,j) = max(logpdf);
            end
        end
    end
elseif ~isempty(L)
    Cp = gammaln(1.5)-log(2)-1.5*log(pi);
    logdeta1 = log(det(L(:,:,1)));
    logdeta2 = log(det(L(:,:,2)));
    L1_inv = inv(L(:,:,1));
    L2_inv = inv(L(:,:,2));
    
    for i = 1:size(XX,1)
        for j = 1:size(XX,2)
            tmp = [XX(i,j),YY(i,j),ZZ(i,j)];
            %             ll1(i,j) = Cp(1)+(-1.5)*log(tmp*A(:,:,1)*tmp');
            %             ll2(i,j) = Cp(2)+(-1.5)*log(tmp*A(:,:,2)*tmp');
            
            B1 = tmp*L1_inv*tmp';
            B2 = tmp*L2_inv*tmp';
            
            if Cp-0.5*logdeta1-1.5*log(B1)>likelihood_threshold(2)
                T1(i,j) = Cp-0.5*logdeta1-1.5*log(B1);
            elseif Cp-0.5*logdeta2-1.5*log(B2)>likelihood_threshold(2)
                T2(i,j) = Cp-0.5*logdeta2-1.5*log(B2);
            end
            
            
            %             if norm(L_chol1*tmp').^2>1-varfactor
            %                 T1(i,j) = norm((L_chol1*tmp')).^2;
            %             elseif norm(L_chol2*tmp').^2>1-varfactor
            %                 T2(i,j) = norm((L_chol2*tmp')).^2;
            %             end
        end
    end
end

% d1 = m1./norm(m1);d2 = m2./norm(m2);d3 = m3./norm(m3);
% line([0,d1(1)],[0,d1(2)],[0,d1(3)],'Color',col1,'LineWidth',2.5)
% line([0,d2(1)],[0,d2(2)],[0,d2(3)],'Color',col2,'LineWidth',2.5)
% line([0,d3(1)],[0,d3(2)],[0,d3(3)],'Color',col3,'LineWidth',2.5)

ax2 = axes;
sh(3) = surf(ax2,XX,YY,ZZ);
set(sh(3),'EdgeColor','none');
set(sh(3),'CData',(T1-min(T1(:)))./(max(T1(:))-min(T1(:))));
view(-29,-13)

ax3 = axes;
sh(4) = surf(ax3,XX,YY,ZZ);
set(sh(4),'EdgeColor','none');
set(sh(4),'CData',(T2-min(T2(:)))./(max(T2(:))-min(T2(:))));
view(-29,-13)

% ax4 = axes;
% sh(4) = surf(ax4,XX,YY,ZZ);
% set(sh(4),'EdgeColor','none');
% set(sh(4),'CData',(T3-min(T3(:)))./(max(T3(:))-min(T3(:))));


hlink = linkprop([ax1,ax2,ax3],{'XLim','YLim','ZLim','CameraUpVector','CameraPosition','CameraTarget','CameraViewAngle'});
% linkaxes([ax1,ax2,ax3,ax4])
ax2.Visible = 'off';
ax2.XTick = [];
ax2.YTick = [];
ax3.Visible = 'off';
ax3.XTick = [];
ax3.YTick = [];
% ax4.Visible = 'off';
% ax4.XTick = [];
% ax4.YTick = [];

col1 = [0,0.4,0];
col2 = [0.5,0,0];
% col3 = [0.5,0,0.5];

% cmaps{1} = ([linspace(1,0,256)',linspace(1,0.5,256)',linspace(1,0,256)']);
% cmaps{2} = ([linspace(1,0.5,256)',linspace(1,0,256)',linspace(1,0,256)']);

cmaps{1} = ([linspace(1,0,256)',linspace(1,0.5,256)',linspace(1,0.5,256)']);
cmaps{2} = ([linspace(1,0.5,256)',linspace(1,0,256)',linspace(1,0.5,256)']);

%
if ~isempty(target)
if ~isempty(mu)&&~isempty(kappa)
    [t1,~] = eigs(target(:,:,1),1);
    % [t2,~] = eigs(target(:,:,2),1);
    if (t1'*mu(:,1))^2<(t1'*mu(:,2))^2
        colormap(ax2,cmaps{2})
        colormap(ax3,cmaps{1})
        order = [2,1];
    else
        colormap(ax2,cmaps{1})
        colormap(ax3,cmaps{2})
        order = [1,2];
    end
elseif ~isempty(L)
    if norm((inv(L(:,:,1)*L(:,:,1)')./norm(inv(L(:,:,1)*L(:,:,1)'))-target(:,:,1)./norm(target(:,:,1))).^2)>norm((inv(L(:,:,2)*L(:,:,2)')./norm(inv(L(:,:,2)*L(:,:,2)'))-target(:,:,1)./norm(target(:,:,1))).^2)
        colormap(ax2,cmaps{2})
        colormap(ax3,cmaps{1})
        order = [2,1];
    else
        colormap(ax2,cmaps{1})
        colormap(ax3,cmaps{2})
        order = [1,2];
    end
end
end



end
