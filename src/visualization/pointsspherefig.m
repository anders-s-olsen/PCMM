%% Figure with random data
% view (-29,-13)
% cols = [0,0.4,0;0.5,0,0;0,0,0.5];
function pointsspherefig(X,cluster_id,cols,viewval,centroid,onlyneg,target)
gridPoints = 1000;
[XX,YY,ZZ] = sphere(gridPoints);
figure('units','normalized','outerposition',[0 0 .5 1]); clf;%'visible','off',

sh(1) = surf(XX,YY,ZZ, 'FaceAlpha', .2, 'EdgeAlpha', .1,'EdgeColor','none','FaceColor','none');
hold on; axis equal;
% xlabel('x'); ylabel('y'); zlabel('z');
grid off
ax = gca;
ax.Visible = 'off';
ax.XTick = [];
ax.YTick = [];
view(viewval(1),viewval(2))

% smaller sphere to show lines on
[x2,y2,z2] = sphere(20); %30

if onlyneg
    for i = 1:size(x2,1)
        for j = 1:size(x2,2)
            tmp = [x2(i,j),y2(i,j),z2(i,j)];
            if sum(tmp>0)>1
                x2(i,j) = nan;
                y2(i,j) = nan;
                z2(i,j) = nan;
            end
        end
    end
end

sh(2) = surf(x2,y2,z2, 'EdgeAlpha', .5,'FaceColor','none','EdgeColor',[0,0,0]);
% set(gca,'XColor', 'none','YColor','none','ZColor','none')
grid off
view(viewval(1),viewval(2))

% cols = [0,0.4,0;0.5,0,0;0,0,0.5];

% if ~isempty(centroid)
%     [t1,~] = eigs(target(:,:,1),1);
%     if ndims(centroid)==2
%         if (t1'*(centroid(:,1)/norm(centroid(:,1)))).^2<(t1'*(centroid(:,2)/norm(centroid(:,2)))).^2
%             cols = [cols(2,:);cols(1,:)];
%         end
%     elseif ndims(centroid)==3
%         if (t1'*centroid(1,:,1)').^2<(t1'*centroid(2,:,1)').^2
%             cols = [cols(2,:);cols(1,:)];
%         end
%     end
% end
for i = 1:numel(unique(cluster_id))
    scatter3(X(cluster_id==i,1), X(cluster_id==i,2), X(cluster_id==i,3),7,cols(i,:),'filled');
end

if ~isempty(centroid)
    if ndims(centroid)==2
        for k = 1:size(centroid,2)
            scatter3(centroid(1,k),centroid(2,k),centroid(3,k),2000,cols(k,:),'x','linewidth',20)
        end
    elseif ndims(centroid)==3
        for k = 1:size(centroid,1)
            scatter3(centroid(k,1,1),centroid(k,2,1),centroid(k,3,1),2000,cols(k,:),'x','linewidth',20)
            scatter3(centroid(k,1,2),centroid(k,2,2),centroid(k,3,2),2000,cols(k,:),'x','linewidth',20)
        end
    end
end

end