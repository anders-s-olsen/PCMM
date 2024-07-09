%% Figure with random data
% view (-29,-13)
% cols = [0,0.4,0;0.5,0,0;0,0,0.5];
function pointsaxisfig(X,cluster_id,cols,viewval,centroid,onlyneg,target)
gridPoints = 1000;
figure('units','normalized','outerposition',[0 0 .5 1]); clf;%'visible','off',
hold on
grid off
ax = gca;
ax.Visible = 'off';
ax.XTick = [];
ax.YTick = [];


% % smaller sphere to show lines on
% [x2,y2,z2] = sphere(20); %30
% 
% if onlyneg
%     for i = 1:size(x2,1)
%         for j = 1:size(x2,2)
%             tmp = [x2(i,j),y2(i,j),z2(i,j)];
%             if sum(tmp>0)>1
%                 x2(i,j) = nan;
%                 y2(i,j) = nan;
%                 z2(i,j) = nan;
%             end
%         end
%     end
% end
% 
% sh(2) = surf(x2,y2,z2, 'EdgeAlpha', .5,'FaceColor','none','EdgeColor',[0,0,0]);


% add the three axes to the figure with bold lines and labels
line([0 1],[0 0],[0 0],'color','k','linewidth',2)
line([0 0],[0 1],[0 0],'color','k','linewidth',2)
line([0 0],[0 0],[0 1],'color','k','linewidth',2)
line([0,-1],[0,0],[0,0],'color','k','linewidth',2)
line([0,0],[0,-1],[0,0],'color','k','linewidth',2)
line([0,0],[0,0],[0,-1],'color','k','linewidth',2)
text(1,0,0,'x','FontSize',20)
text(0,1,0,'y','FontSize',20)
text(0,0,1,'z','FontSize',20)
% text(-1,0,0,'-x','FontSize',20)
% text(0,-1,0,'-y','FontSize',20)
% text(0,0,-1,'-z','FontSize',20)


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

view(viewval(1),viewval(2))
%set axis limits
axis([-1.5,1.5,-1.5,1.5,-1.5,1.5])
end