clear
gridPoints = 100;
[XX,YY,ZZ] = sphere(gridPoints);

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
save('spherepoints1.mat','data')

data2 = zeros(3,2,size(data,1)^2);
[row1, row2] = meshgrid(1:size(data,1), 1:size(data,1));
for i = 1:size(data,1)
%     disp(num2str(i))
    data2(:,1,i) = data(row1(i),:);
    data2(:,2,i) = data(row2(i),:);
%     for j = 1:size(data,1)
%         data2(:,:,count) = [data(i,:)',data(j,:)'];
%     end
end
save('spherepoints2','data2')