% Define the colors
col1 = [0, 0.9, 0];
col2 = [0, 0.5, 0];
col3 = [0.9, 0, 0];
col4 = [0.5, 0, 0];
cols = [col1; col2; col3; col4];

% Define the names for the legend entries
names = {'True comp. 1, leading', 'True comp. 1, trailing', 'True comp. 2, leading', 'True comp. 2, trailing'};


% Create a figure
figure;

% Set up the axes
hold on;
axis off;

% Number of colors
numColors = size(cols, 1);

% Plot squares with the respective colors
for i = 1:numColors
    % Calculate position for each square
    xPos = 1;  % X position
    yPos = numColors - i + 1;  % Y position (reverse order)
    
    % Plot the filled square
    fill([xPos, xPos + 1, xPos + 1, xPos], [yPos, yPos, yPos + 1, yPos + 1], cols(i, :), 'EdgeColor', 'none');
    
    % Add the text label
    text(xPos + 1.5, yPos + 0.5, names{i}, 'VerticalAlignment', 'middle', 'FontSize', 24);
end

% Set the axis limits to fit the legend
xlim([0, 10]);
ylim([0, numColors + 1]);

% Save the figure as an image
saveas(gcf, 'custom_legend1.png');

% Close the figure
close(gcf);

%%

% Define the colors
cols = [0,0.5,0.5;0.5,0,0.5];

% Define the names for the legend entries
names = {'Learned comp. 1','Learned comp. 2'};


% Create a figure
figure;

% Set up the axes
hold on;
axis off;

% Number of colors
numColors = size(cols, 1);

% Plot squares with the respective colors
for i = 1:numColors
    % Calculate position for each square
    xPos = 1;  % X position
    yPos = numColors - i + 1;  % Y position (reverse order)
    
    % Plot the filled square
    fill([xPos, xPos + 1, xPos + 1, xPos], [yPos, yPos, yPos + 1, yPos + 1], cols(i, :), 'EdgeColor', 'none');
    
    % Add the text label
    text(xPos + 1.5, yPos + 0.5, names{i}, 'VerticalAlignment', 'middle', 'FontSize', 24);
end

% Set the axis limits to fit the legend
xlim([0, 10]);
ylim([0, numColors + 1]);

% Save the figure as an image
saveas(gcf, 'custom_legend2.png');

% Close the figure
close(gcf);
