clc, clear all;
load('myDNN_params.mat');
load("imdsTest.mat");
load("imds.mat")
size = 40;

%% 计算准确率，召回率和精确率
% ground truth 
Y_gt = imdsTest.Labels;
files = imdsTest.Files;
imdsTest = augmentedImageDatastore([size size], imdsTest, "ColorPreprocessing", "rgb2gray");
% 预测结果
[Y_pred, scores] = classify(net, imdsTest);

% 将类名转化为对应字符
numClasses = numel(net.Layers(end).Classes);
classes = net.Layers(end).Classes;
lm_struct = jsondecode(fileread("label_match.json"));
lm_cell = struct2cell(lm_struct);
lm_fn = fieldnames(lm_struct);
classes_ = [];
for i = 1: 10
    char = lm_fn{57 + i};
    lm_fn{57 + i} = char(2:end);
end
lm_map = containers.Map(lm_fn, lm_cell);
for i = 1: numClasses
    class = cellstr(classes(i));
    class = class{1};
    classes_ = [classes_, lm_map(class)];
end
classes_ = categorical(cellstr(classes_(:)));
% 计算准确率
accuracy = sum(Y_pred == Y_gt)/numel(Y_gt);

% 得到混淆矩阵
C = confusionmat(Y_gt, Y_pred);

% 计算每个字符的召回率和精确率
recall = zeros(numClasses, 1);
precision = zeros(numClasses, 1);

% 计算精确率和召回率
for i = 1:numClasses
    TP = C(i, i);
    FN = sum(C(i, :)) - TP;
    FP = sum(C(:, i)) - TP;
    recall(i) = TP / (TP + FN);
    precision(i) = TP / (TP + FP);
end

% 绘制每个字符的召回率和精确率的图像
% 绘制柱状图
figure;
set(gcf, 'Position', [100 100 1800 400]);
bar(classes_, recall);
% 添加标签和标题
title('召回率 recall');
saveas(gcf, "recall", 'jpg');


set(gcf, 'Position', [100 100 1800 400]);
bar(classes_, precision);
% 添加标签和标题
title('精确度 precision');
saveas(gcf, "precision", 'jpg');


%% 得到各个字符的数量
labelCount = countEachLabel(imds);
counts = labelCount.Count;
set(gcf, 'Position', [100 100 1800 400]);
% 绘制柱状图
bar(classes_, counts);
xlabel('Label')
ylabel('Count')
title('Distribution of Labels in imds')
saveas(gcf, "distribution", 'jpg');