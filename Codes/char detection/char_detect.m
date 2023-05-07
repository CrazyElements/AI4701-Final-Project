clc, clear all;
load('myDNN_params.mat')

% 得到类别到字符的映射
lm_struct = jsondecode(fileread("label_match.json"));
lm_cell = struct2cell(lm_struct);
lm_fn = fieldnames(lm_struct);
for i = 1: 10
    char = lm_fn{57 + i};
    lm_fn{57 + i} = char(2:end);
end
lm_map = containers.Map(lm_fn, lm_cell);

% 预测
dir = "images\";
levels = ["easy", "medium", "difficult"];
nums = 1: 3;
char_nums = 0: 7;
for level = levels
    for num = nums
        num = num2str(num);
        plate_num = [];
        for char_num = char_nums
            char_num = num2str(char_num);
            filename = [dir, level, '\', num, '\', num, '_char_', char_num, '.jpg'];
            filename = strjoin(filename, '');
            if exist(filename)
                input = imread(filename);
                input = imresize(input, [40 40]);
                output_cell = cellstr(classify(net, input));
                output_varname = output_cell{1};
                output = lm_map(output_varname);
                plate_num = [plate_num, output];
                if char_num == '1'
                    plate_num = [plate_num, "·"];
                end
            end
        end
        disp(strjoin(["难度为 ", level, " 的第 ", num, " 张图像中车辆的车牌号为 ", plate_num], ''))
    end
end
