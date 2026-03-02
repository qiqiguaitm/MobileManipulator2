%% stereo_extrinsics_fixed_intrinsics.m
% ================================================================
% 固定内参，只估计双目外参 R, T
%
% 使用方法 (不需要编辑任何代码):
%   1. 确保当前目录下有:
%      - top/          文件夹 (Top 相机图片)
%      - chassis/      文件夹 (Chassis 相机图片)
%      - camera_intrinsics.m  (由 _cc_dump_intrinsics.py 自动生成)
%   2. 在 MATLAB 命令窗口输入:
%        stereo_extrinsics_fixed_intrinsics
%   3. 等待运行完毕，看输出结果
%
% 输出约定:
%   P_chassis = R * P_top + T
%   Camera1 = Top, Camera2 = Chassis
% ================================================================

clear; clc;

%% ===================== 加载内参 =====================

if ~isfile('camera_intrinsics.m')
    error(['找不到 camera_intrinsics.m !\n' ...
           '请先在机器人端运行:\n' ...
           '  python3 scripts/_cc_dump_intrinsics.py\n' ...
           '然后把生成的 camera_intrinsics.m 拷贝到当前目录']);
end

% 执行 camera_intrinsics.m, 加载变量:
%   fx_top, fy_top, cx_top, cy_top, imageSize_top
%   fx_ch,  fy_ch,  cx_ch,  cy_ch,  imageSize_ch
camera_intrinsics;

fprintf('已加载内参:\n');
fprintf('  Top:     fx=%.2f fy=%.2f cx=%.2f cy=%.2f [%dx%d]\n', ...
    fx_top, fy_top, cx_top, cy_top, imageSize_top(2), imageSize_top(1));
fprintf('  Chassis: fx=%.2f fy=%.2f cx=%.2f cy=%.2f [%dx%d]\n', ...
    fx_ch, fy_ch, cx_ch, cy_ch, imageSize_ch(2), imageSize_ch(1));

%% ===================== 配置 =====================

topDir = fullfile(pwd, 'top');
chassisDir = fullfile(pwd, 'chassis');
squareSize = 25;  % 棋盘格方格大小 (mm), 必须与实际一致!

% 旧外参 (用于一致性对比)
T_old = [49.28, -560.39, -537.47];  % mm, 来自 _back 标定

%% ===================== 创建内参对象 =====================

intrTop = cameraIntrinsics([fx_top fy_top], [cx_top cy_top], imageSize_top);
intrCh  = cameraIntrinsics([fx_ch  fy_ch],  [cx_ch  cy_ch],  imageSize_ch);

%% ===================== 检测棋盘格角点 =====================

topFiles = dir(fullfile(topDir, '*.png'));
[~, idx] = sort({topFiles.name}); topFiles = topFiles(idx);
chFiles  = dir(fullfile(chassisDir, '*.png'));
[~, idx] = sort({chFiles.name}); chFiles = chFiles(idx);

topPaths = fullfile({topFiles.folder}, {topFiles.name})';
chPaths  = fullfile({chFiles.folder},  {chFiles.name})';

fprintf('\n加载图像: Top=%d, Chassis=%d\n', length(topPaths), length(chPaths));

[imagePoints, boardSize, pairsUsed] = detectCheckerboardPoints(topPaths, chPaths);
worldPts = generateCheckerboardPoints(boardSize, squareSize);

nPairs = size(imagePoints, 3);
fprintf('棋盘格检测: %d / %d 对有效\n', nPairs, length(topFiles));

if nPairs < 5
    warning('有效图像对太少 (<5), 结果可能不可靠');
end

%% ===================== 逐对估计外参 =====================

quats  = zeros(nPairs, 4);   % [w x y z]
trans  = zeros(nPairs, 3);   % [tx ty tz] in mm
errors = zeros(nPairs, 1);   % 重投影误差

for i = 1:nPairs
    pts_top = imagePoints(:,:,i,1);   % Nx2
    pts_ch  = imagePoints(:,:,i,2);

    try
        % R2022b+ 用 estimateExtrinsics, 旧版用 extrinsics
        if exist('estimateExtrinsics', 'file')
            tform1 = estimateExtrinsics(pts_top, worldPts, intrTop);
            tform2 = estimateExtrinsics(pts_ch,  worldPts, intrCh);
            R1 = tform1.R;  t1 = tform1.Translation;
            R2 = tform2.R;  t2 = tform2.Translation;
        else
            [R1, t1] = extrinsics(pts_top, worldPts, intrTop);
            [R2, t2] = extrinsics(pts_ch,  worldPts, intrCh);
        end
    catch e
        fprintf('  Pair %d: PnP 失败 (%s)\n', i, e.message);
        errors(i) = Inf;
        continue;
    end

    % 计算相对变换: P_chassis = R_rel * P_top + T_rel
    %   MATLAB 约定: P_cam = [P_world] * R + t  (行向量)
    %   标准约定:    P_cam = R' * P_world + t'
    %   消去 P_world 得:
    %     R_rel = R2' * R1
    %     T_rel = t2' - R_rel * t1'
    R_rel = R2' * R1;
    T_rel = t2' - R_rel * t1';

    quats(i,:) = rotm2quat(R_rel);   % MATLAB: [w x y z]
    trans(i,:) = T_rel';

    % 重投影误差
    P_w_3d = [worldPts, zeros(size(worldPts,1),1)];
    P_in_top = P_w_3d * R1 + t1;
    P_in_ch  = P_w_3d * R2 + t2;

    uv_top_proj = [P_in_top(:,1)./P_in_top(:,3)*fx_top + cx_top, ...
                   P_in_top(:,2)./P_in_top(:,3)*fy_top + cy_top];
    uv_ch_proj  = [P_in_ch(:,1)./P_in_ch(:,3)*fx_ch + cx_ch, ...
                   P_in_ch(:,2)./P_in_ch(:,3)*fy_ch + cy_ch];

    err1 = sqrt(mean(sum((pts_top - uv_top_proj).^2, 2)));
    err2 = sqrt(mean(sum((pts_ch  - uv_ch_proj).^2,  2)));
    errors(i) = (err1 + err2) / 2;
end

%% ===================== 稳健统计 =====================

valid = isfinite(errors) & (errors < 2.0);
nValid = sum(valid);

fprintf('\n========================================\n');
fprintf('有效估计: %d / %d', nValid, nPairs);
if any(~valid)
    fprintf(' (剔除 %d 个离群值)', sum(~valid));
end
fprintf('\n========================================\n');

if nValid < 3
    error('有效估计太少, 无法得到可靠结果');
end

q_mean = mean(quats(valid,:), 1);
q_mean = q_mean / norm(q_mean);
R_final = quat2rotm(q_mean);

T_final = median(trans(valid,:), 1)';

%% ===================== 输出结果 =====================

fprintf('\n============ 最终结果 ============\n');
fprintf('约定: P_chassis = R * P_top + T\n');
fprintf('      Camera1=Top, Camera2=Chassis\n\n');

fprintf('Rotation Matrix:\n');
fprintf('  [%.12f, %.12f, %.12f;\n', R_final(1,:));
fprintf('   %.12f, %.12f, %.12f;\n', R_final(2,:));
fprintf('   %.12f, %.12f, %.12f]\n', R_final(3,:));

fprintf('\nTranslation (mm):  [%.4f, %.4f, %.4f]\n', T_final);
fprintf('Translation (m):   [%.6f, %.6f, %.6f]\n', T_final/1000);

fprintf('\nQuaternion [w, x, y, z]:\n');
fprintf('  [%.16f, %.16f, %.16f, %.16f]\n', q_mean);

%% ===================== 质量评估 =====================

T_std = std(trans(valid,:), 0, 1);
T_range = range(trans(valid,:), 1);

fprintf('\n============ 质量评估 ============\n');
fprintf('平移标准差 (mm):  [%.2f, %.2f, %.2f]\n', T_std);
fprintf('平移极差   (mm):  [%.2f, %.2f, %.2f]\n', T_range);
fprintf('平均重投影误差:   %.3f pixel\n', mean(errors(valid)));

if all(T_std < 5)
    fprintf('>> 平移一致性: 良好 (std < 5mm)\n');
elseif all(T_std < 10)
    fprintf('>> 平移一致性: 一般 (std < 10mm), 建议增加图片\n');
else
    fprintf('>> 平移一致性: 差 (std > 10mm), 标定图片质量不足!\n');
end

%% ===================== 与旧外参对比 =====================

fprintf('\n============ 与旧外参对比 ============\n');
fprintf('旧 T (mm): [%.2f, %.2f, %.2f]\n', T_old);
fprintf('新 T (mm): [%.2f, %.2f, %.2f]\n', T_final);
dT = T_final' - T_old;
fprintf('差异 (mm): [%.2f, %.2f, %.2f]\n', dT);
fprintf('差异范数:  %.2f mm\n', norm(dT));

if norm(dT) < 10
    fprintf('>> 平移与旧值一致 (<10mm), 标定可信!\n');
elseif norm(dT) < 30
    fprintf('>> 平移偏差中等 (10-30mm), 检查标定图片覆盖范围\n');
else
    fprintf('>> 平移偏差过大 (>30mm), 标定质量存疑!\n');
end

%% ===================== 逐对详情 =====================

fprintf('\n============ 逐对估计详情 ============\n');
fprintf('%-5s  %-12s %-12s %-12s  %-10s  %s\n', ...
    'Pair', 'Tx(mm)', 'Ty(mm)', 'Tz(mm)', 'Err(px)', 'Status');
fprintf('%s\n', repmat('-', 1, 70));

for i = 1:nPairs
    if ~isfinite(errors(i))
        status = 'FAILED';
    elseif ~valid(i)
        status = 'OUTLIER';
    else
        status = 'OK';
    end
    fprintf('%-5d  %-12.2f %-12.2f %-12.2f  %-10.3f  %s\n', ...
        i, trans(i,1), trans(i,2), trans(i,3), errors(i), status);
end
