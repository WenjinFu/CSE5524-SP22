% CSE 5524 - Project
% Haidong Tian

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% I. Exploratory Data Analysis
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('I. Exploratory Data Analysis\n');

train = readtable('archive/Train.csv');
test = readtable('archive/Test.csv');

for idx = [200, 9800, 20000, 30300]

    im = imread(strcat('archive/', train.Path{idx, 1}));
    X1 = train.Roi_X1(idx) + 1;
    Y1 = train.Roi_Y1(idx) + 1;
    X2 = train.Roi_X2(idx) + 1;
    Y2 = train.Roi_Y2(idx) + 1;

    imagesc(im);
    hold on;
    rectangle('Position', [X1 Y1 X2 - X1 Y2 - Y1], ...
        'EdgeColor', 'r', 'LineWidth', 2);
    hold off;
    axis('image');
    xticks([]);
    yticks([]);
    saveas(gcf, 'outputs/I/' + string(idx) + 'A.png');
    pause;
    close;

    im = im(Y1 : Y2, X1 : X2, :);

    imagesc(im);
    axis('image');
    xticks([]);
    yticks([]);
    saveas(gcf, 'outputs/I/' + string(idx) + 'O.png');
    pause;
    close;

    imY = rgb2gray(im);
    imagesc(imY);
    axis('image');
    colormap('gray');
    xticks([]);
    yticks([]);
    saveas(gcf, 'outputs/I/' + string(idx) + 'Y.png');
    pause;
    close;

    imagesc(im(:, :, 1));
    axis('image');
    colormap('gray');
    xticks([]);
    yticks([]);
    saveas(gcf, 'outputs/I/' + string(idx) + 'R.png');
    pause;
    close;

    imagesc(im(:, :, 2));
    axis('image');
    colormap('gray');
    xticks([]);
    yticks([]);
    saveas(gcf, 'outputs/I/' + string(idx) + 'G.png');
    pause;
    close;

    imagesc(im(:, :, 3));
    axis('image');
    colormap('gray');
    xticks([]);
    yticks([]);
    saveas(gcf, 'outputs/I/' + string(idx) + 'B.png');
    pause;
    close;

end

fprintf('Loading images...\n');

rng(614);

trnSz = 0;
for cls = 0 : 42
    trnSz = trnSz + round(0.8 * sum(train.ClassId == cls));
end

trn = cell(1, trnSz);
vld = cell(1, height(train) - trnSz);
trnCls = zeros(1, trnSz);
vldCls = zeros(1, height(train) - trnSz);
trnNrm = zeros(1, 43);
trnPos = 1;
vldPos = 1;
for class = 0 : 42
    cur = train(train.ClassId == class, :);
    [~, trnIdx] = datasample(cur, round(0.8 * height(cur)), ...
        'replace', false);
    trnNrm(class + 1) = length(trnIdx);
    vldIdx = setdiff(1 : height(cur), trnIdx);
    for idx = trnIdx
        im = imread(strcat('archive/', cur.Path{idx, 1}));
        X1 = cur.Roi_X1(idx) + 1;
        Y1 = cur.Roi_Y1(idx) + 1;
        X2 = cur.Roi_X2(idx) + 1;
        Y2 = cur.Roi_Y2(idx) + 1;
        trn{trnPos} = im(Y1 : Y2, X1 : X2, :);
        trnCls(trnPos) = class;
        trnPos = trnPos + 1;
    end
    for idx = vldIdx
        im = imread(strcat('archive/', cur.Path{idx, 1}));
        X1 = cur.Roi_X1(idx) + 1;
        Y1 = cur.Roi_Y1(idx) + 1;
        X2 = cur.Roi_X2(idx) + 1;
        Y2 = cur.Roi_Y2(idx) + 1;
        vld{vldPos} = im(Y1 : Y2, X1 : X2, :);
        vldCls(vldPos) = class;
        vldPos = vldPos + 1;
    end
end

tst = cell(1, height(test));
tstCls = zeros(1, height(test));
for idx = 1 : height(test)
    im = imread(strcat('archive/', test.Path{idx, 1}));
    X1 = test.Roi_X1(idx) + 1;
    Y1 = test.Roi_Y1(idx) + 1;
    X2 = test.Roi_X2(idx) + 1;
    Y2 = test.Roi_Y2(idx) + 1;
    tst{idx} = im(Y1 : Y2, X1 : X2, :);
    tstCls(idx) = test.ClassId(idx);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% II. Template Matching
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('II. Template Matching\n');

tmps = zeros(100, 100, 3, 43);
tmpRSub = zeros(100, 100, 43);
tmpGsub = zeros(100, 100, 43);
tmpBSub = zeros(100, 100, 43);
tmpRStd = zeros(1, 43);
tmpGStd = zeros(1, 43);
tmpBStd = zeros(1, 43);
for class = 0 : 42
    tmp = double(imread(sprintf('archive/Meta/%d.png', class)));
    [m, ~, ~] = size(tmp);
    tmp = padarray(tmp, ceil((100 - m) / 2), 0, 'pre');
    tmp = padarray(tmp, floor((100 - m) / 2), 0, 'post');
    tmps(:, :, :, class + 1) = tmp;
    tmpRMean = mean(tmp(:, :, 1), 'all');
    tmpGMean = mean(tmp(:, :, 2), 'all');
    tmpBMean = mean(tmp(:, :, 3), 'all');
    tmpRSub(:, :, class + 1) = tmp(:, :, 1) - tmpRMean;
    tmpGSub(:, :, class + 1) = tmp(:, :, 2) - tmpGMean;
    tmpBSub(:, :, class + 1) = tmp(:, :, 3) - tmpBMean;
    tmpRStd(class + 1) = std(tmp(:, :, 1), 0, 'all');
    tmpGStd(class + 1) = std(tmp(:, :, 2), 0, 'all');
    tmpBStd(class + 1) = std(tmp(:, :, 3), 0, 'all');
end

SADPrd = zeros(1, length(tst));
SSDPrd = zeros(1, length(tst));
NCCPrd = zeros(1, length(tst));

SADAcc = zeros(1, 44);
SSDAcc = zeros(1, 44);
NCCAcc = zeros(1, 44);

nrmFac = 3 * (100 * 100 - 1);

for idx = 1 : length(tst)

    im = imresize(double(tst{idx}), [100 100]);

    imRMean = mean(im(:, :, 1), 'all');
    imGMean = mean(im(:, :, 2), 'all');
    imBMean = mean(im(:, :, 3), 'all');
    imRSub = im(:, :, 1) - imRMean;
    imGSub = im(:, :, 2) - imGMean;
    imBSub = im(:, :, 3) - imBMean;
    imRStd = std(im(:, :, 1), 0, 'all');
    imGStd = std(im(:, :, 2), 0, 'all');
    imBStd = std(im(:, :, 3), 0, 'all');

    SADs = zeros(1, 43);
    SSDs = zeros(1, 43);
    NCCs = zeros(1, 43);

    for i = 1 : 43
        SADs(i) = sum(abs(im - tmps(:, :, :, i)), 'all');
        SSDs(i) = sum((im - tmps(:, :, :, i)) .^ 2, 'all');
        NCCs(i) = (sum(tmpRSub(:, :, i) .* imRSub, 'all') / ...
            (tmpRStd(i) * imRStd) + ...
            sum(tmpGSub(:, :, i) .* imGSub, 'all') / ...
            (tmpGStd(i) * imGStd) + ...
            sum(tmpBSub(:, :, i) .* imBSub, 'all') / ...
            (tmpBStd(i) * imBStd)) / nrmFac;
    end

    [~, sa] = min(SADs);
    [~, ss] = min(SSDs);
    [~, nc] = max(NCCs);

    SADPrd(idx) = sa - 1;
    SSDPrd(idx) = ss - 1;
    NCCPrd(idx) = nc - 1;

end

for i = 1 : 43
    SADAcc(i) = sum(SADPrd == i - 1 & tstCls == i - 1) / ...
        sum(tstCls == i - 1);
    SSDAcc(i) = sum(SSDPrd == i - 1 & tstCls == i - 1) / ...
        sum(tstCls == i - 1);
    NCCAcc(i) = sum(NCCPrd == i - 1 & tstCls == i - 1) / ...
        sum(tstCls == i - 1);
end
SADAcc(44) = sum(SADPrd == tstCls) / length(tst);
SSDAcc(44) = sum(SSDPrd == tstCls) / length(tst);
NCCAcc(44) = sum(NCCPrd == tstCls) / length(tst);

plot(0 : 42, SADAcc(1 : 43), ':ro', 'LineWidth', 1);
xlabel('class');
xlim([0 42]);
ylabel('test accuracy');
ylim([0 1]);
title('SAD (overall test accuracy = ' + ...
    string(round(SADAcc(44), 4)) + ')');
saveas(gcf, 'outputs/II/SAD.png');
writematrix(SADAcc, 'outputs/II/SADAcc.csv');
pause;
close;

plot(0 : 42, SSDAcc(1 : 43), ':ro', 'LineWidth', 1);
xlabel('class');
xlim([0 42]);
ylabel('test accuracy');
ylim([0 1]);
title('SSD (overall test accuracy = ' + ...
    string(round(SSDAcc(44), 4)) + ')');
saveas(gcf, 'outputs/II/SSD.png');
writematrix(SSDAcc, 'outputs/II/SSDAcc.csv');
pause;
close;

plot(0 : 42, NCCAcc(1 : 43), ':ro', 'LineWidth', 1);
xlabel('class');
xlim([0 42]);
ylabel('test accuracy');
title('NCC (overall test accuracy = ' + ...
    string(round(NCCAcc(44), 4)) + ')');
saveas(gcf, 'outputs/II/NCC.png');
writematrix(NCCAcc, 'outputs/II/NCCAcc.csv');
pause;
close;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% III. Histogram of Oriented Gradients (HOG)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('III. Histogram of Oriented Gradients (HOG)\n');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% III1. Preliminary Model Selection
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('III1. Preliminary Model Selection\n');

patchNum = 8;
oriNum = 9;
sigmas = [0.25, 0.5, 0.75, 1];

snYCovPrd = zeros(length(sigmas), length(vld));
unYCovPrd = zeros(length(sigmas), length(vld));
snYVecPrd = zeros(length(sigmas), length(vld));
unYVecPrd = zeros(length(sigmas), length(vld));
snVCovPrd = zeros(length(sigmas), length(vld));
unVCovPrd = zeros(length(sigmas), length(vld));
snVVecPrd = zeros(length(sigmas), length(vld));
unVVecPrd = zeros(length(sigmas), length(vld));
snMCovPrd = zeros(length(sigmas), length(vld));
unMCovPrd = zeros(length(sigmas), length(vld));
snMVecPrd = zeros(length(sigmas), length(vld));
unMVecPrd = zeros(length(sigmas), length(vld));
snTVecPrd = zeros(length(sigmas), length(vld));
unTVecPrd = zeros(length(sigmas), length(vld));

snYCovAcc = zeros(1, length(sigmas));
unYCovAcc = zeros(1, length(sigmas));
snYVecAcc = zeros(1, length(sigmas));
unYVecAcc = zeros(1, length(sigmas));
snVCovAcc = zeros(1, length(sigmas));
unVCovAcc = zeros(1, length(sigmas));
snVVecAcc = zeros(1, length(sigmas));
unVVecAcc = zeros(1, length(sigmas));
snMCovAcc = zeros(1, length(sigmas));
unMCovAcc = zeros(1, length(sigmas));
snMVecAcc = zeros(1, length(sigmas));
unMVecAcc = zeros(1, length(sigmas));
snTVecAcc = zeros(1, length(sigmas));
unTVecAcc = zeros(1, length(sigmas));

for i = 1 : length(sigmas)

    [gdx, gdy] = gaussDeriv2D(sigmas(i));

    trnSnYCov = zeros(oriNum, oriNum, 43);
    trnUnYCov = zeros(oriNum, oriNum, 43);
    trnSnYVec = zeros(patchNum * patchNum, oriNum, 43);
    trnUnYVec = zeros(patchNum * patchNum, oriNum, 43);

    trnSnRCov = zeros(oriNum, oriNum, 43);
    trnSnGCov = zeros(oriNum, oriNum, 43);
    trnSnBCov = zeros(oriNum, oriNum, 43);
    trnUnRCov = zeros(oriNum, oriNum, 43);
    trnUnGCov = zeros(oriNum, oriNum, 43);
    trnUnBCov = zeros(oriNum, oriNum, 43);
    trnSnRVec = zeros(patchNum * patchNum, oriNum, 43);
    trnSnGVec = zeros(patchNum * patchNum, oriNum, 43);
    trnSnBVec = zeros(patchNum * patchNum, oriNum, 43);
    trnUnRVec = zeros(patchNum * patchNum, oriNum, 43);
    trnUnGVec = zeros(patchNum * patchNum, oriNum, 43);
    trnUnBVec = zeros(patchNum * patchNum, oriNum, 43);

    fprintf('Training...\n');

    for idx = 1 : length(trn)

        im = trn{idx};

        imY = double(rgb2gray(im));
        imYdx = imfilter(imY, gdx, 'replicate');
        imYdy = imfilter(imY, gdy, 'replicate');
        snYFeaMtx = hogFea(imYdx, imYdy, patchNum, oriNum, 1);
        unYFeaMtx = hogFea(imYdx, imYdy, patchNum, oriNum, 0);
        trnSnYCov(:, :, trnCls(idx) + 1) = ...
            trnSnYCov(:, :, trnCls(idx) + 1) + cov(snYFeaMtx);
        trnUnYCov(:, :, trnCls(idx) + 1) = ...
            trnUnYCov(:, :, trnCls(idx) + 1) + cov(unYFeaMtx);
        trnSnYVec(:, :, trnCls(idx) + 1) = ...
            trnSnYVec(:, :, trnCls(idx) + 1) + snYFeaMtx;
        trnUnYVec(:, :, trnCls(idx) + 1) = ...
            trnUnYVec(:, :, trnCls(idx) + 1) + unYFeaMtx;

        imR = double(im(:, :, 1));
        imRdx = imfilter(imR, gdx, 'replicate');
        imRdy = imfilter(imR, gdy, 'replicate');
        snRFeaMtx = hogFea(imRdx, imRdy, patchNum, oriNum, 1);
        unRFeaMtx = hogFea(imRdx, imRdy, patchNum, oriNum, 0);
        trnSnRCov(:, :, trnCls(idx) + 1) = ...
            trnSnRCov(:, :, trnCls(idx) + 1) + cov(snRFeaMtx);
        trnUnRCov(:, :, trnCls(idx) + 1) = ...
            trnUnRCov(:, :, trnCls(idx) + 1) + cov(unRFeaMtx);
        trnSnRVec(:, :, trnCls(idx) + 1) = ...
            trnSnRVec(:, :, trnCls(idx) + 1) + snRFeaMtx;
        trnUnRVec(:, :, trnCls(idx) + 1) = ...
            trnUnRVec(:, :, trnCls(idx) + 1) + unRFeaMtx;

        imG = double(im(:, :, 2));
        imGdx = imfilter(imG, gdx, 'replicate');
        imGdy = imfilter(imG, gdy, 'replicate');
        snGFeaMtx = hogFea(imGdx, imGdy, patchNum, oriNum, 1);
        unGFeaMtx = hogFea(imGdx, imGdy, patchNum, oriNum, 0);
        trnSnGCov(:, :, trnCls(idx) + 1) = ...
            trnSnGCov(:, :, trnCls(idx) + 1) + cov(snGFeaMtx);
        trnUnGCov(:, :, trnCls(idx) + 1) = ...
            trnUnGCov(:, :, trnCls(idx) + 1) + cov(unGFeaMtx);
        trnSnGVec(:, :, trnCls(idx) + 1) = ...
            trnSnGVec(:, :, trnCls(idx) + 1) + snGFeaMtx;
        trnUnGVec(:, :, trnCls(idx) + 1) = ...
            trnUnGVec(:, :, trnCls(idx) + 1) + unGFeaMtx;

        imB = double(im(:, :, 3));
        imBdx = imfilter(imB, gdx, 'replicate');
        imBdy = imfilter(imB, gdy, 'replicate');
        snBFeaMtx = hogFea(imBdx, imBdy, patchNum, oriNum, 1);
        unBFeaMtx = hogFea(imBdx, imBdy, patchNum, oriNum, 0);
        trnSnBCov(:, :, trnCls(idx) + 1) = ...
            trnSnBCov(:, :, trnCls(idx) + 1) + cov(snBFeaMtx);
        trnUnBCov(:, :, trnCls(idx) + 1) = ...
            trnUnBCov(:, :, trnCls(idx) + 1) + cov(unBFeaMtx);
        trnSnBVec(:, :, trnCls(idx) + 1) = ...
            trnSnBVec(:, :, trnCls(idx) + 1) + snBFeaMtx;
        trnUnBVec(:, :, trnCls(idx) + 1) = ...
            trnUnBVec(:, :, trnCls(idx) + 1) + unBFeaMtx;

    end

    for j = 1 : 43

        trnSnYCov(:, :, j) = trnSnYCov(:, :, j) / trnNrm(j);
        trnUnYCov(:, :, j) = trnUnYCov(:, :, j) / trnNrm(j);
        trnSnYVec(:, :, j) = trnSnYVec(:, :, j) / trnNrm(j);
        trnUnYVec(:, :, j) = trnUnYVec(:, :, j) / trnNrm(j);

        trnSnRCov(:, :, j) = trnSnRCov(:, :, j) / trnNrm(j);
        trnSnGCov(:, :, j) = trnSnGCov(:, :, j) / trnNrm(j);
        trnSnBCov(:, :, j) = trnSnBCov(:, :, j) / trnNrm(j);
        trnUnRCov(:, :, j) = trnUnRCov(:, :, j) / trnNrm(j);
        trnUnGCov(:, :, j) = trnUnGCov(:, :, j) / trnNrm(j);
        trnUnBCov(:, :, j) = trnUnBCov(:, :, j) / trnNrm(j);
        trnSnRVec(:, :, j) = trnSnRVec(:, :, j) / trnNrm(j);
        trnSnGVec(:, :, j) = trnSnGVec(:, :, j) / trnNrm(j);
        trnSnBVec(:, :, j) = trnSnBVec(:, :, j) / trnNrm(j);
        trnUnRVec(:, :, j) = trnUnRVec(:, :, j) / trnNrm(j);
        trnUnGVec(:, :, j) = trnUnGVec(:, :, j) / trnNrm(j);
        trnUnBVec(:, :, j) = trnUnBVec(:, :, j) / trnNrm(j);

    end

    fprintf('Validating...\n');

    for idx = 1 : length(vld)

        im = vld{idx};

        imY = double(rgb2gray(im));
        imYdx = imfilter(imY, gdx, 'replicate');
        imYdy = imfilter(imY, gdy, 'replicate');
        snYFeaMtx = hogFea(imYdx, imYdy, patchNum, oriNum, 1);
        unYFeaMtx = hogFea(imYdx, imYdy, patchNum, oriNum, 0);

        imR = double(im(:, :, 1));
        imRdx = imfilter(imR, gdx, 'replicate');
        imRdy = imfilter(imR, gdy, 'replicate');
        snRFeaMtx = hogFea(imRdx, imRdy, patchNum, oriNum, 1);
        unRFeaMtx = hogFea(imRdx, imRdy, patchNum, oriNum, 0);

        imG = double(im(:, :, 2));
        imGdx = imfilter(imG, gdx, 'replicate');
        imGdy = imfilter(imG, gdy, 'replicate');
        snGFeaMtx = hogFea(imGdx, imGdy, patchNum, oriNum, 1);
        unGFeaMtx = hogFea(imGdx, imGdy, patchNum, oriNum, 0);

        imB = double(im(:, :, 3));
        imBdx = imfilter(imB, gdx, 'replicate');
        imBdy = imfilter(imB, gdy, 'replicate');
        snBFeaMtx = hogFea(imBdx, imBdy, patchNum, oriNum, 1);
        unBFeaMtx = hogFea(imBdx, imBdy, patchNum, oriNum, 0);

        snRmfdY = zeros(1, 43);
        unRmfdY = zeros(1, 43);
        snDistY = zeros(1, 43);
        unDistY = zeros(1, 43);

        snRmfdR = zeros(1, 43);
        snRmfdG = zeros(1, 43);
        snRmfdB = zeros(1, 43);
        unRmfdR = zeros(1, 43);
        unRmfdG = zeros(1, 43);
        unRmfdB = zeros(1, 43);
        snDistR = zeros(1, 43);
        snDistG = zeros(1, 43);
        snDistB = zeros(1, 43);
        unDistR = zeros(1, 43);
        unDistG = zeros(1, 43);
        unDistB = zeros(1, 43);

        snDistT = zeros(1, 43);
        unDistT = zeros(1, 43);

        for k = 1 : 43

            snRmfdY(k) = RMfold(trnSnYCov(:, :, k), cov(snYFeaMtx));
            unRmfdY(k) = RMfold(trnUnYCov(:, :, k), cov(unYFeaMtx));
            snDistY(k) = sum((trnSnYVec(:, :, k) - snYFeaMtx) .^ 2, ...
                'all') ^ 0.5;
            unDistY(k) = sum((trnUnYVec(:, :, k) - unYFeaMtx) .^ 2, ...
                'all') ^ 0.5;

            snRmfdR(k) = RMfold(trnSnRCov(:, :, k), cov(snRFeaMtx));
            snRmfdG(k) = RMfold(trnSnGCov(:, :, k), cov(snGFeaMtx));
            snRmfdB(k) = RMfold(trnSnBCov(:, :, k), cov(snBFeaMtx));
            unRmfdR(k) = RMfold(trnUnRCov(:, :, k), cov(unRFeaMtx));
            unRmfdG(k) = RMfold(trnUnGCov(:, :, k), cov(unGFeaMtx));
            unRmfdB(k) = RMfold(trnUnBCov(:, :, k), cov(unBFeaMtx));
            snDistR(k) = sum((trnSnRVec(:, :, k) - snRFeaMtx) .^ 2, ...
                'all') ^ 0.5;
            snDistG(k) = sum((trnSnGVec(:, :, k) - snGFeaMtx) .^ 2, ...
                'all') ^ 0.5;
            snDistB(k) = sum((trnSnBVec(:, :, k) - snBFeaMtx) .^ 2, ...
                'all') ^ 0.5;
            unDistR(k) = sum((trnUnRVec(:, :, k) - unRFeaMtx) .^ 2, ...
                'all') ^ 0.5;
            unDistG(k) = sum((trnUnGVec(:, :, k) - unGFeaMtx) .^ 2, ...
                'all') ^ 0.5;
            unDistB(k) = sum((trnUnBVec(:, :, k) - unBFeaMtx) .^ 2, ...
                'all') ^ 0.5;

            snDistT(k) = (snDistR(k) ^ 2 + snDistG(k) ^ 2 + ...
                snDistB(k) ^ 2) ^ 0.5;
            unDistT(k) = (unDistR(k) ^ 2 + unDistG(k) ^ 2 + ...
                unDistB(k) ^ 2) ^ 0.5;

        end

        [~, csY] = min(snRmfdY);
        [~, cuY] = min(unRmfdY);
        [~, vsY] = min(snDistY);
        [~, vuY] = min(unDistY);

        [rsR, csR] = min(snRmfdR);
        [rsG, csG] = min(snRmfdG);
        [rsB, csB] = min(snRmfdB);
        [ruR, cuR] = min(unRmfdR);
        [ruG, cuG] = min(unRmfdG);
        [ruB, cuB] = min(unRmfdB);
        [dsR, vsR] = min(snDistR);
        [dsG, vsG] = min(snDistG);
        [dsB, vsB] = min(snDistB);
        [duR, vuR] = min(unDistR);
        [duG, vuG] = min(unDistG);
        [duB, vuB] = min(unDistB);

        [~, vsT] = min(snDistT);
        [~, vuT] = min(unDistT);

        snYCovPrd(i, idx) = csY - 1;
        unYCovPrd(i, idx) = cuY - 1;
        snYVecPrd(i, idx) = vsY - 1;
        unYVecPrd(i, idx) = vuY - 1;

        if csR == csB
            snVCovPrd(i, idx) = csR - 1;
        else
            snVCovPrd(i, idx) = csG - 1;
        end
        if cuR == cuB
            unVCovPrd(i, idx) = cuR - 1;
        else
            unVCovPrd(i, idx) = cuG - 1;
        end
        if vsR == vsB
            snVVecPrd(i, idx) = vsR - 1;
        else
            snVVecPrd(i, idx) = vsG - 1;
        end
        if vuR == vuB
            unVVecPrd(i, idx) = vuR - 1;
        else
            unVVecPrd(i, idx) = vuG - 1;
        end

        if rsG == min([rsR rsG rsB])
            snMCovPrd(i, idx) = csG - 1;
        elseif rsR == min([rsR rsG rsB])
            snMCovPrd(i, idx) = csR - 1;
        else
            snMCovPrd(i, idx) = csB - 1;
        end
        if ruG == min([ruR ruG ruB])
            unMCovPrd(i, idx) = cuG - 1;
        elseif ruR == min([ruR ruG ruB])
            unMCovPrd(i, idx) = cuR - 1;
        else
            unMCovPrd(i, idx) = cuB - 1;
        end
        if dsG == min([dsR dsG dsB])
            snMVecPrd(i, idx) = vsG - 1;
        elseif dsR == min([dsR dsG dsB])
            snMVecPrd(i, idx) = vsR - 1;
        else
            snMVecPrd(i, idx) = vsB - 1;
        end
        if duG == min([duR duG duB])
            unMVecPrd(i, idx) = vuG - 1;
        elseif duR == min([duR duG duB])
            unMVecPrd(i, idx) = vuR - 1;
        else
            unMVecPrd(i, idx) = vuB - 1;
        end

        snTVecPrd(i, idx) = vsT - 1;
        unTVecPrd(i, idx) = vuT - 1;

    end

    snYCovAcc(i) = sum(snYCovPrd(i, :) == vldCls) / length(vldCls);
    unYCovAcc(i) = sum(unYCovPrd(i, :) == vldCls) / length(vldCls);
    snYVecAcc(i) = sum(snYVecPrd(i, :) == vldCls) / length(vldCls);
    unYVecAcc(i) = sum(unYVecPrd(i, :) == vldCls) / length(vldCls);
    snVCovAcc(i) = sum(snVCovPrd(i, :) == vldCls) / length(vldCls);
    unVCovAcc(i) = sum(unVCovPrd(i, :) == vldCls) / length(vldCls);
    snVVecAcc(i) = sum(snVVecPrd(i, :) == vldCls) / length(vldCls);
    unVVecAcc(i) = sum(unVVecPrd(i, :) == vldCls) / length(vldCls);
    snMCovAcc(i) = sum(snMCovPrd(i, :) == vldCls) / length(vldCls);
    unMCovAcc(i) = sum(unMCovPrd(i, :) == vldCls) / length(vldCls);
    snMVecAcc(i) = sum(snMVecPrd(i, :) == vldCls) / length(vldCls);
    unMVecAcc(i) = sum(unMVecPrd(i, :) == vldCls) / length(vldCls);
    snTVecAcc(i) = sum(snTVecPrd(i, :) == vldCls) / length(vldCls);
    unTVecAcc(i) = sum(unTVecPrd(i, :) == vldCls) / length(vldCls);

end

plot(sigmas, snYCovAcc, ':ro', 'LineWidth', 1);
hold on;
plot(sigmas, unYCovAcc, ':bx', 'LineWidth', 1);
hold off;
xlabel('sigma');
xticks(sigmas);
xlim([0.2 1.05]);
ylabel('overall validation accuracy');
legend('signed', 'unsigned');
title('Grayscale HOG Covariance (best validation accuracy = ' + ...
    string(round(max(max(snYCovAcc), max(unYCovAcc)), 4)) + ')');
saveas(gcf, 'outputs/III/III1/YCov.png');
writematrix(snYCovAcc, 'outputs/III/III1/snYCovAcc.csv');
writematrix(unYCovAcc, 'outputs/III/III1/unYCovAcc.csv');
pause;
close;

plot(sigmas, snYVecAcc, ':ro', 'LineWidth', 1);
hold on;
plot(sigmas, unYVecAcc, ':bx', 'LineWidth', 1);
hold off;
xlabel('sigma');
xticks(sigmas);
xlim([0.2 1.05]);
ylabel('overall validation accuracy');
legend('signed', 'unsigned');
title('Grayscale HOG Vector (best validation accuracy = ' + ...
    string(round(max(max(snYVecAcc), max(unYVecAcc)), 4)) + ')');
saveas(gcf, 'outputs/III/III1/YVec.png');
writematrix(snYVecAcc, 'outputs/III/III1/snYVecAcc.csv');
writematrix(unYVecAcc, 'outputs/III/III1/unYVecAcc.csv');
pause;
close;

plot(sigmas, snVCovAcc, ':ro', 'LineWidth', 1);
hold on;
plot(sigmas, unVCovAcc, ':bx', 'LineWidth', 1);
hold off;
xlabel('sigma');
xticks(sigmas);
xlim([0.2 1.05]);
ylabel('overall validation accuracy');
ylim([0.24 0.32]);
legend('signed', 'unsigned');
title('RGB Covariance Voting (best validation accuracy = ' + ...
    string(round(max(max(snVCovAcc), max(unVCovAcc)), 4)) + ')');
saveas(gcf, 'outputs/III/III1/VCov.png');
writematrix(snVCovAcc, 'outputs/III/III1/snVCovAcc.csv');
writematrix(unVCovAcc, 'outputs/III/III1/unVCovAcc.csv');
pause;
close;

plot(sigmas, snVVecAcc, ':ro', 'LineWidth', 1);
hold on;
plot(sigmas, unVVecAcc, ':bx', 'LineWidth', 1);
hold off;
xlabel('sigma');
xticks(sigmas);
xlim([0.2 1.05]);
ylabel('overall validation accuracy');
legend('signed', 'unsigned');
title('RGB Vector Voting (best validation accuracy = ' + ...
    string(round(max(max(snVVecAcc), max(unVVecAcc)), 4)) + ')');
saveas(gcf, 'outputs/III/III1/VVec.png');
writematrix(snVVecAcc, 'outputs/III/III1/snVVecAcc.csv');
writematrix(unVVecAcc, 'outputs/III/III1/unVVecAcc.csv');
pause;
close;

plot(sigmas, snMCovAcc, ':ro', 'LineWidth', 1);
hold on;
plot(sigmas, unMCovAcc, ':bx', 'LineWidth', 1);
hold off;
xlabel('sigma');
xticks(sigmas);
xlim([0.2 1.05]);
ylabel('overall validation accuracy');
ylim([0.21 0.3]);
legend('signed', 'unsigned');
title('RGB Covariance Minimum (best validation accuracy = ' + ...
    string(round(max(max(snMCovAcc), max(unMCovAcc)), 4)) + ')');
saveas(gcf, 'outputs/III/III1/MCov.png');
writematrix(snMCovAcc, 'outputs/III/III1/snMCovAcc.csv');
writematrix(unMCovAcc, 'outputs/III/III1/unMCovAcc.csv');
pause;
close;

plot(sigmas, snMVecAcc, ':ro', 'LineWidth', 1);
hold on;
plot(sigmas, unMVecAcc, ':bx', 'LineWidth', 1);
hold off;
xlabel('sigma');
xticks(sigmas);
xlim([0.2 1.05]);
ylabel('overall validation accuracy');
legend('signed', 'unsigned');
title('RGB Vector Minimum (best validation accuracy = ' + ...
    string(round(max(max(snMVecAcc), max(unMVecAcc)), 4)) + ')');
saveas(gcf, 'outputs/III/III1/MVec.png');
writematrix(snMVecAcc, 'outputs/III/III1/snMVecAcc.csv');
writematrix(unMVecAcc, 'outputs/III/III1/unMVecAcc.csv');
pause;
close;

plot(sigmas, snTVecAcc, ':ro', 'LineWidth', 1);
hold on;
plot(sigmas, unTVecAcc, ':bx', 'LineWidth', 1);
hold off;
xlabel('sigma');
xticks(sigmas);
xlim([0.2 1.05]);
ylabel('overall validation accuracy');
legend('signed', 'unsigned');
title('RGB Vector Total (best validation accuracy = ' + ...
    string(round(max(max(snTVecAcc), max(unTVecAcc)), 4)) + ')');
saveas(gcf, 'outputs/III/III1/TVec.png');
writematrix(snTVecAcc, 'outputs/III/III1/snTVecAcc.csv');
writematrix(unTVecAcc, 'outputs/III/III1/unTVecAcc.csv');
pause;
close;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% III2. Model Selection in Grayscale
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('III2. Model Selection in Grayscale\n');

[gdx, gdy] = gaussDeriv2D(0.75);
patchNums = [4, 8, 12];
oriNums = [6, 9, 12, 15];

snYCovAcc = zeros(length(patchNums), length(oriNums));
unYVecAcc = zeros(length(patchNums), length(oriNums));

for r = 1 : length(patchNums)
    for c = 1 : length(oriNums)

        patchNum = patchNums(r);
        oriNum = oriNums(c);
        
        snYCovPrd = zeros(1, length(vld));
        unYVecPrd = zeros(1, length(vld));

        trnSnYCov = zeros(oriNum, oriNum, 43);
        trnUnYVec = zeros(patchNum * patchNum, oriNum, 43);

        fprintf('Training...\n');

        for idx = 1 : length(trn)
    
            im = trn{idx};
    
            imY = double(rgb2gray(im));
            imYdx = imfilter(imY, gdx, 'replicate');
            imYdy = imfilter(imY, gdy, 'replicate');
            snYFeaMtx = hogFea(imYdx, imYdy, patchNum, oriNum, 1);
            unYFeaMtx = hogFea(imYdx, imYdy, patchNum, oriNum, 0);
            trnSnYCov(:, :, trnCls(idx) + 1) = ...
                trnSnYCov(:, :, trnCls(idx) + 1) + cov(snYFeaMtx);
            trnUnYVec(:, :, trnCls(idx) + 1) = ...
                trnUnYVec(:, :, trnCls(idx) + 1) + unYFeaMtx;

        end

        for j = 1 : 43

            trnSnYCov(:, :, j) = trnSnYCov(:, :, j) / trnNrm(j);
            trnUnYVec(:, :, j) = trnUnYVec(:, :, j) / trnNrm(j);

        end

        fprintf('Validating...\n');

        for idx = 1 : length(vld)
    
            im = vld{idx};
    
            imY = double(rgb2gray(im));
            imYdx = imfilter(imY, gdx, 'replicate');
            imYdy = imfilter(imY, gdy, 'replicate');
            snYFeaMtx = hogFea(imYdx, imYdy, patchNum, oriNum, 1);
            unYFeaMtx = hogFea(imYdx, imYdy, patchNum, oriNum, 0);

            snRmfdY = zeros(1, 43);
            unDistY = zeros(1, 43);

            for k = 1 : 43

                snRmfdY(k) = RMfold(trnSnYCov(:, :, k), cov(snYFeaMtx));
                unDistY(k) = sum((trnUnYVec(:, :, k) - unYFeaMtx) .^ 2, ...
                    'all') ^ 0.5;

            end

            [~, csY] = min(snRmfdY);
            [~, vuY] = min(unDistY);

            snYCovPrd(idx) = csY - 1;
            unYVecPrd(idx) = vuY - 1;

        end

        snYCovAcc(r, c) = sum(snYCovPrd == vldCls) / length(vldCls);
        unYVecAcc(r, c) = sum(unYVecPrd == vldCls) / length(vldCls);

    end
end

plot(oriNums, snYCovAcc(1, :), ':ro', 'LineWidth', 1);
hold on;
plot(oriNums, snYCovAcc(2, :), ':bx', 'LineWidth', 1);
plot(oriNums, snYCovAcc(3, :), ':k^', 'LineWidth', 1);
hold off;
xlabel('oriNum');
xticks(oriNums);
xlim([5 16]);
ylabel('overall validation accuracy');
ylim([0.12 0.42]);
legend('patchNum = ' + string(patchNums(1)), ...
    'patchNum = ' + string(patchNums(2)), ...
    'patchNum = ' + string(patchNums(3)));
title('Signed HOG Covariance (best validation accuracy = ' + ...
    string(round(max(snYCovAcc, [], 'all'), 4)) + ')');
saveas(gcf, 'outputs/III/III2/snYCov.png');
writematrix(snYCovAcc, 'outputs/III/III2/snYCovAcc.csv');
pause;
close;

plot(oriNums, unYVecAcc(1, :), ':ro', 'LineWidth', 1);
hold on;
plot(oriNums, unYVecAcc(2, :), ':bx', 'LineWidth', 1);
plot(oriNums, unYVecAcc(3, :), ':k^', 'LineWidth', 1);
hold off;
xlabel('oriNum');
xticks(oriNums);
xlim([5 16]);
ylabel('overall validation accuracy');
ylim([0.6 0.9]);
legend('patchNum = ' + string(patchNums(1)), ...
    'patchNum = ' + string(patchNums(2)), ...
    'patchNum = ' + string(patchNums(3)));
title('Unsigned HOG Vector (best validation accuracy = ' + ...
    string(round(max(unYVecAcc, [], 'all'), 4)) + ')');
saveas(gcf, 'outputs/III/III2/snYVec.png');
writematrix(unYVecAcc, 'outputs/III/III2/unYVecAcc.csv');
pause;
close;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% III3. Test Results
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('III3. Test Results\n');

fprintf('Signed HOG Covariance\n');

[gdx, gdy] = gaussDeriv2D(0.75);
patchNum = 12;
oriNum = 15;

trnSnYCov = zeros(oriNum, oriNum, 43);

fprintf('Training...\n');

for idx = 1 : length(trn)

    im = trn{idx};

    imY = double(rgb2gray(im));
    imYdx = imfilter(imY, gdx, 'replicate');
    imYdy = imfilter(imY, gdy, 'replicate');
    snYFeaMtx = hogFea(imYdx, imYdy, patchNum, oriNum, 1);
    trnSnYCov(:, :, trnCls(idx) + 1) = ...
                trnSnYCov(:, :, trnCls(idx) + 1) + cov(snYFeaMtx);

end

for idx = 1 : length(vld)

    im = vld{idx};

    imY = double(rgb2gray(im));
    imYdx = imfilter(imY, gdx, 'replicate');
    imYdy = imfilter(imY, gdy, 'replicate');
    snYFeaMtx = hogFea(imYdx, imYdy, patchNum, oriNum, 1);
    trnSnYCov(:, :, vldCls(idx) + 1) = ...
                trnSnYCov(:, :, vldCls(idx) + 1) + cov(snYFeaMtx);

end

for j = 1 : 43
    trnSnYCov(:, :, j) = trnSnYCov(:, :, j) / sum(train.ClassId == j - 1);
end

snYCovAcc = zeros(1, 44);
snYCovPrd = zeros(1, length(tst));

fprintf('Testing...\n');

for idx = 1 : length(tst)

    im = tst{idx};
    
    imY = double(rgb2gray(im));
    imYdx = imfilter(imY, gdx, 'replicate');
    imYdy = imfilter(imY, gdy, 'replicate');
    snYFeaMtx = hogFea(imYdx, imYdy, patchNum, oriNum, 1);

    snRmfdY = zeros(1, 43);

    for k = 1 : 43
        snRmfdY(k) = RMfold(trnSnYCov(:, :, k), cov(snYFeaMtx));
    end

    [~, csY] = min(snRmfdY);
    snYCovPrd(idx) = csY - 1;

end

for i = 1 : 43
    snYCovAcc(i) = sum(snYCovPrd == i - 1 & tstCls == i - 1) / ...
        sum(tstCls == i - 1);
end
snYCovAcc(44) = sum(snYCovPrd == tstCls) / length(tst);

plot(0 : 42, snYCovAcc(1 : 43), ':ro', 'LineWidth', 1);
xlabel('class');
xlim([0 42]);
ylabel('test accuracy');
ylim([0 1]);
title('Signed HOG Covariance (overall test accuracy = ' + ...
    string(round(snYCovAcc(44), 4)) + ')');
saveas(gcf, 'outputs/III/III3/snYCov.png');
writematrix(snYCovAcc, 'outputs/III/III3/snYCovAcc.csv');
pause;
close;

fprintf('Unsigned HOG Vector\n');

[gdx, gdy] = gaussDeriv2D(0.75);
patchNum = 12;
oriNum = 6;

trnUnYVec = zeros(patchNum * patchNum, oriNum, 43);

fprintf('Training...\n');

for idx = 1 : length(trn)

    im = trn{idx};

    imY = double(rgb2gray(im));
    imYdx = imfilter(imY, gdx, 'replicate');
    imYdy = imfilter(imY, gdy, 'replicate');
    unYFeaMtx = hogFea(imYdx, imYdy, patchNum, oriNum, 0);
    trnUnYVec(:, :, trnCls(idx) + 1) = ...
                trnUnYVec(:, :, trnCls(idx) + 1) + unYFeaMtx;

end

for idx = 1 : length(vld)

    im = vld{idx};

    imY = double(rgb2gray(im));
    imYdx = imfilter(imY, gdx, 'replicate');
    imYdy = imfilter(imY, gdy, 'replicate');
    unYFeaMtx = hogFea(imYdx, imYdy, patchNum, oriNum, 0);
    trnUnYVec(:, :, vldCls(idx) + 1) = ...
                trnUnYVec(:, :, vldCls(idx) + 1) + unYFeaMtx;

end

for j = 1 : 43
    trnUnYVec(:, :, j) = trnUnYVec(:, :, j) / sum(train.ClassId == j - 1);
end

unYVecAcc = zeros(1, 44);
unYVecPrd = zeros(1, length(tst));

fprintf('Testing...\n');

for idx = 1 : length(tst)

    im = tst{idx};
    
    imY = double(rgb2gray(im));
    imYdx = imfilter(imY, gdx, 'replicate');
    imYdy = imfilter(imY, gdy, 'replicate');
    unYFeaMtx = hogFea(imYdx, imYdy, patchNum, oriNum, 0);

    unDistY = zeros(1, 43);

    for k = 1 : 43
        unDistY(k) = sum((trnUnYVec(:, :, k) - unYFeaMtx) .^ 2, ...
            'all') ^ 0.5;
    end

    [~, vuY] = min(unDistY);
    unYVecPrd(idx) = vuY - 1;

end

for i = 1 : 43
    unYVecAcc(i) = sum(unYVecPrd == i - 1 & tstCls == i - 1) / ...
        sum(tstCls == i - 1);
end
unYVecAcc(44) = sum(unYVecPrd == tstCls) / length(tst);

plot(0 : 42, unYVecAcc(1 : 43), ':ro', 'LineWidth', 1);
xlabel('class');
xlim([0 42]);
ylabel('test accuracy');
title('Unsigned HOG Vector (overall test accuracy = ' + ...
    string(round(unYVecAcc(44), 4)) + ')');
saveas(gcf, 'outputs/III/III3/unYVec.png');
writematrix(snYCovAcc, 'outputs/III/III3/unYVecAcc.csv');
pause;
close;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Functions Defined Below
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Gaussian Derivative Mask
function [gdx, gdy] = gaussDeriv2D(sigma)
    gdsize = 2 * ceil(3 * sigma) + 1;
    dm = ceil(gdsize / 2);
    gdx = zeros(gdsize, gdsize);
    gdy = zeros(gdsize, gdsize);
    for r = 1 : gdsize
        for c = 1 : gdsize
            gdx(r, c) = exp(-((r - dm) ^ 2 + (c - dm) ^ 2) / ...
                (2 * sigma ^ 2)) * (c - dm) / (2 * pi * sigma ^ 4);
            gdy(r, c) = exp(-((r - dm) ^ 2 + (c - dm) ^ 2) / ...
                (2 * sigma ^ 2)) * (r - dm) / (2 * pi * sigma ^ 4);
        end
    end
    gdx = gdx / sum(abs(gdx), 'all');
    gdy = gdy / sum(abs(gdy), 'all');
end

% HOG Features
function feaMtx = hogFea(imdx, imdy, patchNum, oriNum, signed)
    [R, C] = size(imdx);
    rStarts = round((0 : patchNum - 1) / patchNum * R) + 1;
    rEnds = round((1 : patchNum) / patchNum * R);
    cStarts = round((0 : patchNum - 1) / patchNum * C) + 1;
    cEnds = round((1 : patchNum) / patchNum * C);
    feaMtx = zeros(patchNum * patchNum, oriNum);
    if signed
        ori = atan2(imdy, imdx);
        for r = 1 : patchNum
            for c = 1 : patchNum
                cur = ori(rStarts(r) : rEnds(r), cStarts(c) : cEnds(c));
                idx = (r - 1) * patchNum + c;
                for i = 1 : oriNum
                    feaMtx(idx, i) = sum(...
                        cur > (2 * (i - 1) / oriNum - 1) * pi & ...
                        cur <= (2 * i  / oriNum - 1) * pi, 'all');
                end
                feaMtx(idx, oriNum) = feaMtx(idx, oriNum) + ...
                    sum(cur == -pi, 'all');
            end
        end
    else
        ori = atan(imdy ./ imdx);
        for r = 1 : patchNum
            for c = 1 : patchNum
                cur = ori(rStarts(r) : rEnds(r), cStarts(c) : cEnds(c));
                idx = (r - 1) * patchNum + c;
                for i = 1 : oriNum
                    feaMtx(idx, i) = sum(...
                        cur > ((i - 1) / oriNum - 0.5) * pi & ...
                        cur <= (i / oriNum - 0.5) * pi, 'all');
                end
                feaMtx(idx, oriNum) = feaMtx(idx, oriNum) + ...
                    sum(cur == -pi / 2, 'all');
            end
        end
    end
    feaMtx = feaMtx / (sum(feaMtx .^ 2, 'all') ^ 0.5);
end

% Riemannian Manifold
function Rho = RMfold(C1, C2)
    [~, D] = eig(C1, C2);
    Rho = (sum(log(diag(D)) .^ 2)) ^ 0.5;
end