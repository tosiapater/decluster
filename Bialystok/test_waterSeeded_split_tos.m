function out = test_waterSeeded_split_tos (bw_img, gradientOnBW, img, img_fgm)

out = [];

bw_img = logical(bw_img);

B = bwboundaries(bw_img, 8,'noholes');
B_length = cellfun(@length, B);
B_big = B{find(B_length == max(B_length),1)};

s = regionprops(bw_img, 'Centroid', 'Area', 'Perimeter','Solidity', 'Eccentricity','Orientation', 'MajorAxisLength', 'MinorAxisLength', 'PixelList', 'PixelIdxList', 'BoundingBox');
s_max = s( find( ([s.Area]) == max([s.Area]) , 1) );

bw_object = true(size(bw_img));
bw_object(s_max.PixelIdxList) = false;

%% fgm (foreground markers)
% se2 = strel(ones(5,5));
% fgm2 = imclose(imcomplement(bw_object), se2);
% fgm3 = imerode(fgm2, se2);
% fgm3 = imerode(fgm3, se2);
% fgm4 = bwareaopen(fgm3, 20);

fgm4 = img_fgm;

%% bgm (background markers)
    [m1,n1] = size(bw_object);
    expand = 10;
    big_DAB = logical(ones(m1+expand, n1+expand) * 255);
    big_DAB((1+expand/2):(1+expand/2)+m1-1, (1+expand/2):(1+expand/2)+n1-1) = bw_object;

D = bwdist( imcomplement(big_DAB)); %imcomplement
DL = watershed(D);
bgm = DL == 0;

bgm = bgm((1+expand/2):(1+expand/2)+m1-1, (1+expand/2):(1+expand/2)+n1-1);

%figure; imagesc(bgm); colormap('gray'); title('Watershed ridge lines (bgm)')

%% watershed
if gradientOnBW
    [gradmag, Gdir] = imgradient(bw_object,'CentralDifference');
else
    [gradmag, Gdir] = imgradient(img,'CentralDifference');
end
%gradmag2 = imimposemin(gradmag, bgm | fgm4);
gradmag2 = imimposemin(gradmag, bgm | fgm4);
L = watershed(gradmag2);

% out = bw_img;
% out(imdilate(L == 0, ones(3, 3)) | bgm | fgm4) = 255;

% if options.visualize1
%     %figure; imagesc(I4); title('Markers and object boundaries superimposed on original image (I4)')
% Lrgb = label2rgb(L, 'jet', 'w', 'shuffle');
% figure;imagesc(bw_object);hold on;
% himage = imagesc(Lrgb);
% himage.AlphaData = 0.3;
% title('test waterSeeded split')
% 
%     %figure; imshowpair(bw_object, I4, 'montage'); title('Markers and object boundaries & original image (I4)')
% end
% out = L;
out = bw_img;
out( imdilate(L == 0, ones(3, 3)) ) = false;