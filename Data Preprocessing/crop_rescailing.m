files = dir('C:/Users/kwon/Mobile_Mask_RCNN/data/deficiency/label/train/*.jpg');
scale = 1000;
cnt =0;

for i = 1:length(files)
    img = imread(['C:/Users/kwon/Mobile_Mask_RCNN/data/deficiency/label/train/' files(i).name]);
    [height, width, channel] = size(img);
    fprintf('original image: %.f %f.\n', height, width);
    imshow(img);
    
    % User click the left and right column to measure column width
    [x1,y1,btn] = ginput(1);
    [x2,y2,btn] = ginput(1);
    
    col_width = round(pdist([x1 x2; y1 y2], 'euclidean'));
    img2 = imresize(img, (scale/col_width));
    
    [new_height, new_width, new_channel] = size(img2);
    fprintf('rescaled image: %f. %f.\n',new_height, new_width);
    imshow(img2);
    
    while true
        [x,y,btn] = ginput(1);
        
        if btn == 3
            break;
        end
        
        xi = round(x);
        yi = round(y);
        
        hold on        
        plot([xi-32+1, xi+32, xi+32, xi-32+1, xi-32+1], [yi-32+1, yi-32+1, yi+32, yi+32, yi-32+1]);
        hold off
        
        patch = img2(yi-32+1:yi+32, xi-32+1:xi+32,:);
        imwrite(patch, sprintf('C:/Users/kwon/Mobile_Mask_RCNN/data/deficiency/temp/%05d_jang.jpg', cnt));
        cnt = cnt+1;
    end
end