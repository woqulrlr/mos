%im = imread(img_files(800,:));
fi1 = imcrop(im, [x_p(mos_count),y_p(mos_count),rect(3),rect(4)]);
figure;imshow(fi1)

[aa bb] = find(d == min(min(d(:))))

for mos_count = 1:size(y_p,2)
    rectangle('Position', [x_p(mos_count),y_p(mos_count),rect(3),rect(4)],'EdgeColor','r');
    xxx = x_p(mos_count);
    yyy = y_p(mos_count);
    xxx = double(xxx);
    yyy = double(yyy);
    text('Position',[xxx yyy],'string',mos_count,'Color','red','FontSize',14);
end


for ci = 1:size(score_matrix)
            score_matrix(ci,9) = score_matrix(ci,6) * score_matrix(ci,8);
end


fi2 = imcrop(im, [x_p(25),y_p(25),rect(3),rect(4)]);
imwrite(fi2,'./image/v12/vvv/i25.jpg')

%img1 feature: f_target
%img2 feature: f_i23
img1 = f_target;
img2 = f_i34;
count = 0;
for i = 1:size(img1,1)
    list = zeros(1,size(img1,1));
    for j = 1:size(img2,1)
        X = img1(i,:);
        Y = img2(j,:);
        D = pdist2(X,Y,'euclidean');
        list(j) = D;
    end
    s_list = sort(list);
    if s_list(1)/s_list(2) < 0.5
        count = count +1;
    end
end


