%img1 feature: f_target
%img2 feature: f_i23
img1 = i_target;
img2 = i34;
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
    if s_list(1)/s_list(2) < 0.9
        count = count + 1;
    end
    
end
fprintf('ponit num : %d \n',count);