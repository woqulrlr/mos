% get images from source directory
datadir = '../data/';
dataset = 'Jogging';
path = [datadir dataset];
img_path = [path '/img/'];
D = dir([img_path, '*.jpg']);
seq_len = length(D(not([D.isdir])));
if exist([img_path num2str(1, '%04i.jpg')], 'file'),
    img_files = num2str((1:seq_len)', [img_path '%04i.jpg']);
else
    error('No image files found in the directory.');
end

% select target from first frame
im = imread(img_files(1,:));
f = figure('Name', 'Select object to track'); imshow(im);
%rect = getrect;
%rect_save = rect;
%rect = [37.5943  195.9869   67.2643  112.7666];%lemming rect
%rect = [38.0000  196.0000   70.0000  110.0000];%2-lemming rect
%rect = [299.0000  158.0000   45.0000   85.0000];%coke rect
%rect = [285.6868  159.2692   68.5714   83.3407];%2-coke rect
%rect = [293.7921  161.3655   54.4049   78.1453];%3-%coke
%rect = [301.0000  139.0000   70.0000  101.0000];%4-%coke-frame-243
%rect = [9.0000  144.0000   69.0000  117.0000];%4-%Lemming-frame-265
rect = [104.0000   90.0000   43.0000  114.0000];%Jogging;
close(f); clear f;
center = [rect(2)+rect(4)/2 rect(1)+rect(3)/2];

% plot gaussian
sigma = 100;
gsize = size(im);
[R,C] = ndgrid(1:gsize(1), 1:gsize(2));
g = gaussC(R,C, sigma, center);
g = mat2gray(g);

% randomly warp original image to create training set
if (size(im,3) == 3) 
    img = rgb2gray(im); 
end

%read the first img
train_img = imcrop(img, rect);
g = imcrop(g, rect);
G = fft2(g);
height = size(g,1);
width = size(g,2);
fi = preprocess(imresize(train_img, [height width]));
Ai = (G.*conj(fft2(fi)));
Bi = (fft2(fi).*conj(fft2(fi)));
N = 128;
for i = 1:N
    fi = preprocess(rand_warp(train_img));
    Ai = Ai + (G.*conj(fft2(fi)));
    Bi = Bi + (fft2(fi).*conj(fft2(fi)));
end






% MOSSE online training regimen
%add_heatmap = [];

eta = 0.125;
fig = figure('Name', 'MOSSE');
mkdir(['results_' dataset]);
%for i = 690:size(img_files, 1)

%score_matrix = zeros(14*48,5);
score_matrix = [];

%for i = 1:30:size(img_files, 1)
%for i = 251:2:size(img_files, 1)
for i = 68:2:size(img_files, 1)
    img = imread(img_files(i,:));
    im = img;
    if (size(img,3) == 3)
        img = rgb2gray(img);
    end
    if (i == 68)
        Ai = eta.*Ai;
        Bi = eta.*Bi;   
    else
        %====================================================
        %sliding-windows                          %
        %====================================================
        %Hi = Ai./Bi;
        %from img and mouse
        %rect = [0,0,55,105];
        %size = [860,480];
        imgpixel = size(img,1)*size(img,2);
        targetpixel = rect(3)*rect(4);
        dis_block = 70;
        
        %[labels,numlabels,x_p,y_p] = func_SLICdemo(img,floor(imgpixel/targetpixel),dis_block);
        %[labels,numlabels,x_center_pos,y_center_pos] = func_SLICdemo(img,floor(imgpixel/targetpixel),dis_block);
        [labels,numlabels,x_center_pos,y_center_pos] = func_SLICdemo(img,50,dis_block);
        
        num_of_sift_point = zeros(size(y_center_pos,2),1);
        
        for mos_count = 1:size(y_center_pos,2)
            count = 0;
            response = 0;
            x_p(mos_count) = x_center_pos(mos_count) - floor(rect(3)/2);
            y_p(mos_count) = y_center_pos(mos_count) - floor(rect(4)/2); 
            
            x_detect = x_p(mos_count);
            y_detect = y_p(mos_count);
            
            %boundary detection
            x_detect((x_detect+rect(3)/2)>size(img,2))= size(img,2) - rect(3)/2;
            y_detect((y_detect+rect(4)/2)>size(img,1))= size(img,1) - rect(4)/2;
            x_detect(x_detect<0)= 0;
            y_detect(y_detect<0)= 0;       

            
            x_p(mos_count) = x_detect;
            y_p(mos_count) = y_detect;

        %re-location, count is number of re-location times

            Hi = Ai./Bi;

            fi1 = imcrop(img, [x_p(mos_count),y_p(mos_count),rect(3),rect(4)]);
            fi = preprocess(imresize(fi1, [height width]));
            gi_val = ifft2(Hi.*fft2(fi));
            maxval = max(gi_val(:));
            sub_response(mos_count) = maxval;
          
            %else
                %break;
            %end
                
            
               
            
            
            %Hi = Ai./Bi;
            %x_p((x_p+rect(3)/2)>size(img,2)) = size(img,2) - rect(3)/2;
            %y_p((y_p+rect(4)/2)>size(img,1)) = size(img,1) - rect(4)/2;
            %x_shift = x_p(mos_count) - floor(rect(3)/2);
            %y_shift = y_p(mos_count) - floor(rect(4)/2);
            %x_shift(x_shift<0) = 0;
            %y_shift(y_shift<0) = 0;
            %x_p(mos_count) = x_shift;
            %y_p(mos_count) = y_shift;
            %fi1 = imcrop(img, [x_p(mos_count),y_p(mos_count),rect(3),rect(4)]);
            %fi = preprocess(imresize(fi1, [height width]));
            %=================================
            %search windows  histograms                %
            %=================================
            %histograms before preprocess!!!!
            %fi1 = fi;
            %M=rgb2gray(im);
            im_ora = imread(img_files(1,:));
            fi_ora = imcrop(im_ora, [rect(1),rect(2),rect(3),rect(4)]);
            %I = train_img;
            %I=rgb2gray(M);
            fi2 = imcrop(im, [x_p(mos_count),y_p(mos_count),rect(3),rect(4)]);
            %imwrite(fi2,['/home/lr/Desktop/mosse-tracker-master/src/image/v12/' (num2str(mos_count,'%02i.jpg'))])
            imwrite(fi_ora,['D:\moss\home\lr\Desktop\mosse-tracker-master\src\image\v12\vvv\tmp1.jpg']);
            %sift_target = sift_code();
            imwrite(fi2,['D:\moss\home\lr\Desktop\mosse-tracker-master\src\image\v12\vvv\tmp2.jpg']);
            [sift_target sift_canidate] = sift_code();
            
            img1 = sift_target;
            img2 = sift_canidate;
            count = 0;
            for count_i = 1:size(img1,1)
                list = zeros(1,size(img1,1));
                for count_j = 1:size(img2,1)
                    X = img1(count_i,:);
                    Y = img2(count_j,:);
                    D = pdist2(X,Y,'euclidean');
                    list(count_j) = D;
                end
                s_list = sort(list);
                if s_list(1)/s_list(2) < 0.9
                    count = count +1;
                end
            end
            
            num_of_sift_point(mos_count) = count;
            
            
            
            
            %x1=imread('b1.jpg');
            %y1=imresize(x1,1/200);
%             y1 = fi_ora;
%             y2 = fi2;
%             k1 = [];
%             k2 = [];
%             
%             r1=y1(:,:,1);
%             g1=y1(:,:,2);
%             b1=y1(:,:,3);
%             rr1 = imhist(r1);
%             rr1 = rr1/sum(rr1);
%             gg1 = imhist(g1);
%             gg1 = gg1/sum(gg1);
%             bb1 = imhist(b1);
%             bb1 = bb1/sum(bb1);
%             k1 = cat(1,rr1(:),gg1(:),bb1(:));
%             
%             r2=y2(:,:,1);
%             g2=y2(:,:,2);
%             b2=y2(:,:,3);
%             rr2 = imhist(r2);
%             rr2 = rr2/sum(rr2);
%             gg2 = imhist(g2);
%             gg2 = gg2/sum(gg2);
%             bb2 = imhist(b2);
%             bb2 = bb2/sum(bb2);
%             k2 = cat(1,rr2(:),gg2(:),bb2(:));
%             
%             dd = bhattacharyya(k1,k2);
%             d(mos_count) = dd;
            
            
%             I1 = rgb2gray(fi_ora);
%             I2 = rgb2gray(fi2);
%             corners1 =   detectMinEigenFeatures(I1);
%             corners2 =   detectMinEigenFeatures(I2);
%             [features1, valid_corners] = extractFeatures(I1, corners1);
%             [features2, valid_corners] = extractFeatures(I2, corners2);
%             %indexPairs = matchFeatures(features1,features2,'MaxRatio',0.9);
%             indexPairs = matchFeatures(features1,features2,'MatchThreshold',50);
%             ff(mos_count) = size(indexPairs,1);
            
            
%             features11 = extractLBPFeatures(I1,'Radius',5,'NumNeighbors',24);
%             features22 = extractLBPFeatures(I2,'Radius',5,'NumNeighbors',24);
%             lbp_ponit = matchFeatures(features11,features22);
%             lbp(mos_count) = size(lbp_ponit,1);

        end
        
        score = [];
        num_canidate = size(y_center_pos,2);
        frame_id = i*ones(num_canidate,1);
        score(:,1) = frame_id;
        superpixels_id = [1:num_canidate];
        score(:,2) = superpixels_id;
        score(:,3) = x_p;
        score(:,4) = y_p;   
        %inv_d = 1-d;
        %cut_d = prctile(inv_d,75);
        %inv_d(inv_d<cut_d) = 0;
        %score(:,6) = inv_d;
        %score(:,7) = ff;
        %score(:,8) = lbp;
        
        %standard_res = (sub_response-min(sub_response))/(max(sub_response)-min(sub_response));
        %%standard_ff = (ff-min(ff))/(max(ff)-min(ff));
        %score(:,5) = standard_res;
        %score(:,5) = standard_sift;
        %score(:,6) = standard_d;
        %score(:,7) = standard_ff;
        %score(:,7) = ff
        %vote_score = 0.2*standard_res + 0.6*standard_d + 0.2*ff;
        %vote_score = 0.2*standard_sift + 0.6*standard_d(:)+0.2*standard_ff(:);
        %vote_score = 0.15*standard_sift + 0.85*standard_d(:);
        %standard_d = (max(d)-d)/(max(d)-min(d)); 
        %standard_d = (inv_d-min(inv_d))/(max(inv_d)-min(inv_d));
        standard_sift = (num_of_sift_point-min(num_of_sift_point))/(max(num_of_sift_point)-min(num_of_sift_point));
        standard_r = (sub_response-min(sub_response))/(max(sub_response)-min(sub_response));
        
        
        %vote_score = 0.3*standard_sift + 0.4*standard_d(:) + 0.3*standard_r(:);
        
        score(:,5) = num_of_sift_point;
        score(:,6) = standard_sift;
        score(:,7) = sub_response;
        score(:,8) = standard_r;
        %score_matrix = sortrows(score_matrix,8)
        score_matrix = cat(1,score_matrix,score);
        %sortrows(score_matrix,8)
        fprintf('390');
        
        
       

        
        %Hi = Ai./Bi;
        %fi = imcrop(img, rect); 
        %fi = preprocess(imresize(fi, [height width]));
        %gi = uint8(255*mat2gray(ifft2(Hi.*fft2(fi))));
        %maxval = max(gi(:));
        %[P, Q] = find(gi == maxval);
        %dx = mean(P)-height/2;
        %dy = mean(Q)-width/2;
        
        %rect = [rect(1)+dy rect(2)+dx width height];
        %fi = imcrop(img, rect); 
        %fi = preprocess(imresize(fi, [height width]));
        %Ai = eta.*(G.*conj(fft2(fi))) + (1-eta).*Ai;
        %Bi = eta.*(fft2(fi).*conj(fft2(fi))) + (1-eta).*Bi;
    end
    
    
    
    % visualization
    text_str = ['Frame: ' num2str(i)];
    box_color = 'green';
    position=[1 1];
    result = insertText(im, position,text_str,'FontSize',15,'BoxColor',...
                     box_color,'BoxOpacity',0.4,'TextColor','white');
    result = insertShape(result, 'Rectangle', rect, 'LineWidth', 3);
    imwrite(result, ['results_' dataset num2str(i, '/%04i.jpg')]);
    imshow(result);
end
