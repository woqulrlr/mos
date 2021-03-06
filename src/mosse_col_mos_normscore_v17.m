% get images from source directory
datadir = '../data/';
dataset = 'Lemming';
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
rect = [37.5943  195.9869   67.2643  112.7666];
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
for i = 690:size(img_files, 1)
    img = imread(img_files(i,:));
    im = img;
    if (size(img,3) == 3)
        img = rgb2gray(img);
    end
    if (i == 690)
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
        [labels,numlabels,x_center_pos,y_center_pos] = func_SLICdemo(img,floor(imgpixel/targetpixel),dis_block);
        
        
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
            while(count < 4)
                count = count + 1;
                Hi = Ai./Bi;
                
                fi1 = imcrop(img, [x_p(mos_count),y_p(mos_count),rect(3),rect(4)]);
                fi = preprocess(imresize(fi1, [height width]));
                gi_val = ifft2(Hi.*fft2(fi));
                maxval = max(gi_val(:));
                if response < maxval
                    sub_response(mos_count) = maxval;
                    response = maxval;
                %re-mosse again
                    [P, Q] = find(gi_val == maxval);
                    dx = mean(P)-height/2;
                    dy = mean(Q)-width/2; 
                    x_p(mos_count) = x_p(mos_count) + dy;
                    y_p(mos_count) = y_p(mos_count) + dx;
                end
                %else
                    %break;
                %end
                
            end
               
            
            
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

            %x1=imread('b1.jpg');
            %y1=imresize(x1,1/200);
            y1 = fi_ora;
            y2 = fi2;
            k1 = [];
            k2 = [];
            
            r1=y1(:,:,1);
            g1=y1(:,:,2);
            b1=y1(:,:,3);
            rr1 = imhist(r1);
            rr1 = rr1/sum(rr1);
            gg1 = imhist(g1);
            gg1 = gg1/sum(gg1);
            bb1 = imhist(b1);
            bb1 = bb1/sum(bb1);
            k1 = cat(1,rr1(:),gg1(:),bb1(:));
            
            r2=y2(:,:,1);
            g2=y2(:,:,2);
            b2=y2(:,:,3);
            rr2 = imhist(r2);
            rr2 = rr2/sum(rr2);
            gg2 = imhist(g2);
            gg2 = gg2/sum(gg2);
            bb2 = imhist(b2);
            bb2 = bb2/sum(bb2);
            k2 = cat(1,rr2(:),gg2(:),bb2(:));
            
            dd = bhattacharyya(k1,k2);
            d(mos_count) = dd;
            
           
   
            
        end
        
        [B I] = sort(sub_response(:));
        inve_B = flipud(B);
        inve_I = flipud(I);
        %d_2 = sort(d(:));
        new_arr = zeros(size(y_p,2),3);
        new_arr_2(:,2) = inve_B;
        %new_arr_2(:,3) = d_2;
        new_arr_2(:,1) = inve_I;
        for iii = 1:size(y_p,2)
            tmp = new_arr_2(iii,1);
            new_arr_2(iii,3) = d(tmp);
        end
        
        new_arr_2 = sortrows(new_arr_2,3);
        
        
        %norm_data_col = (max(col_score)-col_score+min(col_score))/( max(col_score)-min(col_score) )
        %norm_data_col = (col_score - min(col_score)) / ( max(col_score) - min(col_score) )
        
        aa = new_arr_2(:,2);    
        bb = new_arr_2(:,3);
        score1 = (aa-min(aa))/(max(aa)-min(aa));
        score2 = (max(bb)-bb)/(max(bb)-min(bb));
        new_col = score1 + score2;
        new_arr_2 = [new_arr_2 new_col];
        new_arr_2 = sortrows(new_arr_2,4);
        inve_new_arr_2 = flipud(new_arr_2);
        
        
        
        
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
