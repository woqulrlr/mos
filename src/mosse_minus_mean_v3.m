% get images from source directory
datadir = '../data/';
dataset = 'Coke';
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
rect = getrect;
%rect = [42,212,55,105];
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
for i = 287:size(img_files, 1)
    img = imread(img_files(i,:));
    im = img;
    if (size(img,3) == 3)
        img = rgb2gray(img);
    end
    if (i == 287)
        Ai = eta.*Ai;
        Bi = eta.*Bi;      
    else
        %====================================================
        %sliding-windows
        %====================================================
        %Hi = Ai./Bi;
        %from img and mouse
        %rect = [0,0,55,105];
        %size = [860,480];
        rect_box = [0,0,rect(3),rect(4)];

        %overlap 
        %overlap(1):height  ,   overlap(2):width
        overlap_parameter = 2;
        overlap = [floor(rect(3)/overlap_parameter),floor(rect(4)/overlap_parameter)];

        %steps of sliding windows
        steps = [ceil(size(img,1)/overlap(2)),ceil(size(img,2)/overlap(1))];

        %product sliding windows
        %subwindows matrix:col = n-th,cow = condition and box size
        subwindows = zeros(steps(1)*steps(2),4);
        sub_response = zeros(steps(2),steps(1));
        
        count = 1;
        for num_height = 1:steps(1)
            for num_width = 1:steps(2)
                sub_window_x = rect_box(1)+(num_width-1)*overlap(1);
                sub_window_y = rect_box(2)+(num_height-1)*overlap(2);
                %detect boundary
                sub_window_x((sub_window_x+rect(3))>size(img,2)) = size(img,2) - rect(3);
                sub_window_y((sub_window_y+rect(4))>size(img,1)) = size(img,1) - rect(4);
                subwindows(count,:) = [sub_window_x,sub_window_y,rect_box(3),rect_box(4)];
                
                Hi = Ai./Bi;
                fi = imcrop(img, subwindows(count,:)); 
                fi = preprocess(imresize(fi, [height width]));
                %gi = uint8(255*mat2gray(ifft2(Hi.*fft2(fi))));
                gi_val = ifft2(Hi.*fft2(fi));
                maxval = max(gi_val(:));
                sub_response(count) = maxval;
                
                
                count = count +1; 
            end
        end
        
        if i == 288
            add_heatmap = zeros(steps(2),steps(1));
        end
        res_mean = mean(sub_response(:));
        sub_response(sub_response<res_mean) = 0;
        
        add_heatmap = add_heatmap+sub_response;
        
        
        if i == 281
            fprintf('390');
        end

        
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
