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
        Hi = Ai./Bi;
        I = img;
        I = preprocess(I);
        Hi1 = fftshift(Hi);
        [img_r,img_c] = size(img);
        [f_r,f_c] = size(Hi);
        
        c1 = floor((img_c-f_c)/2);
        c2 = img_c - c1 - f_c;
        
        r1 = floor((img_r-f_r)/2);
        r2 = img_r - r1 - f_r;
        
        zero_c1 = zeros(f_r,c1);
        zero_c2 = zeros(f_r,c2);
        Hi2 = [zero_c1,Hi1,zero_c2];
        
        zero_r1 = zeros(r1,img_c);
        zero_r2 = zeros(r2,img_c);
        Hi3 = [zero_r1;Hi2;zero_r2];
        
        Hi4 = fftshift(Hi3);
        
        global_heatmap = uint8(255*mat2gray(real(ifft2(Hi4.*fft2(I),img_r,img_c))));
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
