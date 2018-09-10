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
% Select a mid frame and do forward and backward tracking

%im = imread(img_files(1,:));
spec_frame = 92;
im = imread(img_files(spec_frame,:));
f = figure('Name', 'Select object to track'); imshow(im);
%rect = getrect;
%rect = [315.0000   89.0000   69.0000  117.0000];coke
%rect = [346.0000  156.0000   54.4049   78.1453];%lemming
rect = [116    99    43   114];%Jogging
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
img = imcrop(img, rect);
g = imcrop(g, rect);
G = fft2(g);
height = size(g,1);
width = size(g,2);
fi = preprocess(imresize(img, [height width]));
Ai = (G.*conj(fft2(fi)));
Bi = (fft2(fi).*conj(fft2(fi)));
N = 128;
for i = 1:N
    fi = preprocess(rand_warp(img));
    Ai = Ai + (G.*conj(fft2(fi)));
    Bi = Bi + (fft2(fi).*conj(fft2(fi)));
end

% MOSSE online training regimen
file_num = size(img_files,1);
invese_rect = zeros(file_num,4);

eta = 0.125;
fig = figure('Name', 'MOSSE');
mkdir(['results_' dataset]);
Ai1 = Ai;
Ai2 = Ai;
Bi1 = Bi;
Bi2 = Bi;
rect_back = rect;
rect_forward = rect;




for cc = spec_frame:-1:1
    %invese i
    %ci = spec_frame-cc+1;
    imgg = imread(img_files(cc,:));
    im = imgg;
    if (size(imgg,3) == 3)
        imgg = rgb2gray(imgg);
    end
    if (cc == spec_frame)
        Ai2 = eta.*Ai2;
        Bi2 = eta.*Bi2;
    else
        Hi2 = Ai2./Bi2;
        fi = imcrop(imgg,rect_forward); 
        fi = preprocess(imresize(fi, [height width]));
        gi = uint8(255*mat2gray(ifft2(Hi2.*fft2(fi))));
        maxval = max(gi(:));
        [P, Q] = find(gi == maxval);
        dx = mean(P)-height/2;
        dy = mean(Q)-width/2;
        
        aaa = rect_forward(1)+dy;
        bbb = rect_forward(2)+dx;
        aaa(aaa>640) = 640 - height/2;
        bbb(bbb>480) = 480 - width/2;
        %rect = [rect(1)+dy rect(2)+dx width height];
        rect_forward = [aaa bbb width height];
        fi = imcrop(imgg, rect_forward); 
        fi = preprocess(imresize(fi, [height width]));
        Ai2 = eta.*(G.*conj(fft2(fi))) + (1-eta).*Ai2;
        Bi2 = eta.*(fft2(fi).*conj(fft2(fi))) + (1-eta).*Bi2;
        invese_rect(cc,:) = rect_forward;
        fprintf('tacking in %d frame \n',cc);
    end
end

for i = spec_frame:size(img_files, 1)
    img = imread(img_files(i,:));
    im = img;
    if (size(img,3) == 3)
        img = rgb2gray(img);
    end
    if (i == spec_frame)
        Ai1 = eta.*Ai1;
        Bi1 = eta.*Bi1;
    else
        Hi1 = Ai1./Bi1;
        fi1 = imcrop(img, rect_back);%rect_back 
        fi1 = preprocess(imresize(fi1, [height width]));
        gi = uint8(255*mat2gray(ifft2(Hi1.*fft2(fi1))));
        maxval = max(gi(:));
        [P, Q] = find(gi == maxval);
        dx1 = mean(P)-height/2;
        dy1 = mean(Q)-width/2;
        
        rect_back = [rect_back(1)+dy1 rect_back(2)+dx1 width height];
        fi1 = imcrop(img, rect_back); 
        fi1 = preprocess(imresize(fi1, [height width]));
        Ai1 = eta.*(G.*conj(fft2(fi1))) + (1-eta).*Ai1;
        Bi1 = eta.*(fft2(fi1).*conj(fft2(fi1))) + (1-eta).*Bi1;
        invese_rect(i,:) = rect_back;
        fprintf('tacking in %d frame \n',i);    
        
        
    end
end

%for i=32:-1:1    
%for cc = 1:spec_frame


for j = 1:file_num
    % visualization
    im_v = imread(img_files(j,:));
    text_str = ['Frame: ' num2str(j)];
    box_color = 'green';
    position=[1 1];
    result = insertText(im_v, position,text_str,'FontSize',15,'BoxColor',...
                     box_color,'BoxOpacity',0.4,'TextColor','white');
    result = insertShape(result, 'Rectangle', invese_rect(j,:), 'LineWidth', 3);
    imwrite(result, ['results_' dataset num2str(j, '/%04i.jpg')]);
    imshow(result);
end