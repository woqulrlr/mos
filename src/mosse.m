% get images from source directory
datadir = '../data/';
dataset = 'Dudek';
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
%im = imread(img_files(1,:));
im = imread(img_files(1,:));
f = figure('Name', 'Select object to track'); imshow(im);
rect = getrect;
%rect_save = rect;
%rect = [104.0000   90.0000   43.0000  114.0000];%Jogging
%rect = [293.7921  161.3655   54.4049   78.1453];%3-%coke
%rect = [63 239 55 100]%lemming
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
else
    img = im;
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
eta = 0.125;
fig = figure('Name', 'MOSSE');
mkdir(['results_' dataset]);
for i = 1:size(img_files, 1)
%for i = 1:size(img_files, 1)
%for i = 1:72
    img = imread(img_files(i,:));
    im = img;
    if (size(img,3) == 3)
        img = rgb2gray(img);
    end
    if (i == 1)
        Ai = eta.*Ai;
        Bi = eta.*Bi;
    else
        Hi = Ai./Bi;
        fi = imcrop(img, rect); 
        fi = preprocess(imresize(fi, [height width]));
        %gi = uint8(255*mat2gray(ifft2(Hi.*fft2(fi))));
        matrix_tmp = ifft2(Hi.*fft2(fi));
        gi = uint8(255*mat2gray(matrix_tmp));
        
        maxval = max(gi(:));
        %fprintf('###');
        max_sc = max(matrix_tmp(:));
        %fprintf('%f',max_sc);
        %fprintf('###');
        [P, Q] = find(gi == maxval);
        dx = mean(P)-height/2;
        dy = mean(Q)-width/2;
        
        rect = [rect(1)+dy rect(2)+dx width height];
        
        matrix_A = [];
        matrix_A = matrix_tmp;
        p = 0;
        q = 0;
        p = mean(P);%P->Y
        q = mean(Q);
        %a = nzero(11,11);
        %matrix_A(p:p+11,q:q+11) = 0;
        sma_p = p - 4;
        big_p = p + 4;
        sma_q = q - 4;
        big_q = q + 4;

        l = size(matrix_A,1);
        w = size(matrix_A,2);
        
        sidesocre = [];
        co = 1;
        for l = 2:size(matrix_A,1)-1
            for w = 2:size(matrix_A,2)-1
                if(matrix_A(l,w) > matrix_A(l-1,w))
                    if(matrix_A(l,w) > matrix_A(l,w-1))
                        if(matrix_A(l,w) > matrix_A(l+1,w))
                            if(matrix_A(l,w) > matrix_A(l,w+1))
                                if sma_p > l | l > big_p
                                    if sma_q > w | w > big_q
                                        %fprintf('%f  \n',matrix_A(l,w));
                                        sidesocre(co) = matrix_A(l,w);
                                        co = co + 1;
                                    end
                                end
                            end
                        end
                    end
                end
            end
        end
        
        s_mean = mean(sidesocre);
        s_std = std(sidesocre);
        psr = (max_sc - s_mean)/s_std;
        fprintf('psr %f \n',psr);
        
        fi = imcrop(img, rect); 
        fi = preprocess(imresize(fi, [height width]));
        Ai = eta.*(G.*conj(fft2(fi))) + (1-eta).*Ai;
        Bi = eta.*(fft2(fi).*conj(fft2(fi))) + (1-eta).*Bi;
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