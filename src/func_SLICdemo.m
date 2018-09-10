function [labels,numlabels,kx,ky] = func_SLICdemo(img,ii,jj)

%img = imread('0410.jpg');
[labels,numlabels,kx,ky] = slicmex(img,ii,jj);%numlabels is the same as number of superpixels


i = size(kx);
i = i(1,2);

for j=1:i
    xx = kx(j);
    yy = ky(j);
    labels(yy,xx) = 50;
end

end