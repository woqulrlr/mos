im1 = img;
        im1 = preprocess(im1);
        hh = ifft2(Hi);
        [ma , na] = size(im1);
        [mb , nb] = size(hh);
        
        hh(ma*mb-1, na*nb-1) = 0;
        im1(ma*mb-1 , na*nb-1) = 0;
        
        c = iff2(fft2(hh).*fft2(im1));
        c1 = c(1:ma+mb-1 , 1:na+nb-1);
        c1 = uint8(255*mat2gray(c1));
        
        
        %================================
        % 3D gauss
        %================================
        if(isequal(c2,c3))
            c2 = c1;
        else
            c2 = c2 + c1;
        end