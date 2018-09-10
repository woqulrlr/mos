condition = 0
while(condition<3)
    condition = condition+1
end


fi2 = imcrop(im, [x_p(mos_count),y_p(mos_count),rect(3),rect(4)]);
figure;imshow(fi2)