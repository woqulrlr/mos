function [canidate1 canidate2] = sift_code()
addpath('sift');

img_dir = 'image';                  
data_dir = 'data';                  
dataSet = 'v12';
sift_result_dir = 'siftResult';     

rt_img_dir = fullfile(img_dir, dataSet);   
rt_data_dir = fullfile(data_dir, dataSet); 
rt_sift_result_dir = fullfile(sift_result_dir, dataSet);
          
gridSpacing = 12;                   
patchSize = 16;                     
maxImSize = 300;                    
nrml_threshold = 1; 

[database, im_info, sift_vector] = CalculateSiftDescriptor(10,rt_img_dir, rt_data_dir, rt_sift_result_dir, gridSpacing, patchSize, maxImSize, nrml_threshold);

canidate1 = im_info(1001).patches;
canidate2 = im_info(1002).patches;
%target = im_info(1002).patches

%vpath = fullfile(rt_sift_result_dir,'database.mat');
%save(vpath,'database');    

%vpath = fullfile(rt_sift_result_dir,'im_info.mat');
%save(vpath,'im_info');   

%vpath = fullfile(rt_sift_result_dir,'sift_vector.mat');
%save(vpath,'sift_vector');

%test2 = im_info(1002).patches
%indexPairs = matchFeatures(test1,test2)

