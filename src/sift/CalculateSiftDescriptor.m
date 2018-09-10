
function [database, im_info, sift_vector ] = CalculateSiftDescriptor(percent_tr,rt_img_dir, rt_data_dir, rt_result_dir, gridSpacing, patchSize, maxImSize, nrml_threshold)

%==========================================================================

% Creat data for output

%==========================================================================

im_info =[];

sift_vector = [];



database = [];

database.imnum = 0;   

database.cname = {};  

database.label = [];  

database.nclass = 0;  

database.cnum = 0;    



%==========================================================================



% an instant (subfolders) direct to the dir. which stored the dataset

subfolders = dir(rt_img_dir);
%ii = 0;
%fprintf(' %s\n',rt_img_dir);

for ii = 1:length(subfolders) %check all files in subfolders   

    % the name of a file in subfolders

    subname = subfolders(ii).name;

    

    %except the hiding folds
    %fprintf(' %s\n',subname);
    if ~strncmp(subname,'.',1) % find a class

        

        % update the information of database

        database.nclass = database.nclass + 1;  
        database.cname{database.nclass} = subname;


        % extracting all images in this sub-fold

         %frames = dir(fullfile(rt_img_dir, subname, '*.png')); 
         frames = dir(fullfile(rt_img_dir, subname, '*.jpg')); 

        % the total file number of this fold (include image files and other files)

        c_num = length(frames); 

        

        %each image in Linux has one hiding file, so cnum is 2 times than

        %fprintf(' -- nclass %d, cnum %d -- \n',database.nclass,c_num);

   

        

        %'jj' is a parameter for calculate the idx of images

        jj = 0;   

        for x = 1:c_num %check all files in this class

            if ~strncmp(frames(x).name,'.',1)% find an image

                

                jj = jj + 1;

                idx = database.nclass*1000 + jj;% idx is the index of image (jj is 1 at the begining, and have no 0)

                

                % extract the image

                imgpath = fullfile(rt_img_dir, subname, frames(x).name);

                I = imread(imgpath); 

              



                % clolor images exchange to glay image 

                if ndims(I) == 3           

                   I = im2double(rgb2gray(I));

                else

                   I = im2double(I);

                end

                 

                %limit the size of image

                [im_h, im_w] = size(I);

                if max(im_h, im_w) > maxImSize

                   I = imresize(I, maxImSize/max(im_h, im_w), 'bicubic');

                   [im_h, im_w] = size(I);   

                end


         

                % make grid sampling SIFT descriptors

                remX = mod(im_w-patchSize,gridSpacing);            

                offsetX = floor(remX/2)+1;

                remY = mod(im_h-patchSize,gridSpacing);

                offsetY = floor(remY/2)+1;

    



                

                [gridX,gridY] = meshgrid(offsetX:gridSpacing:im_w-patchSize+1, offsetY:gridSpacing:im_h-patchSize+1);



                %fprintf('\nidx: %d. Processing %s: wid %d, hgt %d, grid size: %d x %d, %d patches.\n',idx, frames(jj).name, im_w, im_h, size(gridX, 2), size(gridX, 1), numel(gridX));

             

                % find SIFT descriptors

                siftArr = sp_find_sift_grid(I, gridX, gridY, patchSize, 0.8);

                siftArr= sp_normalize_sift(siftArr, nrml_threshold);

           

                im_info(idx).patches = siftArr; 

                im_info(idx).hist = [];

                % if train_label is 1, this image include in training set

                im_info(idx).train_label = 0;

                %sift_vector = [sift_vector; siftArr];

                

                %==========================================================

                % a part for generate a set which store all SIFT descriptors of

                % traing set (the result is 'sift_vector') 

                %==========================================================

                if strncmp(frames(x).name,'t',1) % if the image is one in the training set

                    %fprintf('in class = %d, one in train set is: %s. Its idx is %d\n',database.nclass,frames(x).name,idx);

                    im_info(idx).train_label = 1; % change the label

                    sift_vector = [sift_vector; im_info(idx).patches]; % generate 'sift_vector' 

                end

                    



                



            end % an image has been extra SIFT

            

        end % all images in this class have been extract SIFT

        

        % update the information of database 

        database.cnum(database.nclass) = jj;

        database.imnum = database.imnum + database.cnum(database.nclass);

        database.label = [database.label; ones(database.cnum(database.nclass), 1)*database.nclass];

        

        

        

        

        %==========================================================

        %

        % a part for generate a set which store all SIFT descriptors of

        % traing set (the result is 'sift_vector')

        % 

        %==========================================================

        

        % calculate the total number of taring set in this class.

        %num_traing = database.cnum(database.nclass) * percent_tr;

        %num_traing = fix(num_traing);

        

        % ramdomly generate the indeied of images which are in the traing set.

        %idx_class = randperm(database.cnum(database.nclass),num_traing)



        % generate 'sift_vector'

        %for iiii = 1:num_traing  

        %    fprintf('in class = %d, the image%d of trian set is: image%d.\n',database.nclass,iiii,idx_class(iiii));

        %    idx = database.nclass * 1000 + idx_class(iiii);

            % label image is in training set

        %    im_info(idx).train_label = 1;

        %    sift_vector = [sift_vector; im_info(idx).patches];

        %end



    end % an subfold in dataset have been done

end

%code of carried

%vpath = fullfile(rt_result_dir,'vector.mat');

%save(vpath,'sift_vector');



%lenStat = hist(siftLens, 100);   %colums 100





