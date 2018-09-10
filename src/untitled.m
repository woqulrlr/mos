matrix_A = importdata('matrix.mat');
p = 15;
q = 15;
p = p - 5;
q = q - 5;
%a = nzero(11,11);
matrix_A(p:p+11,q:q+11) = 0;

l = size(matrix_A,1);
w = size(matrix_A,2);

for l = 2:size(matrix_A,1)-1
    for w = 2:size(matrix_A,2)-1
        if(matrix_A(l,w) > matrix_A(l-1,w))
            if(matrix_A(l,w) > matrix_A(l,w-1))
                if(matrix_A(l,w) > matrix_A(l+1,w))
                    if(matrix_A(l,w) > matrix_A(l,w+1))
                        fprintf('%f \n',matrix_A(l,w));
                    end
                end
            end
        end
    end
end
      
                