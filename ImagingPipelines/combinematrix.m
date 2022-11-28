function ndata = combinematrix(data,nrow,ncol)

[rs,cs,z] = size(data);

%ndata = reshape(data,[rs,nrow,cs,ncol]);
%ndata = permute(ndata,[1,3,2,4]);
%ndata = reshape(ndata,[rs,cs,nrow * ncol]);

fprintf(1,'Combining matrix');
ndata = zeros(rs * nrow,cs * ncol,class(data));
for i = 1:nrow
    fprintf(1,'.');
    pos = (i-1) * ncol;
    for j = 1:ncol
        ndata((1 + (i-1) * rs):(i * rs),(1 + (j-1) * cs):(j * cs)) = data(:,:,pos + j); 
    end;
end;
fprintf(1,'\n');

