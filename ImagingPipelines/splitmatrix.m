function ndata = splitmatrix(data,nrow,ncol)

[r,c] = size(data);
rs = r / nrow;
cs = c / ncol;


%ndata = reshape(data,[rs,nrow,cs,ncol]);
%ndata = permute(ndata,[1,3,2,4]);
%ndata = reshape(ndata,[rs,cs,nrow * ncol]);

fprintf(1,'Splitting matrix...\n');
ndata = zeros(rs,cs,nrow * ncol,class(data));
for i = 1:nrow
    fprintf(1,'.');
    pos = (i-1) * ncol;
    for j = 1:ncol
        ndata(:,:,pos + j) = data((1 + (i-1) * rs):(i * rs),(1 + (j-1) * cs):(j * cs));
    end;
end;
fprintf(1,'** end of splitmatrix.. james \n');

