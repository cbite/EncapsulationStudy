function d = sortmatrix(d)

[r,c,z] = size(d);
%sort on dimension 3
%do it in block to prevent memory problems

fprintf(1,'Sorting signal intensities')
stepsize=1;
for i = 1:64
    if ((mod(r,i) == 0) && (mod(c,i) == 0))
        stepsize = i;
    end;
end;

rs = r / stepsize;
rc = c / stepsize;
for i = 1:stepsize
    fprintf(1,',');
    for j = 1:stepsize
        fprintf(1,'.');
        dx = d((1 + (i-1) * rs):(i * rs),(1 + (j-1) * rc):(j * rc),:);
        d((1 + (i-1) * rs):(i * rs),(1 + (j-1) * rc):(j * rc),:) = sort(dx,3);
    end;
end;
fprintf(1,'\n');


