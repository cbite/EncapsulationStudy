function focus = detect_focus(ar)
%do not edit anything in this file - james

[r,c] = size(ar.data);
nrow = (ar.nrow * ar.nrow_sub);
ncol = (ar.ncol * ar.ncol_sub);

rres = r / nrow;
cres = c / ncol;

focus = zeros(nrow,ncol);
fprintf(1,'Detecting focus: now doing laplace thing to images...\n');
for i = 1:nrow
    fprintf(1,'.');
    for j = 1:ncol
        br = (i-1) * rres + 1;
        er = i * rres;
        bc = (j-1) * cres + 1;
        ec =  j * cres;
        part = single(ar.data(br:er,bc:ec));
        %part = part - mean(part(:));
        %part = part ./ std(part(:));
        lpart = struct(laplace(dip_image(part),3)).data;
        focus(i,j) = std(lpart(:));
    end;
end;
fprintf(1,'\n');


