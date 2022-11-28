function [names, res]= perform_stats(channel, name)

xsize = 850; %should be same as perform_crop
data = channel.data;
mindata = min(data(:));
maxdata = single(max(data(:)));
scaledata = single(255.0 ./ maxdata);
cropnr = 0; %only used for number checks
fprintf(1,'Calculating stats\n');

[vsize,hsize] = size(channel.data);
nrow = channel.nrow * channel.nrow_sub;
ncol = channel.ncol * channel.ncol_sub;
vsize = vsize / nrow;
hsize = hsize / ncol;

name_pre = {'mean', 'median', 'min', 'max', 'std'};
names = {};
for i = 1:length(name_pre)
    names{i} = sprintf('%s%s',name, name_pre{i});
end;
res = zeros(channel.nrow * channel.nrow_sub, channel.ncol * channel.ncol_sub, length(name_pre));
for i = 1:nrow
    fprintf(1,'.');
    for j = 1:ncol
        cropnr = cropnr + 1;
        
        vpos = (i -1) * vsize + 1;
        hpos = (j-1) * hsize + 1;
        x = data(vpos:(vpos + vsize -1), hpos:(hpos + hsize - 1));
        x = uint8(single(x - mindata) * scaledata);

        [r,c] = size(x);

        rp = max(round((r - xsize) / 2),1);
        cp = max(round((c - xsize) / 2),1);
        
        x = x(rp:end, cp:end);
        x = x(1:min(xsize,size(x,1)), 1:min(xsize,size(x,2)));
        x = single(x(:));

        res(i,j,1) = mean(x);
        res(i,j,2) = median(x);
        res(i,j,3) = min(x);
        res(i,j,4) = max(x);
        res(i,j,5) = std(x);
    end;
end;
fprintf(1,'\n');
