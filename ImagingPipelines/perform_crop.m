function crops = perform_crop(channel, channelnr, cpath)
fx = channel_name(channelnr);
xsize = 850; %changed from 800 - should be same in perform_stats


data = channel.data;
mindata = min(data(:));
maxdata = single(max(data(:)));
scaledata = single(255.0 ./ maxdata); 

[vsize,hsize] = size(channel.data);
nrow = channel.nrow * channel.nrow_sub;
ncol = channel.ncol * channel.ncol_sub;
vsize = vsize / nrow;
hsize = hsize / ncol;

cropnr = 0;
fprintf(1,'Storing crops');
for i = 1:nrow
    fprintf(1,'.');
    for j = 1:ncol
        cropnr = cropnr + 1;
       
        vpos = (i-1) * vsize + 1;
        hpos = (j-1) * hsize + 1;
        x = data(vpos:(vpos + vsize - 1), hpos:(hpos + hsize -1));
        x = uint8(single(x - mindata) * scaledata);

        [r,c] = size(x);

        rp = max(round((r - xsize) / 2),1);
        cp = max(round((c - xsize) / 2),1);
        
        x = x(rp:end, cp:end);
        x = x(1:min(xsize,size(x,1)), 1:min(xsize,size(x,2)));
        
        fn = sprintf('%scropped/%s_%0.10d.png',char(cpath), fx, cropnr);
        imwrite(x, fn,'PNG');
    end;
end;

fprintf(1,'\n');
