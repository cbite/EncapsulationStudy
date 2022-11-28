function write_image(channel, channelnr, cpath, phase)
fx = channel_name(channelnr);

fn = sprintf('%sprocessed/%s_%d.mat',char(cpath), fx, phase);
save(fn,'channel','-v7.3');

data = channel.data(1:9:end, 1:9:end);
data = data - min(data(:));
data = single(data);
data = uint8((data ./ max(data(:))) * 255);

[vsize,hsize] = size(channel.data);
nrow = channel.nrow * channel.nrow_sub;
ncol = channel.ncol * channel.ncol_sub;
vsize = vsize / nrow;
hsize = hsize / ncol;

for i = 1:(nrow-1)
    q = round((vsize * i) / 9);
    data(:,q) = 0;
end;
for i = 1:(ncol-1)
    q = round((hsize * i) / 9);
    data(q,:) = 0;
end;

fn = sprintf('%svalidation/%s_%d.png',char(cpath), fx, phase);
imwrite(data, fn,'PNG');

