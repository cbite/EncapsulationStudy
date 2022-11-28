
function [channel,minval,maxval] = correct_range(channel, range)
if nargin < 2
    range = 0.2;
end;

d = splitmatrix(channel.data,channel.nrow*channel.nrow_sub,channel.ncol*channel.ncol_sub);

[r,c,z] = size(d);

minval = double(prctile(squeeze(min(min(d,[],2),[],1)),range * 100));
maxval = double(prctile(squeeze(max(max(d,[],2),[],1)),(1.0 - range) * 100));

ndata = zeros(size(d),'uint16'); % did try at 16 but it could be wrong - it generates more data j
%changed from 8
fprintf(1,'Signal ranging....');
for i = 1:z
    if(mod(i,100) == 0)
        fprintf(1,'100-');
    end;
   
    x = double(d(:,:,i) - minval);
    nmaxval = maxval - minval;

    x = uint16(x / (nmaxval/65355.0)); % did try at 16 but it could be wrong - j
    %changed from 8 and 255.0 some images are now too faint - generates more data
    ndata(:,:,i) = x;
end;
fprintf(1,'*** end of correct range with uint of 16... james\n');
clear d;
channel.data = combinematrix(ndata,channel.nrow * channel.nrow_sub, channel.ncol * channel.ncol_sub);
clear ndata;


