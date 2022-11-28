function [channel, bgchannel] = correct_background2(channel)

filter_size = 40.0; %changed by james from 40
fprintf(1,'The FILTER_SIZE is..%d \n',filter_size);
avg_filter = fspecial('disk',filter_size);
fprintf(1,'Correct Background II, using disk filter....\n');
%diff = 3; %james was diff=5
d = splitmatrix(channel.data,channel.nrow*channel.nrow_sub,channel.ncol*channel.ncol_sub);
%nd = zeros(size(d), 'uint16');
%fprintf(1,'The size of matrix d is...\n');
[r,c,z] = size(d);

fprintf(1,'Determining background signal..');
fprintf(1,'\n');
ndepth = 5; %this value does not get used..
beach=size(r,c);

for i = 1:z
%     if(mod(i,10) == 0)
%         fprintf(1,'loop step 10 ****\n');
%     end;
    
    s = single(squeeze(d(:,:,i)));
    %fprintf(1,'now squeeze the 3d images..\n');
    %r   esize_stack = {};
    %    resize_stack{1} = imresize(s,0.5);
    %    for i = 2:ndepth
    %        resize_stack{i} = imresize(resize_stack{i-1}, 0.5);
    %    end;
    %    resize_stack = fliplr(resize_stack);
    %    xfilter = fspecial('disk',2);
    %    xs = imfilter(s,xfilter,'replicate');
    xs = imresize(s,1.0); %changed from 0.5 to 1.0, silly i know
    %fprintf(1,'Changing image size..\n');
    
    b = xs;
    diff = 5; %james was diff=5    
    for j = 1:5 %james was 1:5 for differr
        bnew = imfilter(b,avg_filter,'replicate');
        idx = xs > (bnew + 2 * diff);
        %fprintf(1,'What the freaking hell does this line do!\n');
        b(idx) = bnew(idx);
    end;
    
        %b = imresize(b,2); %Marc old code, <12dec2013;
        % Marc suggestion 12dec2013 so will try it;
        b = imresize(b, [r,c]); 
        %fprintf(1,'The size of image B is row %d\n',r); 
        %fprintf(1,' and col %d\n',c);
        %will use this for next run JJR >12dec2013;
        d(:,:,i) = s-b;
        %nd(:,:,i) = uint16(b);
end;

  fprintf(1,'**** james end of correct background2 ...\n');
  ndata = combinematrix(d,channel.nrow * channel.nrow_sub, channel.ncol * channel.ncol_sub);
  clear d

  bgchannel.data = channel.data - ndata;
  bgchannel.nrow = channel.nrow;
  bgchannel.nrow_sub = channel.nrow_sub;
  bgchannel.ncol_sub = channel.ncol_sub;
  bgchannel.ncol = channel.ncol;
  channel.data = ndata;


