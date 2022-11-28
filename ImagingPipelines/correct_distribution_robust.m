function [channel,res] = correct_distribution_robust(channel)

d = splitmatrix(channel.data,channel.nrow * channel.nrow_sub,channel.ncol * channel.ncol_sub);

[r,c,z] = size(d);

%copy inline to prevent memory issues
%ds = sortmatrix(d);
fprintf(1,'Sorting signal intensities\n');
stepsize=1;
for i = 1:64
    if ((mod(r,i) == 0) && (mod(c,i) == 0))
        stepsize = i;
    end;
end;

rs = r / stepsize;
rc = c / stepsize;
for i = 1:stepsize
    fprintf(1,'*');
    for j = 1:stepsize
        %if(mod(j,100)==0);fprintf(1,'-100-*-');end; %james silly print
        dx = d((1 + (i-1) * rs):(i * rs),(1 + (j-1) * rc):(j * rc),:);
        d((1 + (i-1) * rs):(i * rs),(1 + (j-1) * rc):(j * rc),:) = sort(dx,3);
    end;
end;
fprintf(1,'\n');
%end inline

meanprof = squeeze(single(mean(mean(d,1),2)));
meanprof(meanprof == 0) = 1;
res = zeros(r,c);
fprintf(1,'Determining signal gain\n');
for i = 1:r
    if(mod(i,250) == 0)
        fprintf(1,'250-');
    end;
 
    for j = 1:c
        res(i,j) = median(meanprof ./ single(squeeze(d(i,j,:))));
    end; 
end;
fprintf(1,'\n');
clear d

d = splitmatrix(channel.data,channel.nrow * channel.nrow_sub,channel.ncol * channel.ncol_sub);

fprintf(1,'Correcting signal gain..\n');
for i = 1:z
    if(mod(i,100) == 0)
        fprintf(1,'100-');
    end;
    d(:,:,i) = squeeze(single(d(:,:,i))) .* res;
end;
fprintf(1,'\n');
channel.data = combinematrix(d,channel.nrow * channel.nrow_sub, channel.ncol * channel.ncol_sub);
