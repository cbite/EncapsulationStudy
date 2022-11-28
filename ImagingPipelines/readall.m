function results = readall(p,channel,rows,cols)

fprintf(1,'### james start of readall ..\n');

if nargin < 4
    cols = 66; %j 22
end; 
if nargin < 3
    rows = 66; %j 22
end;

if(channel == 1)
   fx = '';
end;
if(channel == 2)
   fx = ''; %changed by james
end;
if(channel == 3)
   fx = '';
end;
if(channel == 4)
   fx = '';
end;
if(channel == 5)
   fx = '';
end;
if(channel == 6)
   fx = ''; %not used
end;
if(channel == 7)
   fx = ''; %not used
end;
if(channel == 8)
   fx = ''; %changed by james
end;

tiffnumber = 0;
jamestiffin= ([rows,cols]);

forwardflag = mod(rows,2);
counter = 0;
fprintf(1,'Reading chip data');
for ri = fliplr(1:rows)
    fprintf(1,'.');
    rowres = {};
    for ci = 1:cols
        filename = sprintf(fx,counter);
        dp = sprintf('%s/%s.tif',p,filename); %removed ; by james
        if ~exist(dp,'file')
            ix = zeros(size(ix),class(ix));
        else
            ix = readim(dp,'');
            ix = struct(ix).data;
        end;
        counter = counter + 1 ; % removed ; by james
        rowres{ci} = ix;
        tiffnumber = ( counter - 1 );
        jamestiffin(ri,ci)= tiffnumber;
    end;
    if mod(ri,2) == forwardflag
        rowsres{ri} = cat(2,rowres{:});
    else
        rowres = fliplr(rowres);
        rowsres{ri} = cat(2,rowres{:});
    end;
end;
fprintf(1,'**** james end of readll..\n');
results.data = cat(1,rowsres{:});
results.nrow = rows;
results.nrow_sub = 1;
results.ncol = cols;
results.ncol_sub = 1;
