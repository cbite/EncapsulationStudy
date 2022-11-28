function fx = channel_name(channel)

%do not forget to add channel name also to readall
if(channel == 1)
    fx = 'DAPI';
elseif(channel == 2)
    fx = 'CMO';
elseif(channel == 3)
    fx = 'FarRed';
elseif(channel == 4)
    fx = 'BF';
elseif(channel == 5)
    fx = 'RGB';
elseif(channel == 6)
    fx = 'alexa555';
elseif(channel == 7)
    fx = 'gfp';
elseif(channel == 8)
    fx = 'alexa594';
end;

