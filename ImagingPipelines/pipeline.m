function pipeline(paths, fchannels, bgchannels)

dip_initialise_libs;

if nargin < 2
    bgchannels = fchannels;
end;

for i = 1:length(paths)
    p = paths{i};
    mkdir(p, 'validation');
    mkdir(p, 'cropped');
    mkdir(p, 'processed');
    fprintf(1,'validation,cropped and processed folders have been created..in folder\n');
    fprintf(1,'the directory was %s\n',p);
    focus = {};
    minval = zeros(length(fchannels),1);
    maxval = zeros(length(fchannels),1);

    featnames = {};
    featstats = [];
    c = [];
    for j = fchannels
        %if(j == overlap_channel)
        %    c = channelo;
        %else
        clear c;
        c = readall(p,j);
        write_image(c, j, p, 0); 
        c = correct_distribution_robust(c);
        write_image(c, j, p, 1); 
        [c,minval(j),maxval(j)] = correct_range(c);
        write_image(c, j, p, 2); 
        if ismember(j,bgchannels)
            [c,bg] = correct_background2(c);
            write_image(c, j, p, 3); 
        else
            [c2,bg] = correct_background2(c);
            write_image(c2, j, p, 3); 
            clear c2
        end;
        
        %bg-stats
        write_image(bg, j, p, 4); 
        [nnames, nstats] = perform_stats(c, sprintf('bg%d',j));
        featnames = cat(2, featnames, nnames);
        featstats = cat(3, featstats, nstats);
        clear bg
       
        %fg-stats
        [nnames, nstats] = perform_stats(c, sprintf('fg%d',j));
        featnames = cat(2, featnames, nnames);
        featstats = cat(3, featstats, nstats);
        perform_crop(c, j, p);

        %focusstats
        focus{j} = detect_focus(c);                    
    end;
    fn = sprintf('%svalidation/feat.mat',char(p));
    save(fn, 'featnames', 'featstats','focus','minval','maxval','fchannels');
end;
