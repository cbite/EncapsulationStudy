function nc = filter_section(channel,startv,stopv,starth,stoph)

nrows = channel.nrow * channel.nrow_sub;
ncols = channel.ncol * channel.ncol_sub;

[rowsize,colsize] = size(channel.data);

per_rowsize = rowsize / nrows;
per_colsize = colsize / ncols;

nc.nrow = stopv - startv + 1;
nc.ncol = stoph - starth + 1;
nc.data = channel.data(((startv - 1) * per_rowsize + 1):(stopv * per_rowsize),((starth - 1) * per_colsize + 1):(stoph * per_colsize));
nc.nrow_sub = 1;
nc.ncol_sub = 1;
