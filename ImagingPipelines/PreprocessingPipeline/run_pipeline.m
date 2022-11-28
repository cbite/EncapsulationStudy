function run_pipeline(p, nr)

nr = str2num(nr)
x = load(p);

p = x.p
fchannels = x.fchannels
bgchannels = x.bgchannels

pipeline(p(nr), fchannels,bgchannels);
