% setup the environment
curDir = pwd;
dependDir = sprintf('%s/3rdparty', curDir);
mkdir(dependDir);

% compile LLC max pooling code
display('Compiling LLC_pooling_mex_sparse.cc...');
mex -largeArrayDims LLC_pooling_mex_sparse.cc;

% download and compile liblinear
cd(dependDir);
display('Downloading LIBLINEAR...');
cmd = 'wget http://www.csie.ntu.edu.tw/~cjlin/liblinear/liblinear-1.93.tar.gz; tar -xvf liblinear-1.93.tar.gz; rm -f liblinear-1.93.tar.gz;';
unix(cmd);
cd('liblinear-1.93/matlab');
make;
cd(curDir);

% download vlfeat and unpack it
cd(dependDir);
cmd = 'wget http://www.vlfeat.org/download/vlfeat-0.9.17-bin.tar.gz; tar -xvf vlfeat-0.9.17-bin.tar.gz; rm -f vlfeat-0.9.17-bin.tar.gz;';
unix(cmd);
cd(curDir);

% download flann and compile it
cd(dependDir);
cmd = 'wget http://www.cs.ubc.ca/research/flann/uploads/FLANN/flann-1.8.4-src.zip; unzip flann-1.8.4-src.zip; rm -f flann-1.8.4-src.zip; cd flann-1.8.4-src; mkdir build; cd build; cmake ../; make';
unix(cmd);
cd('flann-1.8.4-src/src/matlab');
mex nearest_neighbors.cpp -I../cpp  -DFLANN_STATIC -lflann_s -L../../build/lib/ CFLAGS="\$CFLAGS -fopenmp" LDFLAGS="\$LDFLAGS -fopenmp"
cd(curDir);
