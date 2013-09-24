% setup the environment
curDir = pwd;
dependDir = sprintf('%s/3rdparty', curDir);
mkdir(dependDir);

% compile LLC max pooling code
display('Compiling LLC_pooling_mex_sparse.cc...');
mex -largeArrayDims LLC_pooling_mex_sparse.cc;

% download and compile SelectiveSearch bounding boxes detection
cd(dependDir);
disp('Downloading SelectiveSearchCode from Koen Van de Sande...');
cmd = sprintf('wget http://koen.me/research/downloads/SelectiveSearchCodeIJCV.zip; unzip SelectiveSearchCodeIJCV.zip; rm -f SelectiveSearchCodeIJCV.zip;', dependDir);
unix(cmd);
display('Compiling SelectiveSearchCode from Koen Van de Sande...\n');
mex SelectiveSearchCodeIJCV/Dependencies/anigaussm/anigauss_mex.c SelectiveSearchCodeIJCV/Dependencies/anigaussm/anigauss.c -output SelectiveSearchCodeIJCV/anigauss;
mex SelectiveSearchCodeIJCV/Dependencies/FelzenSegment/mexFelzenSegmentIndex.cpp -output SelectiveSearchCodeIJCV/mexFelzenSegmentIndex;
mex SelectiveSearchCodeIJCV/Dependencies/mexCountWordsIndex.cpp -output SelectiveSearchCodeIJCV/mexCountWordsIndex;
cd(curDir);

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
cd(curDir);
