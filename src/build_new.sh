make -f makefile.cuda

gcc -O3 -o skeleton -march=native  -Wfatal-errors   -Wunused-function -Wunused-value -Wunused-parameter -fstack-protector-all  -O3   -D__USE_MISC -D_POSIX_SOURCE -DNDEBUG  -I./  -I/usr/local/cuda/include -I/usr/include/hdf5/serial flappie_matrix.c flappie_util.c  skeleton.c fast5_interface.c util.c flappie_common.c layers.c nnfeatures.c  -L/usr/local/cuda/lib64  -lpthread -lblas -lm -lstdc++ -lcublas -lcudart -lhdf5_serial
