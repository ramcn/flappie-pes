/*  Copyright 2018 Oxford Nanopore Technologies, Ltd */

/*  This Source Code Form is subject to the terms of the Oxford Nanopore
 *  Technologies, Ltd. Public License, v. 1.0. If a copy of the License 
 *  was not  distributed with this file, You can obtain one at
 *  http://nanoporetech.com
 */
#include "fast5_interface.h"
#include "layers.h"
//#include "nanonet_rgr.h"
#include "nnfeatures.h"
#include "flappie_common.h"
#include "flappie_util.h"
#include "skeleton.h"
#include <sys/time.h>
struct timeval start, end_time;
long useconds, seconds, mseconds;


int main(int argc, char * argv[]){
    int trim_start = 200;
    int trim_end = 10;
    int varseg_chunk = 100;
    float varseg_thresh = 0;

    const guppy_model * net = &flipflop_r941native_guppy;
    char * filename = argv[1];

    raw_table signal = read_raw(filename, true);
    signal =  trim_and_segment_raw(signal, trim_start, trim_end, varseg_chunk, varseg_thresh);


    flappie_matrix raw_mat = features_from_raw(signal);
    write_flappie_matrix("input.crp", raw_mat);
    flappie_matrix conv = convolution(raw_mat, net->conv_W, net->conv_b, net->conv_stride, NULL);
    //elu_activation_inplace(conv);
    tanh_activation_inplace(conv);
    write_flappie_matrix("convolution.crp", conv);

    raw_mat = free_flappie_matrix(raw_mat);

    gettimeofday(&start, NULL);

    flappie_matrix gruB1in = feedforward_linear(conv, net->gruB1_iW, net->gruB1_b, NULL);
    conv = free_flappie_matrix(conv);
    flappie_matrix gruB1 = grumod_backward(gruB1in, net->gruB1_sW, NULL);
    
    write_flappie_matrix("gruB1.crp", gruB1);
    gruB1 = free_flappie_matrix(gruB1);

    gettimeofday(&end_time, NULL);
    useconds = end_time.tv_usec - start.tv_usec;
    seconds = end_time.tv_sec - start.tv_sec;
    mseconds = ((seconds) * 1000 + useconds/1000.0) + 0.5;
    fprintf(stderr,"Time elapsed (doing 1 GRU): %ld msec\n", mseconds );

}
