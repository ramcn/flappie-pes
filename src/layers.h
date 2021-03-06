/*  Copyright 2018 Oxford Nanopore Technologies, Ltd */

/*  This Source Code Form is subject to the terms of the Oxford Nanopore
 *  Technologies, Ltd. Public License, v. 1.0. If a copy of the License
 *  was not  distributed with this file, You can obtain one at
 *  http://nanoporetech.com
 */

#pragma once
#ifndef LAYERS_H
#    define LAYERS_H

#    include "flappie_matrix.h"

void tanh_activation_inplace(flappie_matrix C);
void tanh_activation_inplace_vec(flappie_matrix_vec C, int nfiles);
void exp_activation_inplace(flappie_matrix C);
void log_activation_inplace(flappie_matrix C);
void elu_activation_inplace(flappie_matrix C);
void robustlog_activation_inplace(flappie_matrix C, float min_prob);

flappie_matrix embedding(int const * index, size_t n, const_flappie_matrix E,
                          flappie_matrix C);
flappie_matrix window(const_flappie_matrix input, size_t w, size_t stride);
flappie_matrix convolution_linear(const_flappie_matrix X, const_flappie_matrix W, const_flappie_matrix b, size_t stride, flappie_matrix C,
				const_flappie_matrix iW, const_flappie_matrix bG);
flappie_matrix convolution(const_flappie_matrix X, const_flappie_matrix W, const_flappie_matrix b, size_t stride, flappie_matrix C);
flappie_matrix_vec convolution_vec(flappie_matrix_vec X, const_flappie_matrix W, const_flappie_matrix b, size_t stride, int nfiles);
flappie_matrix feedforward_linear(const_flappie_matrix X, const_flappie_matrix W, const_flappie_matrix b, flappie_matrix C);
flappie_matrix_vec feedforward_linear_vec(const_flappie_matrix_vec X, const_flappie_matrix W, const_flappie_matrix b, flappie_matrix_vec C);
flappie_matrix feedforward_tanh(const_flappie_matrix X,
                                 const_flappie_matrix W,
                                 const_flappie_matrix b, flappie_matrix C);
flappie_matrix feedforward_exp(const_flappie_matrix X,
                                const_flappie_matrix W,
                                const_flappie_matrix b, flappie_matrix C);
flappie_matrix residual(const_flappie_matrix X, const_flappie_matrix fX,
                         flappie_matrix C);
void residual_inplace(const_flappie_matrix X, flappie_matrix fX);
flappie_matrix softmax(const_flappie_matrix X, const_flappie_matrix W,
                        const_flappie_matrix b, flappie_matrix C);
flappie_matrix softmax_with_temperature(flappie_matrix X, const_flappie_matrix W,
                                         const_flappie_matrix b, float tempW, float tempb,
                                         flappie_matrix C);

flappie_matrix feedforward2_tanh(const_flappie_matrix Xf,
                                  const_flappie_matrix Xb,
                                  const_flappie_matrix Wf,
                                  const_flappie_matrix Wb,
                                  const_flappie_matrix b, flappie_matrix C);

flappie_matrix gru_forward(const_flappie_matrix X, const_flappie_matrix sW,
                            const_flappie_matrix sW2, flappie_matrix res);
flappie_matrix gru_backward(const_flappie_matrix X, const_flappie_matrix sW,
                             const_flappie_matrix sW2, flappie_matrix res);
void gru_step(const_flappie_matrix x, const_flappie_matrix istate,
              const_flappie_matrix sW, const_flappie_matrix sW2,
              flappie_matrix xF, flappie_matrix ostate);

flappie_matrix grumod_forward(const_flappie_matrix X, const_flappie_matrix sW, flappie_matrix res);
flappie_matrix_vec grumod_forward_vec(const_flappie_matrix_vec X, const_flappie_matrix sW, flappie_matrix_vec res);
flappie_matrix grumod_backward(const_flappie_matrix X, const_flappie_matrix sW, flappie_matrix res);
flappie_matrix_vec grumod_backward_vec(const_flappie_matrix_vec X, const_flappie_matrix sW, flappie_matrix_vec res);
flappie_matrix aes_grumod_linear(const_flappie_matrix X, const_flappie_matrix sW, flappie_matrix ostate, int backward, const_flappie_matrix W, const_flappie_matrix b);
flappie_matrix aes_grumod(const_flappie_matrix X, const_flappie_matrix sW, flappie_matrix ostate, bool backward, const_flappie_matrix W, const_flappie_matrix b);
flappie_matrix_vec aes_grumod_vec( flappie_matrix_vec Xin, const_flappie_matrix sW, flappie_matrix_vec ostate, bool backward, const_flappie_matrix W, const_flappie_matrix b); 

flappie_matrix_vec aes_grumod_vec_forward( flappie_matrix_vec Xin, const_flappie_matrix sW, flappie_matrix_vec ostate, const_flappie_matrix W, const_flappie_matrix b, int nfiles); 

flappie_matrix_vec aes_grumod_vec_backward( flappie_matrix_vec Xin, const_flappie_matrix sW, flappie_matrix_vec ostate, const_flappie_matrix W, const_flappie_matrix b, int nfiles); 

void grumod_step(const_flappie_matrix x, const_flappie_matrix istate,
                 const_flappie_matrix sW, flappie_matrix xF,
                 flappie_matrix ostate);

flappie_matrix gru_relu_forward(const_flappie_matrix X, const_flappie_matrix sW,
                                const_flappie_matrix sW2, flappie_matrix res);
flappie_matrix gru_relu_backward(const_flappie_matrix X, const_flappie_matrix sW,
                                 const_flappie_matrix sW2, flappie_matrix res);
void gru_relu_step(const_flappie_matrix x, const_flappie_matrix istate,
                   const_flappie_matrix sW, const_flappie_matrix sW2,
                   flappie_matrix xF, flappie_matrix ostate);

flappie_matrix lstm_forward(const_flappie_matrix X, const_flappie_matrix sW,
                            flappie_matrix output);
flappie_matrix lstm_backward(const_flappie_matrix X, const_flappie_matrix sW,
                             flappie_matrix output);
void lstm_step(const_flappie_matrix x, const_flappie_matrix out_prev,
               const_flappie_matrix sW, flappie_matrix xF,
               flappie_matrix state, flappie_matrix output);


double crf_manystay_partition_function(const_flappie_matrix C);
flappie_matrix globalnorm_manystay(const_flappie_matrix X, const_flappie_matrix W,
                                   const_flappie_matrix b, float temperature,
                                   flappie_matrix C);
size_t nbase_from_flipflop_nparam(size_t nparam);
flappie_matrix globalnorm_flipflop(const_flappie_matrix X, const_flappie_matrix W,
                                   const_flappie_matrix b, float temperature,
                                   flappie_matrix C);
size_t nbase_from_runlength_nparam(size_t nparam);
flappie_matrix globalnorm_runlength(const_flappie_matrix X, const_flappie_matrix W,
                                    const_flappie_matrix b, float temperature,
                                    flappie_matrix C);
size_t nbase_from_crf_runlength_nparam(size_t nparam);
flappie_matrix globalnorm_runlengthV2(const_flappie_matrix X, const_flappie_matrix W,
                                      const_flappie_matrix b, float temperature,
                                      flappie_matrix C);
#endif                          /* LAYERS_H */
