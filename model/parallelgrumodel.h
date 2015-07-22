/*
 * Copyright (c) 2015 Vrije Universiteit Brussel
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#ifndef __PARALLELGRUMODEL_H__
#define __PARALLELGRUMODEL_H__

#include "recurrentnnetmodel.h"

/**
 * @brief Recurrent neural network based on Gated Recurrent Units
 *
 * This network consists of a single-hidden-layer perceptron working in parallel
 * with a layer of GRU units. The output of the GRU is connected to the input of
 * the hidden layer, and the output of the hidden layer is connected to the input
 * of the GRU. The output of the hidden layer and the GRU are added to produce
 * the output of the network.
 *
 * This somewhat complex architecture is the one used in
 * "Reinforcement Learning with Long Short-Term Memory", Bram Bakker, 2001. The
 * LSTM units have been replaced with GRU units.
 */
class ParallelGRUModel : public RecurrentNnetModel
{
    public:
        ParallelGRUModel(unsigned int hidden_neurons);

        virtual Network *createNetwork(Episode *first_episode) const;

    private:
        unsigned int _hidden_neurons;
};

#endif