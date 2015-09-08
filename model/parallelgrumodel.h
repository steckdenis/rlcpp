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
 * This network uses an architecture as described in "Reinforcement Learning
 * with Long Short-Term Memory": a GRU node works in parallel with a tanh node.
 * The input is connected to both nodes, the nodes are both connected to the output,
 * and the output of each node is connected to the input of the other.
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
