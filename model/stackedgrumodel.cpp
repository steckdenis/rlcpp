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

#include "stackedgrumodel.h"
#include "episode.h"

#include <nnetcpp/gru.h>
#include <nnetcpp/dense.h>
#include <nnetcpp/activation.h>

StackedGRUModel::StackedGRUModel(unsigned int hidden_neurons)
: _hidden_neurons(hidden_neurons)
{
}

Network *StackedGRUModel::createNetwork(Episode *first_episode) const
{
    static const float learning_rate = 1e-3;

    Network *net = new Network(first_episode->stateSize());
    Dense *dense_in = new Dense(_hidden_neurons, learning_rate);
    Dense *dense_z = new Dense(_hidden_neurons, learning_rate);
    Dense *dense_r = new Dense(_hidden_neurons, learning_rate);
    GRU *gru = new GRU(_hidden_neurons, learning_rate);
    Dense *out = new Dense(first_episode->valueSize(), learning_rate);

    dense_in->setInput(net->inputPort());
    dense_z->setInput(net->inputPort());
    dense_r->setInput(net->inputPort());
    gru->addInput(dense_in->output());
    gru->addZ(dense_z->output());
    gru->addR(dense_r->output());
    out->setInput(gru->output());

    net->addNode(dense_in);
    net->addNode(dense_z);
    net->addNode(dense_r);
    net->addNode(gru);
    net->addNode(out);

    return net;
}
