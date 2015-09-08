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

#include "parallelgrumodel.h"
#include "episode.h"

#include <nnetcpp/gru.h>
#include <nnetcpp/dense.h>
#include <nnetcpp/activation.h>
#include <nnetcpp/mergesum.h>

static const float learning_rate = 1e-3;

ParallelGRUModel::ParallelGRUModel(unsigned int hidden_neurons)
: _hidden_neurons(hidden_neurons)
{
}

Network *ParallelGRUModel::createNetwork(Episode *first_episode) const
{
    Network *net = new Network(first_episode->encodedStateSize());
    Dense *dense_in_in = new Dense(_hidden_neurons, learning_rate);
    Dense *dense_in_z = new Dense(_hidden_neurons, learning_rate);
    Dense *dense_in_r = new Dense(_hidden_neurons, learning_rate);
    Dense *dense_h_in = new Dense(_hidden_neurons, learning_rate);
    Dense *dense_h_z = new Dense(_hidden_neurons, learning_rate);
    Dense *dense_h_r = new Dense(_hidden_neurons, learning_rate);
    Dense *dense_in_h = new Dense(_hidden_neurons, learning_rate);
    Dense *dense_gru_h = new Dense(_hidden_neurons, learning_rate);
    GRU *gru = new GRU(_hidden_neurons, learning_rate);
    MergeSum *h = new MergeSum;
    TanhActivation *h_activation = new TanhActivation;
    Dense *out_gru = new Dense(first_episode->valueSize(), learning_rate);
    Dense *out_h = new Dense(first_episode->valueSize(), learning_rate);
    MergeSum *out = new MergeSum;

    dense_in_in->setInput(net->inputPort());
    dense_in_z->setInput(net->inputPort());
    dense_in_r->setInput(net->inputPort());

    gru->addInput(dense_in_in->output());
    gru->addInput(dense_h_in->output());
    gru->addZ(dense_in_z->output());
    gru->addZ(dense_h_z->output());
    gru->addR(dense_in_r->output());
    gru->addR(dense_h_r->output());
    dense_gru_h->setInput(gru->output());
    out_gru->setInput(gru->output());

    dense_in_h->setInput(net->inputPort());
    h->addInput(dense_in_h->output());
    h->addInput(dense_gru_h->output());
    h_activation->setInput(h->output());
    dense_h_in->setInput(h_activation->output());
    dense_h_z->setInput(h_activation->output());
    dense_h_r->setInput(h_activation->output());
    out_h->setInput(h_activation->output());

    out->addInput(out_gru->output());
    out->addInput(out_h->output());

    net->addNode(dense_in_in);
    net->addNode(dense_in_z);
    net->addNode(dense_in_r);
    net->addNode(dense_in_h);
    net->addNode(gru);
    net->addNode(h);
    net->addNode(h_activation);
    net->addNode(dense_h_in);
    net->addNode(dense_h_z);
    net->addNode(dense_h_r);
    net->addNode(dense_gru_h);
    net->addNode(out_gru);
    net->addNode(out_h);
    net->addNode(out);

    // The links between GRU and the hidden layer are recurrent
    net->addRecurrentNode(dense_h_in);
    net->addRecurrentNode(dense_h_z);
    net->addRecurrentNode(dense_h_r);
    net->addRecurrentNode(dense_gru_h);

    return net;
}
