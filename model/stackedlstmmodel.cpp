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

#include "stackedlstmmodel.h"
#include "episode.h"

#include <nnetcpp/lstm.h>
#include <nnetcpp/dense.h>
#include <nnetcpp/activation.h>

static const float learning_rate = 2e-3;

StackedLSTMModel::StackedLSTMModel(unsigned int hidden_neurons)
: _hidden_neurons(hidden_neurons)
{
}

Network *StackedLSTMModel::createNetwork(Episode *first_episode) const
{
    Network *net = new Network(first_episode->encodedStateSize());
    Dense *dense_in = new Dense(_hidden_neurons, learning_rate);
    Dense *dense_ingate = new Dense(_hidden_neurons, learning_rate);
    Dense *dense_outgate = new Dense(_hidden_neurons, learning_rate);
    Dense *dense_forgetgate = new Dense(_hidden_neurons, learning_rate);
    LSTM *lstm = new LSTM(_hidden_neurons, learning_rate);
    Dense *out = new Dense(first_episode->valueSize(), learning_rate);

    dense_in->setInput(net->inputPort());
    dense_ingate->setInput(net->inputPort());
    dense_outgate->setInput(net->inputPort());
    dense_forgetgate->setInput(net->inputPort());
    lstm->addInput(dense_in->output());
    lstm->addInGate(dense_ingate->output());
    lstm->addOutGate(dense_outgate->output());
    lstm->addForgetGate(dense_forgetgate->output());
    out->setInput(lstm->output());

    net->addNode(dense_in);
    net->addNode(dense_ingate);
    net->addNode(dense_outgate);
    net->addNode(dense_forgetgate);
    net->addNode(lstm);
    net->addNode(out);

    return net;
}
