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

#include "perceptronmodel.h"
#include "episode.h"

#include <nnetcpp/dense.h>
#include <nnetcpp/activation.h>

PerceptronModel::PerceptronModel(unsigned int hidden_neurons)
: _hidden_neurons(hidden_neurons)
{
    _hidden_activation = new TanhActivation;
}

Network *PerceptronModel::createNetwork(Episode *first_episode) const
{
    Network *network = new Network(first_episode->encodedStateSize());

    Dense *hidden = new Dense(_hidden_neurons, 1e-2);
    Dense *dense2 = new Dense(first_episode->valueSize(), 1e-2);

    hidden->setInput(network->inputPort());
    _hidden_activation->setInput(hidden->output());
    dense2->setInput(_hidden_activation->output());

    network->addNode(hidden);
    network->addNode(_hidden_activation);
    network->addNode(dense2);

    return network;
}

void PerceptronModel::hiddenValues(Episode *episode, std::vector<float> &rs)
{
    // Call values(), that takes care of the actual prediction
    values(episode, rs);

    // The hidden state is the output of the hidden layer
    NnetModel::getNodeOutput(_hidden_activation, rs);
}
