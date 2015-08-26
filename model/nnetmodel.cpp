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

#include "nnetmodel.h"
#include "episode.h"

NnetModel::NnetModel()
: _network(nullptr)
{
}

NnetModel::~NnetModel()
{
    if (_network) {
        delete _network;
    }
}

void NnetModel::values(Episode *episode, std::vector<float> &rs)
{
    if (!_network) {
        // No model available, clear out rs
        rs.resize(episode->valueSize());
        std::fill(rs.begin(), rs.end(), 0.0f);
    } else {
        // Convert the last state to an Eigen vector
        Vector last_state;

        episode->encodedState(episode->length() - 1, rs);
        vectorToVector(rs, last_state);

        // Feed this input to the network
        Vector prediction = _network->predict(last_state);

        rs.resize(episode->valueSize());

        for (std::size_t i=0; i<rs.size(); ++i) {
            rs[i] = prediction(i);
        }
    }
}

void NnetModel::learn(const std::vector<Episode *> &episodes)
{
    std::vector<float> state;
    std::vector<float> values;

    // Create a big matrix with one column per input/output pair
    std::size_t total_size = 0;

    for (Episode *episode : episodes) {
        total_size += episode->length() - 1;
    }

    Eigen::MatrixXf inputs(episodes[0]->encodedStateSize(), total_size);
    Eigen::MatrixXf outputs(episodes[0]->valueSize(), total_size);

    // Fill this matrix
    int index = 0;

    for (Episode *episode : episodes) {
        // Create the network if needed
        if (!_network) {
            _network = createNetwork(episode);
        }

        // Learn all the values obtained during the episode
        for (unsigned int t=0; t < episode->length() - 1; ++t) {
            episode->encodedState(t, state);
            episode->values(t, values);

            vectorToCol(state, inputs, index);
            vectorToCol(values, outputs, index);

            ++index;
        }
    }

    // Train the network on that data
    _network->train(inputs, outputs, 10, 4);
}

void NnetModel::vectorToVector(const std::vector<float> &stl, Vector &eigen)
{
    eigen.resize(stl.size());

    for (std::size_t i=0; i<stl.size(); ++i) {
        eigen(i) = stl[i];
    }
}

void NnetModel::vectorToCol(const std::vector<float> &stl, Matrix &matrix, int col)
{
    for (std::size_t i=0; i<stl.size(); ++i) {
        matrix(i, col) = stl[i];
    }
}

void NnetModel::getNodeOutput(AbstractNode *node, std::vector<float> &rs)
{
    const Vector &o = node->output()->value;

    rs.resize(o.rows());

    for (std::size_t i=0; i<rs.size(); ++i) {
        rs[i] = o(i);
    }
}
