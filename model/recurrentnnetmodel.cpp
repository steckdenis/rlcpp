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

#include "recurrentnnetmodel.h"
#include "nnetmodel.h"
#include "episode.h"

RecurrentNnetModel::RecurrentNnetModel()
: _network(nullptr),
  _last_episode_length(0)
{
}

RecurrentNnetModel::~RecurrentNnetModel()
{
    if (_network) {
        delete _network;
    }
}

void RecurrentNnetModel::values(Episode *episode, std::vector<float> &rs)
{
    if (!_network) {
        // No model available, clear out rs
        rs.resize(episode->valueSize());
        std::fill(rs.begin(), rs.end(), 0.0f);
    } else {
        // Reset the network if a new episode has been started
        if (episode->length() <= _last_episode_length) {
            _network->reset();
        }

        _last_episode_length = episode->length();

        // Convert the last state to an Eigen vector
        Vector last_state;

        episode->state(episode->length() - 1, rs);
        NnetModel::vectorToVector(rs, last_state);

        // Feed this input to the network
        Vector prediction = _network->predict(last_state);

        rs.resize(episode->valueSize());

        for (std::size_t i=0; i<rs.size(); ++i) {
            rs[i] = prediction(i);
        }
    }
}

void RecurrentNnetModel::learn(const std::vector<Episode *> &episodes)
{
    std::vector<float> state;
    std::vector<float> values;
    Vector input;
    Vector output;

    // Learn all the episodes separately, because they represent sequences
    // of observations that must be kept in order
    for (Episode *episode : episodes) {
        // Create the network if needed
        if (!_network) {
            _network = createNetwork(episode);
        }

        // Learn all the values obtained during the episode
        unsigned int start_t = std::max(0U, episode->length() - 100);

        for (int i=0; i<4; ++i) {
            _network->reset();

            for (unsigned int t=start_t; t < episode->length() - 1; ++t) {
                unsigned int action = episode->action(t);

                episode->state(t, state);
                episode->values(t, values);

                NnetModel::vectorToVector(state, input);
                NnetModel::vectorToVector(values, output);

                // Use only the value associated with the action that has been taken
                // when computing the errors
                _network->trainSample(input, output);
            }
        }
    }
}