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

void RecurrentNnetModel::nextEpisode()
{
    _last_episode_length = 0;

    // Reset the network between time steps
    if (_network) {
        _network->reset();
    }
}

void RecurrentNnetModel::values(Episode *episode, std::vector<float> &rs)
{
    if (!_network) {
        // No model available, clear out rs
        rs.resize(episode->valueSize());
        std::fill(rs.begin(), rs.end(), 0.0f);
    } else {
        // Use _last_episode_length..length time steps for prediction. This allows
        // the model to be kept "up-to-date" if time steps are skipped in the episode,
        // for instance of AbstractWorld copies N time steps from an episode and
        // then tries to predict the next one.
        for (unsigned int t=_last_episode_length; t<episode->length(); ++t) {
            // Tell the network which time-step it considers
            _network->setCurrentTimestep(t);

            // Convert the last state to an Eigen vector
            Vector last_state;

            episode->encodedState(t, rs);
            NnetModel::vectorToVector(rs, last_state);

            // Feed this input to the network
            Vector prediction = _network->predict(last_state);

            // If this is the last time-step to predict, copy its output to rs
            if (t == episode->length() - 1) {
                rs.resize(episode->valueSize());

                for (std::size_t i=0; i<rs.size(); ++i) {
                    rs[i] = prediction(i);
                }
            }
        }

        _last_episode_length = episode->length();
    }
}

void RecurrentNnetModel::hiddenValues(Episode *episode, std::vector<float> &rs)
{
    // Call values(), that takes care of the actual prediction
    values(episode, rs);

    // The hidden state is simply the output of a "hidden unit" of the network
    const Vector &o = hiddenNode()->output()->value;

    rs.resize(o.rows());

    for (std::size_t i=0; i<rs.size(); ++i) {
        rs[i] = o(i);
    }
}

void RecurrentNnetModel::learn(const std::vector<Episode *> &episodes)
{
    std::vector<float> state;
    std::vector<float> values;

    // Learn all the episodes separately, because they represent sequences
    // of observations that must be kept in order
    for (int i=0; i<50; ++i) {
        for (Episode *episode : episodes) {
            // Create the network if needed
            if (!_network) {
                _network = createNetwork(episode);
            }

            // Learn all the values obtained during the episode
            unsigned int size = episode->length() - 1;

            Eigen::MatrixXf inputs(episode->encodedStateSize(), size);
            Eigen::MatrixXf outputs(episode->valueSize(), size);

            for (unsigned int t=0; t < episode->length() - 1; ++t) {
                episode->encodedState(t, state);
                episode->values(t, values);

                NnetModel::vectorToCol(state, inputs, t);
                NnetModel::vectorToCol(values, outputs, t);
            }

            // Train the network on that data
            _network->trainSequence(inputs, outputs, 1);
        }
    }
}
