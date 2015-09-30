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

#include "psrmodel.h"
#include "nnetmodel.h"
#include "episode.h"

#include <iostream>
#include <random>
#include <cmath>

PSRModel::PSRModel(unsigned int history_length,
                   unsigned int test_length,
                   unsigned int rank,
                   unsigned int random_features)
: _history_length(history_length),
  _test_length(test_length),
  _rank(rank),
  _random_features(random_features),
  _psr(nullptr),
  _last_episode(nullptr)
{
}

PSRModel::~PSRModel()
{
    if (_psr) {
        delete _psr;
    }
}

void PSRModel::swapModels()
{
    // TODO
}

void PSRModel::values(Episode *episode, std::vector<float> &rs)
{
    rs.resize(episode->valueSize());

    if (!_psr) {
        // No model available, clear out rs
        std::fill(rs.begin(), rs.end(), 0.0f);
    } else {
        // Reset the PSR model if needed
        if (episode != _last_episode || _last_episode_length >= episode->length()) {
            _last_episode_length = 0;
            _psr->reset();
        }

        _last_episode_length = episode->length();
        _last_episode = episode;

        // Update PSR with the action-observation-values of the last time-step
        if (episode->length() > 1) {
            _psr->update(
                makeSequence(episode, episode->length()-2, episode->length()-1)
            );
        }

        // episode[-1] contains a state, but no action nor value yet. Try all the
        // actions to get the values
        for (unsigned int a=0; a<episode->valueSize(); ++a) {
            rs[a] = valueOfAction(episode, a);
        }
    }
}

float PSRModel::valueOfAction(Episode *episode, unsigned int action)
{
    Eigen::VectorXf ao(3 * _random_features);
    Eigen::VectorXf eigen;
    std::vector<float> state;

    episode->state(episode->length()-1, state);
    NnetModel::vectorToVector(state, eigen);

    // Encode the current state and action in the action-observation pair
    encodeVector(ao, 0*_random_features, eigen);

    eigen.resize(1);
    eigen(0) = float(action);

    encodeVector(ao, 1*_random_features, eigen);

    // Try different values
    float rs = 0.0f;
    float max_proba = -100.0f;

    for (float v=-10.0f; v<20.0f; v+=0.1f) {
        eigen(0) = v;
        encodeVector(ao, 2*_random_features, eigen);

        // Predict the probability of this value
        float proba = _psr->predict(ao);

        if (proba > max_proba) {
            rs = v;
            max_proba = proba;
        }
    }

    return rs;
}

void PSRModel::learn(const std::vector<Episode *> &episodes)
{
    unsigned int ao_length = 3 * _random_features;  // action, observation, values

    // Create the model if needed
    if (!_psr) {
        _psr = new PSR(_history_length * ao_length, ao_length, _test_length * ao_length, _rank);

        // Create random features for the state
        _features.reserve(_random_features);

        for (unsigned int i=0; i<_random_features; ++i) {
            _features.push_back(Eigen::VectorXf::Random(episodes[0]->stateSize()) * 10.0f);
        }
    }

    // Create histories, action-observations and tests for all the episodes
    for (Episode *episode : episodes) {
        for (unsigned int t=_history_length; t<episode->length()-_test_length-1; ++t) {
            _psr->train(
                makeSequence(episode, t-_history_length, t),
                makeSequence(episode, t, t+1),
                makeSequence(episode, t+1, t+_test_length+1)
            );
        }
    }
}

Eigen::VectorXf PSRModel::makeSequence(Episode *episode,
                                       unsigned int from,
                                       unsigned int to)
{
    Eigen::VectorXf eigen;
    std::vector<float> vector;

    unsigned int length = to - from;
    unsigned int step_size = 3 * _random_features;
    Eigen::VectorXf rs(length * step_size);

    for (unsigned int t=0; t<length; ++t) {
        // Encode the state
        episode->state(from + t, vector);
        NnetModel::vectorToVector(vector, eigen);

        encodeVector(rs, t*step_size + 0*_random_features, eigen);

        // Encode the action
        unsigned int action = episode->action(from + t);

        eigen.resize(1);
        eigen(0) = float(action);

        encodeVector(rs, t*step_size + 1*_random_features, eigen);

        // Encode the value of the action that was taken
        episode->values(from + t, vector);
        eigen(0) = vector[action];

        encodeVector(rs, t*step_size + 2*_random_features, eigen);
    }

    return rs;
}

static float gaussianKernel(const Eigen::VectorXf &a, const Eigen::VectorXf &b)
{
    return std::exp(-(a - b).squaredNorm() / 2.0f);
}

void PSRModel::encodeVector(Eigen::VectorXf &sequence,
                            unsigned int offset,
                            const Eigen::VectorXf &x)
{
#if 0
    Eigen::VectorXf random_vector(x.rows());
    std::default_random_engine engine;
    std::normal_distribution<float> normal(0.0f, 1.0f);
    float normalization = 1.0f / x.rows();

    for (int i=0; i<_random_features/2; ++i) {
        // Make a random vector (following a normal distribution so that the gaussian
        // kernel is properly approximated by the dot product between random
        // features)
        for (int j=0; j<x.rows(); ++j) {
            random_vector(j) = normal(engine);
        }

        // Build the random feature
        float v = random_vector.dot(x);

        sequence(offset + i * 2) = normalization * std::sin(v);
        sequence(offset + i * 2 + 1) = normalization * std::cos(v);
    }
#else
    for (int i=0; i<_random_features; ++i) {
        if (x.cols() == 1) {
            // Single value, built the kernel on the fly
            float kernel = float(i - (_random_features / 2)) * 0.5f;
            float delta = x(0) - kernel;
            float val = std::exp(-delta*delta / 2.0f);

            sequence(offset + i) = val;
        } else {
            assert(_features.size() > 0);

            // Use the gaussian kernel
            sequence(offset + i) = gaussianKernel(x, _features[i]);
        }
    }
#endif

    // The first feature must be a one
    sequence(offset) = 1.0f;
}
