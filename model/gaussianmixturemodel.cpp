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

#include "gaussianmixturemodel.h"
#include "episode.h"
#include "functionapproximators/gaussianmixture.h"

#include <algorithm>
#include <random>
#include <iostream>

GaussianMixtureModel::GaussianMixtureModel(float var_initial, float novelty, float noise, bool mask_actions)
: _var_initial(var_initial),
  _novelty(novelty),
  _mask_actions(mask_actions),
  _noise_distribution(0.0f, noise)
{
}

GaussianMixtureModel::~GaussianMixtureModel()
{
    // Delete all the models
    for (GaussianMixture *model : _models) {
        delete model;
    }
}

void GaussianMixtureModel::values(Episode *episode, std::vector<float> &rs)
{
    if (_models.size() == 0) {
        // No model available, clear out rs
        rs.resize(episode->valueSize());
        std::fill(rs.begin(), rs.end(), 0.0f);
    } else {
        // Convert the last state to an Eigen vector
        Eigen::VectorXf input(episode->stateSize());

        episode->state(episode->length() - 1, rs);
        vectorToVectorXf(rs, input);

        // Pass this input to all the models
        rs.resize(episode->valueSize());

        for (std::size_t i=0; i<rs.size(); ++i) {
            rs[i] = _models[i]->value(input);
        }
    }
}

void GaussianMixtureModel::learn(const std::vector<Episode *> &episodes)
{
    std::vector<float> state;
    std::vector<float> values;

    for (Episode *episode : episodes) {
        Eigen::VectorXf input(episode->stateSize());

        // Create the models if needed
        if (_models.size() == 0) {
            for (unsigned int a=0; a<episode->valueSize(); ++a) {
                _models.push_back(new GaussianMixture(_var_initial, _novelty));
            }
        }

        // Train the models on this episode
        for (unsigned int t=0; t < episode->length() - 1; ++t) {
            unsigned int action = episode->action(t);

            episode->state(t, state);
            episode->values(t, values);

            vectorToVectorXf(state, input);

            if (_mask_actions) {
                // Update the model of the selected action
                _models[action]->setValue(input, values[action]);
            } else {
                // Update all the models
                for (unsigned int a=0; a<episode->valueSize(); ++a) {
                    _models[a]->setValue(input, values[a]);
                }
            }
        }
    }

    // Print the number of clusters in the models
    std::cout << "[Gaussian mixture model] Number of clusters:";

    for (GaussianMixture *model : _models) {
        std::cout << ' ' << model->numberOfClusters();
    }

    std::cout << std::endl;
}

void GaussianMixtureModel::vectorToVectorXf(const std::vector<float> &stl, Eigen::VectorXf &eigen)
{
    for (std::size_t i=0; i<stl.size(); ++i) {
        // Add a bit of noise in order to avoid having vectors too close to each
        // other, and hence having a null variance.
        eigen(i) = stl[i] + _noise_distribution(_random_engine);
    }
}

