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

#include "fusionartmodel.h"
#include "episode.h"

#include <algorithm>

FusionARTModel::FusionARTModel(bool mask_actions)
: _mask_actions(mask_actions),
  _prediction_model(nullptr),
  _learning_model(nullptr)
{
}

FusionARTModel::~FusionARTModel()
{
    if (_learning_model) {
        delete _learning_model;
    }

    if (_prediction_model) {
        delete _prediction_model;
    }
}

void FusionARTModel::swapModels()
{
    std::unique_lock<std::mutex> lock(_mutex);
    std::swap(_learning_model, _prediction_model);
}

void FusionARTModel::values(Episode *episode, std::vector<float> &rs)
{
    if (!_prediction_model) {
        // No model available, clear out rs
        rs.resize(episode->valueSize());
        std::fill(rs.begin(), rs.end(), 0.0f);
    } else {
        std::unique_lock<std::mutex> lock(_mutex);

        // Put the state somewhere, it will be used for predicting the value of each action
        episode->encodedState(episode->length() - 1, _state);

        // Predict the value of all the actions
        rs.resize(episode->valueSize());

        for (unsigned int a=0; a<episode->valueSize(); ++a) {
            // Put the last state in the state port (that is overwritten by every
            // call to run()).
            vectorToArrayXf(_state, _prediction_model->state.value);

            // One-hot encoding of the action
            _prediction_model->action.value.setZero();
            _prediction_model->action.value(a) = 1.0f;

            // Clear the value port
            _prediction_model->value.value.setOnes();

            // Run the model without learning
            _prediction_model->model.run(false);

            // v0 / v1 gives the value, v2 - v3 gives the sign
            Eigen::ArrayXf &value = _prediction_model->value.value;

            rs[a] = (value(2) - value(3)) * value(0) / value(1);
        }
    }
}

void FusionARTModel::learn(const std::vector<Episode *> &episodes)
{
    std::vector<float> state;
    std::vector<float> values;

    // Create the model if needed
    if (!_learning_model) {
        _learning_model = new Model;

        _learning_model->model.addPort(&_learning_model->state);
        _learning_model->model.addPort(&_learning_model->action);
        _learning_model->model.addPort(&_learning_model->value);

        _learning_model->state.value.resize(episodes[0]->encodedStateSize());
        _learning_model->state.weight = 0.5f;
        _learning_model->state.vigilence = 0.6f;
        _learning_model->action.value.resize(episodes[0]->valueSize());
        _learning_model->action.weight = 0.5f;
        _learning_model->action.vigilence = 0.7f;
        _learning_model->value.value.resize(4);                     // Q(s, a) = v0 / v1 gives the absolute value, v2 - v3 gives the sign.
        _learning_model->value.weight = 0.0f;                       // --- The value is what has to be learned, so don't use it to discriminate clusters.
        _learning_model->value.vigilence = 0.4f;                    // -/
    }

    if (_prediction_model) {
        // Synchronize the learning model with the prediction model so that learning
        // builds on up-to-date data.
        std::unique_lock<std::mutex> lock(_mutex);

        _learning_model->model.copyFrom(_prediction_model->model);
    }

    for (Episode *episode : episodes) {
        // Train the models on this episode
        for (unsigned int t=0; t < episode->length() - 1; ++t) {
            unsigned int action = episode->action(t);

            episode->encodedState(t, state);
            episode->values(t, values);

            // Learn all the actions
            for (unsigned int a=0; a<episode->valueSize(); ++a) {
                if (_mask_actions && a != action) {
                    continue;
                }

                // Put the state in the state port
                vectorToArrayXf(state, _learning_model->state.value);

                // One-hot encode the action
                _learning_model->action.value.setZero();
                _learning_model->action.value(a) = 1.0f;

                // Encode the value
                Eigen::ArrayXf &value = _learning_model->value.value;
                float v = values[a];

                if (v < 0) {
                    value(2) = 0.0f;
                    value(3) = 1.0f;
                    v = -v;
                } else {
                    value(2) = 1.0f;
                    value(3) = 0.0f;
                }

                if (v < 1) {
                    // value < 1 -> value / 1 = value, with both terms smaller than one
                    value(0) = v;
                    value(1) = 1.0f;
                } else {
                    // value > 1 -> 1 / (1 / value) = value, with both terms smaller than one
                    value(0) = 1.0f;
                    value(1) = 1.0f / v;
                }

                // Train the model
                _learning_model->model.run(true);
            }
        }
    }
}

void FusionARTModel::vectorToArrayXf(const std::vector<float> &stl, Eigen::ArrayXf &eigen)
{
    for (std::size_t i=0; i<stl.size(); ++i) {
        eigen(i) = stl[i];
    }
}

