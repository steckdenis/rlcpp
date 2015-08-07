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

#include "modelworld.h"
#include "model/episode.h"
#include "model/abstractmodel.h"

ModelWorld::ModelWorld(AbstractWorld *world, AbstractModel *model)
: AbstractWorld(world->numActions()),
  _world(world),
  _model(model),
  _episode(nullptr)
{
}

ModelWorld::~ModelWorld()
{
    if (_episode) {
        delete _episode;
    }
}

void ModelWorld::initialState(std::vector<float> &state)
{
    // Return the same initial state as the one of the wrapped world
    _world->initialState(state);
}

void ModelWorld::reset()
{
    // Fetch the initial state of the world
    initialState(_world_state);

    // Start a new episode
    unsigned int value_size = _world_state.size() + 1;  // Predict state gradient and reward

    if (_episode) {
        delete _episode;
    }

    _episode = new Episode(value_size, value_size);
}

void ModelWorld::step(unsigned int action,
                      bool &finished,
                      float &reward,
                      std::vector<float> &state)
{
    // Add the current state and the action to the episode
    makeModelState(_world_state, action, _model_state);

    _episode->addState(_model_state);
    _episode->addAction(action);

    // Use this updated episode to predict the next state
    _model->values(_episode, _values);

    for (std::size_t i=0; i<_world_state.size(); ++i) {
        _world_state[i] += _values[i];          // Add delta to the current state in order to get the next one
    }

    // Update the output parameters
    finished = false;
    reward = _values.back();
    state = _world_state;

    // And complete the episode
    _episode->addReward(reward);
    _episode->addValues(_values);
}

void ModelWorld::stepSupervised(unsigned int action,
                                const std::vector<float> &target_state)
{
    // Perform the action as usual, so that the episode is complete (and even
    // has near-valid rewards if a model uses this information)
    bool finished;
    float reward;

    step(action, finished, reward, _world_state);

    // Copy the target state to _world_state, which will be added to the episode
    // at the next step (and hence represents the target state)
    _world_state = target_state;
}

void ModelWorld::learn(const std::vector<Episode *> episodes)
{
    std::vector<Episode *> model_episodes;

    // Convert "world episodes" to "model episodes". This conversion basically
    // consists in converting state sequences to state deltas.
    unsigned int value_size = _world_state.size() + 1;
    std::vector<float> next_state;
    std::vector<float> state;

    for (Episode *episode : episodes) {
        Episode *model_episode = new Episode(value_size, value_size);

        for (unsigned int t = 0; t < episode->length() - 1; ++t) {
            unsigned int action = episode->action(t);
            float reward = episode->reward(t);

            episode->state(t, state);
            episode->state(t + 1, next_state);

            // Compute the state delta
            for (std::size_t i=0; i<state.size(); ++i) {
                _values[i] = next_state[i] - state[i];
            }

            _values[state.size()] = reward;

            // Add the state delta and expected prediction to the model episode
            makeModelState(state, action, _model_state);

            model_episode->addState(_model_state);
            model_episode->addAction(action);
            model_episode->addReward(reward);
            model_episode->addValues(_values);
        }

        model_episodes.push_back(model_episode);
    }

    // Train the model on the new episodes
    _model->learn(episodes);

    for (Episode *episode : model_episodes) {
        delete episode;
    }
}

void ModelWorld::makeModelState(const std::vector<float> &world_state,
                                unsigned int action,
                                std::vector<float> &model_state)
{
    // Copy the state, and add one variable per possible action (one-hot encoding)
    model_state.resize(world_state.size() + numActions());

    for (std::size_t i=0; i<model_state.size(); ++i) {
        if (i < world_state.size()) {
            model_state[i] = world_state[i];
        } else if (i - world_state.size() == action) {
            model_state[i] = 1.0f;
        } else {
            model_state[i] = 0.0f;
        }
    }
}
