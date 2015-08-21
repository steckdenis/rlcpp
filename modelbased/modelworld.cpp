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

ModelWorld::ModelWorld(AbstractWorld *world, AbstractModel *model, Episode::Encoder encoder)
: AbstractWorld(world->numActions()),
  _world(world),
  _model(model),
  _encoder(encoder),
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
    unsigned int value_size = _world_state.size() + 2;  // Predict state gradient, reward and finished

    if (_episode) {
        delete _episode;
    }

    _episode = new Episode(value_size, value_size, _encoder);
}

void ModelWorld::step(unsigned int action,
                      bool &finished,
                      float &reward,
                      std::vector<float> &state)
{
    unsigned int state_size = _world_state.size();

    // Add the current state and the action to the episode
    makeModelState(_world_state, action, _model_state);

    _episode->addState(_model_state);
    _episode->addAction(action);

    // Use this updated episode to predict the next state
    _model->values(_episode, _values);

    for (unsigned int i=0; i<state_size; ++i) {
        _world_state[i] += _values[i];          // Add delta to the current state in order to get the next one
    }

    // Update the output parameters
    reward = _values[state_size];
    finished = (_values[state_size + 1] > 0.5f);
    state = _world_state;

    // And complete the episode
    _episode->addReward(reward);
    _episode->addValues(_values);
}

void ModelWorld::stepSupervised(unsigned int action,
                                const std::vector<float> &target_state,
                                float reward)
{
    // Dummy values
    unsigned int value_size = _world_state.size() + 2;

    _values.resize(value_size);
    std::fill(_values.begin(), _values.end(), 0.0f);

    _values[value_size - 2] = reward;

    // Add the current state to the episode
    makeModelState(_world_state, action, _model_state);

    _episode->addState(_model_state);
    _episode->addAction(action);
    _episode->addReward(reward);
    _episode->addValues(_values);

    // Copy the target state to _world_state, which will be added to the episode
    // at the next step (and hence represents the target state)
    _world_state = target_state;
}

void ModelWorld::learn(const std::vector<Episode *> episodes)
{
    std::vector<Episode *> model_episodes;

    // Convert "world episodes" to "model episodes". This conversion basically
    // consists in converting state sequences to state deltas.
    unsigned int value_size = _world_state.size() + 2;
    std::vector<float> next_state;
    std::vector<float> state;

    for (Episode *episode : episodes) {
        Episode *model_episode = new Episode(value_size, value_size, _encoder);

        for (unsigned int t = 0; t < episode->length() - 1; ++t) {
            unsigned int action = episode->action(t);
            float reward = episode->reward(t);
            bool finished = false;

            episode->state(t, state);
            episode->state(t + 1, next_state);

            // Compute the state delta (between unencoded states)
            for (std::size_t i=0; i<state.size(); ++i) {
                _values[i] = next_state[i] - state[i];
            }

            // If we have reached the end of the episode, see whether it was
            // because we have reached the goal or because the time limit has
            // been reached.
            if (t == episode->length() - 2) {
                model_episode->setAborted(episode->wasAborted());
                finished = !episode->wasAborted();
            }

            _values[value_size - 2] = reward;
            _values[value_size - 1] = (finished ? 1.0f : 0.0f);

            // Add the state delta and expected prediction to the model episode
            makeModelState(state, action, _model_state);

            model_episode->addState(_model_state);
            model_episode->addAction(action);
            model_episode->addReward(reward);
            model_episode->addValues(_values);
            model_episode->setAborted(episode->wasAborted());
        }

        model_episodes.push_back(model_episode);
    }

    // Train the model on the new episodes
    _model->learn(model_episodes);

    for (Episode *episode : model_episodes) {
        delete episode;
    }
}

void ModelWorld::makeModelState(const std::vector<float> &world_state,
                                unsigned int action,
                                std::vector<float> &model_state)
{
    // Copy the state, and add one variable for the action (an encoder
    // may encode this variable if needed by the model)
    model_state = world_state;

    model_state.resize(world_state.size() + 1);
    model_state[world_state.size()] = float(int(action));
}
