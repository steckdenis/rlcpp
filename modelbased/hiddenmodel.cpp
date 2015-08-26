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

#include "hiddenmodel.h"
#include "modelworld.h"

#include "model/episode.h"

#include <iostream>
#include <assert.h>

HiddenModel::HiddenModel(AbstractWorld *world,
                         AbstractModel *world_model,
                         AbstractModel *values_model)
: _world(new ModelWorld(world, world_model, nullptr, false)),
  _model(values_model)
{
}

HiddenModel::~HiddenModel()
{
    // Clear the list of episodes
    nextEpisode();
}

void HiddenModel::values(Episode *episode, std::vector<float> &rs)
{
    Episode *model_episode;
    unsigned int start_t;

    // Start a new episode if needed
    if (_real_episodes.size() == 0 || _real_episodes.back() != episode) {
        model_episode = new Episode(episode->valueSize(), episode->numActions(), nullptr);  // No need for an encoder, state representations are already compatible with neural networks.

        _model_episodes.push_back(model_episode);
        _real_episodes.push_back(episode);
        _world->reset();

        start_t = 0;
    } else {
        model_episode = _model_episodes.back();
        start_t = model_episode->length() - 1;
    }

    // Episode contains states and actions. Use _world->stepSupervised in order
    // to reach the last state of the episode. This produces a prediction of the
    // next state representation, which can be used as state in _episodes.
    for (unsigned int t=start_t; t<episode->length()-1; ++t) {
        unsigned int action = episode->action(t);
        float reward = episode->reward(t);

        episode->state(t + 1, rs);

        _world->stepSupervised(action, rs, reward);
    }

    // Use _world is now at time-step episode->length()-2, and has tried to predict
    // episode->length()-1. Its hidden values therefore represent a state representation
    // for the current state.
    std::cout << "State submitted to model:";
    for (float v : rs) {
        std::cout << ' ' << v;
    }

    _world->lastHiddenValues(rs);

    std::cout << " -> ";
    for (float v : rs) {
        std::cout << ' ' << v;
    }
    std::cout << std::endl;

    if (episode->length() == 1) {
        // Initial state set to the null state
        std::fill(rs.begin(), rs.end(), 0.0f);
    }

    // model_episode now contains the last state representation. Use it to predict
    // the action values
    model_episode->addState(rs);
    _model->values(model_episode, rs);
    model_episode->addValues(rs);
}

void HiddenModel::learn(const std::vector<Episode *> &episodes)
{
    // The world learns using the episodes, that contain real observations
    _world->learn(episodes);

    // The model learns using the episodes built by values(), but the values of
    // each episode has to be adjusted, because the learning algorithm (unknown
    // to this class) might have changed values in episodes
    assert(episodes.size() == _model_episodes.size());

    for (std::size_t i=0; i<episodes.size(); ++i) {
        Episode *real_episode = episodes[i];
        Episode *model_episode = _model_episodes[i];

        assert(real_episode->length() == model_episode->length());

        // Copy the values from the real episode to the model episode
        model_episode->copyValues(*real_episode);
        model_episode->copyActions(*real_episode);
        model_episode->copyRewards(*real_episode);
    }

    _model->learn(_model_episodes);
}

void HiddenModel::nextEpisode()
{
    // Clear the old list of episodes and start afresh
    for (Episode *e : _model_episodes) {
        delete e;
    }

    _model_episodes.clear();
    _real_episodes.clear();         // They are owned by AbstractWorld, don't delete them
}
