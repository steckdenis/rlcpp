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

#include "dynamodel.h"
#include "modelworld.h"

#include "model/episode.h"

DynaModel::DynaModel(AbstractWorld *world,
                     AbstractModel *world_model,
                     AbstractModel *values_model,
                     AbstractLearning *learning,
                     unsigned int rollout_length,
                     unsigned int num_rollouts,
                     Episode::Encoder encoder)
: _world(new ModelWorld(world, world_model, encoder)),
  _model(values_model),
  _learning(learning),
  _encoder(encoder),
  _rollout_length(rollout_length),
  _num_rollouts(num_rollouts)
{
}

void DynaModel::values(Episode *episode, std::vector<float> &rs)
{
    // Perform some rollouts from the current state
    std::vector<Episode *> episodes = _world->run(_model,
                                                  _learning,
                                                  _num_rollouts,
                                                  _rollout_length,
                                                  _num_rollouts,
                                                  _encoder,
                                                  false,
                                                  episode);

    for (Episode *e : episodes) {
        delete e;   // Don't leak the rollout episodes
    }

    // Use the model trained by the rollouts to predict the values
    _model->values(episode, rs);
}

void DynaModel::valuesForPlotting(Episode *episode, std::vector<float> &rs)
{
    _model->valuesForPlotting(episode, rs);
}

void DynaModel::learn(const std::vector<Episode *> &episodes)
{
    _model->learn(episodes);
    _world->learn(episodes);
}
