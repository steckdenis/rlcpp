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

#include "texploremodel.h"
#include "modelworld.h"

#include "model/episode.h"

TExploreModel::TExploreModel(AbstractWorld *world,
                             AbstractModel *world_model,
                             AbstractModel *values_model,
                             AbstractLearning *learning,
                             unsigned int rollout_length)
: _world(new ModelWorld(world, world_model)),
  _model(values_model),
  _learning(learning),
  _rollout_length(rollout_length)
{
}

void TExploreModel::values(Episode *episode, std::vector<float> &rs)
{
    std::vector<float> &state = rs;

    // Perform some rollouts from the current state
    unsigned int num_rollouts = 3;
    unsigned int batch_size = 1;

    episode->state(episode->length() - 1, state);

    std::vector<Episode *> episodes = _world->run(_model,
                                                  _learning,
                                                  num_rollouts,
                                                  _rollout_length,
                                                  batch_size,
                                                  false,
                                                  episode);

    for (Episode *e : episodes) {
        delete e;   // Don't leak the rollout episodes
    }

    // Use the model trained by the rollouts to predict the values
    _model->values(episode, rs);
}

void TExploreModel::valuesForPlotting(Episode *episode, std::vector<float> &rs)
{
    _model->valuesForPlotting(episode, rs);
}

void TExploreModel::learn(const std::vector<Episode *> &episodes)
{
    _world->learn(episodes);
}
