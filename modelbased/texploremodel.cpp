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

#include <atomic>
#include <unistd.h>

TEXPLOREModel::TEXPLOREModel(AbstractWorld *world,
                             AbstractModel *world_model,
                             AbstractModel *values_model,
                             AbstractLearning *learning,
                             unsigned int rollout_length,
                             Episode::Encoder encoder)
: _world(new ModelWorld(world, world_model, encoder, false)),
  _model(values_model),
  _world_model(world_model),
  _learning(learning),
  _encoder(encoder),
  _rollout_length(rollout_length),
  _finish(false),
  _base_episode(nullptr),
  _update_world_thread(&TEXPLOREModel::updateWorldThread, this),
  _update_model_thread(&TEXPLOREModel::updateModelThread, this)
{
}

TEXPLOREModel::~TEXPLOREModel()
{
    // Tell the threads that they should finish
    _finish = true;

    // Join the threads
    _update_world_thread.join();
    _update_model_thread.join();

    // Free any memory that has to be freed
    if (_base_episode) {
        delete _base_episode;
    }

    for (Episode *e : _base_episodes) {
        delete e;
    }

    for (Episode *e : _world_episodes) {
        delete e;
    }
}

void TEXPLOREModel::updateWorldThread()
{
    std::vector<Episode *> episodes;

    while (!_finish) {
        // Copy _world_episodes to episodes, so that learning can happen on a
        // list of episodes that will not change and is not accessed by other threads.
        {
            std::unique_lock<std::mutex> lock(_world_episodes_lock);

            while (_world_episodes.size() == 0) {
                _world_episodes_cond.wait(lock);
            }

            episodes = _world_episodes;
            _world_episodes.clear();
        }

        // Learn
        _world->learn(episodes);

        // Delete the copy of the episodes that this thread has as they are no
        // longer used.
        for (Episode *e : episodes) {
            delete e;
        }

        // Use the new world for the rollouts
        {
            std::unique_lock<std::mutex> lock(_world_lock);

            _world_model->swapModels();
        }
    }
}

void TEXPLOREModel::updateModelThread()
{
    std::vector<Episode *> episodes;

    // Perform rollouts until the thread has to end
    while (!_finish) {
        {
            std::unique_lock<std::mutex> lock(_world_lock);     // _world->run() uses the world model, that must therefore be locked.

            episodes = _world->run(_model,
                                   _learning,
                                   1,
                                   _rollout_length,
                                   1,
                                   _encoder,
                                   false,
                                   false,
                                   _base_episode);
        }

        for (Episode *e : episodes) {
            delete e;   // Don't leak the rollout episodes
        }

        // Swap the models so that the main thread can use updated action values
        {
            std::unique_lock<std::mutex> lock(_model_lock);

            _model->swapModels();

            // Delete the base episodes that are not needed anymore (the latest
            // episode is in _base_episode but not _base_episodes, so it will
            // not be deleted). _model_lock is the lock used by values(), which
            // explains why these deletions are performed here.
            for (Episode *e : _base_episodes) {
                delete e;
            }

            _base_episodes.clear();
        }
    }
}

void TEXPLOREModel::values(Episode *episode, std::vector<float> &rs)
{
    // Small sleep in order to avoid predictions that are so fast that no
    // rollout can be performed between them
    usleep(200);

    // Use the model trained by the rollouts to predict the values
    std::unique_lock<std::mutex> lock(_model_lock);

    _model->values(episode, rs);

    // Swap _base_episode with a new copy of episode, so that rollouts start
    // at the latest position in the world.
    Episode *old_episode = _base_episode.exchange(new Episode(*episode));

    if (old_episode) {
        // Keep track of all the base episodes so that they can be deleted when
        // not needed anymore
        _base_episodes.push_back(old_episode);
    }
}

void TEXPLOREModel::valuesForPlotting(Episode *episode, std::vector<float> &rs)
{
    // Tell the threads to exit, no more rollouts are needed. This reduces contention
    // on the locks and speeds-up plotting
    _finish = true;

    values(episode, rs);
}

void TEXPLOREModel::learn(const std::vector<Episode *> &episodes)
{
    std::unique_lock<std::mutex> lock(_world_episodes_lock);

    // Add the episodes to the list of episodes from which the world has to learn
    for (Episode *e : episodes) {
        _world_episodes.push_back(new Episode(*e));
    }

    // Tell the world thread that it can resume learning
    _world_episodes_cond.notify_one();
}

void TEXPLOREModel::swapModels()
{
    // Nothing to do, the threads already swap models when needed
}
