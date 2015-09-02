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

#ifndef __TEXPLOREMODEL_H__
#define __TEXPLOREMODEL_H__

#include "model/abstractmodel.h"
#include "model/episode.h"

#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>

class AbstractWorld;
class AbstractLearning;
class ModelWorld;

/**
 * @brief Model based on TEXPLORE. This model acts like DynaModel but uses separate
 *        concurrent threads for updating the world model, the action values model,
 *        and selecting actions.
 */
class TEXPLOREModel : public AbstractModel
{
    public:
        /**
         * @brief Constructor
         * @sa DynaModel::DynaModel
         */
        TEXPLOREModel(AbstractWorld *world,
                      AbstractModel *world_model,
                      AbstractModel *values_model,
                      AbstractLearning *learning,
                      unsigned int rollout_length,
                      Episode::Encoder encoder = nullptr);
        ~TEXPLOREModel();

        virtual void values(Episode *episode, std::vector<float> &rs);
        virtual void valuesForPlotting(Episode *episode, std::vector<float> &rs);
        virtual void learn(const std::vector<Episode *> &episodes);
        virtual void swapModels();

    private:
        void updateWorldThread();
        void updateModelThread();

    private:
        ModelWorld *_world;
        AbstractModel *_model;
        AbstractModel *_world_model;
        AbstractLearning *_learning;
        Episode::Encoder _encoder;
        unsigned int _rollout_length;

        std::atomic<bool> _finish;
        std::vector<Episode *> _world_episodes;
        std::vector<Episode *> _base_episodes;
        std::atomic<Episode *> _base_episode;                       // Episode from which rollouts are performed

        std::mutex _base_episodes_lock;
        std::mutex _world_episodes_lock;
        std::condition_variable _world_episodes_cond;

        std::thread _update_world_thread;
        std::thread _update_model_thread;
};

#endif
