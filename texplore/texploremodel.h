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

class AbstractWorld;
class AbstractLearning;
class ModelWorld;

/**
 * @brief Model based on TExplore (Hester, 2013), that learns a model of the world
          and use it to produce Q or Advantage values.
 */
class TExploreModel : public AbstractModel
{
    public:
        /**
         * @brief Constructor
         *
         * @param world World in which the agent runs (the "real" world)
         * @param world_model Model used to approximate the real world
         * @param values_model Model used to approximate the Q or Advantage values
         *                     obtained by the rollouts.
         * @param learning Learning algorithm used to perform the rollouts. Should
         *                 be the same one as the one used in the real world, so that
         *                 values produced by the rollouts are meaningful in the
         *                 real world.
         * @param rollout_length Length of the rollouts, in time steps.
         * @param encoder State encoder used during the rollouts, if any. Must be
         *                the same encoder as the one used in the "real" world.
         */
        TExploreModel(AbstractWorld *world,
                      AbstractModel *world_model,
                      AbstractModel *values_model,
                      AbstractLearning *learning,
                      unsigned int rollout_length,
                      Episode::Encoder encoder = nullptr);

        virtual void values(Episode *episode, std::vector<float> &rs);
        virtual void valuesForPlotting(Episode *episode, std::vector<float> &rs);
        virtual void learn(const std::vector<Episode *> &episodes);

    private:
        ModelWorld *_world;
        AbstractModel *_model;
        AbstractLearning *_learning;
        Episode::Encoder _encoder;
        unsigned int _rollout_length;
};

#endif
