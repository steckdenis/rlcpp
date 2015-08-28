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

#ifndef __HIDDENMODEL_H__
#define __HIDDENMODEL_H__

#include "model/abstractmodel.h"
#include "model/episode.h"

class AbstractWorld;
class ModelWorld;

/**
 * @brief Model that uses ModelWorld in order to predict the hidden state of
 *        a partially observable environment.
 *
 * If ModelWorld uses a recurrent neural network model for predicting the next
 * state and reward, the output of the hidden layer of the neural network can
 * be used as a fully-observable state representation. This allows simpler models
 * to map state representations to action values.
 *
 * This is close to what actor-critic learning is considered to be, even if it
 * does not completely fits the actor-critic framework.
 */
class HiddenModel : public AbstractModel
{
    public:
        /**
         * @brief Constructor
         *
         * @param world World in which the agent runs (the "real" world)
         * @param world_model Model used to approximate the real world. Be sure that
         *                    this model does not mask actions, if applicable
         *                    (PerceptronModel, GaussianMixtureModel). This model
         *                    should be a recurrent neural network if the world
         *                    is partially observable.
         * @param values_model Model used to approximate the Q or Advantage values.
         *                     It is wrapped by this model and receives state
         *                     representations instead of observations.
         */
        HiddenModel(AbstractWorld *world,
                    AbstractModel *world_model,
                    AbstractModel *values_model);
        virtual ~HiddenModel();

        virtual void values(Episode *episode, std::vector<float> &rs);
        virtual void learn(const std::vector<Episode *> &episodes);

    private:
        ModelWorld *_world;
        AbstractModel *_model;

        std::vector<Episode *> _model_episodes;
        std::vector<Episode *> _real_episodes;
};

#endif
