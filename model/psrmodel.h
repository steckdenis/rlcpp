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

#ifndef __PSRMODEL_H__
#define __PSRMODEL_H__

#include "abstractmodel.h"
#include "functionapproximators/psr.h"

#include <mutex>
#include <vector>

/**
 * @brief Model representing its belief state using Predictive State Representations
 *
 * This model is able to model sequences and can therefore be used in partially
 * observable environments.
 */
class PSRModel : public AbstractModel
{
    public:
        /**
         * @brief Constructor
         *
         * @param history_length Number of action-observations used to build an
         *                       history.
         * @param observation_length Number of actions-observations used to build
         *                           a future.
         * @param rank "Complexity" of the model, should be lower than history_length
         *             and observation_length. This is roughly the number of hidden
         *             states in a POMDP.
         * @param random_features Number of random features used to encode
         *                        states, actions and values
         */
        PSRModel(unsigned int history_length,
                 unsigned int test_length,
                 unsigned int rank,
                 unsigned int random_features);
        virtual ~PSRModel();

        virtual void values(Episode *episode, std::vector<float> &rs);
        virtual void learn(const std::vector<Episode *> &episodes);
        virtual void swapModels();

    private:
        /**
         * @brief Make a vector containing a sequence of action-observations
         *
         * @param episode Episode containing actions and observations
         * @param from First time-step from the episode that will be copied
         *             to @p sequence
         * @param to Last time-step (exluded) to be copied to @p sequence
         *
         * @param Vector containing encoded action-observations
         */
        Eigen::VectorXf makeSequence(Episode *episode,
                                     unsigned int from,
                                     unsigned int to);

        /**
         * @brief Encode a vector using random features
         *
         * @param sequence Vector that will receive the encoded features
         * @param offset Offset in @p sequence at which the features are placed
         * @param x Vector to encode
         */
        void encodeVector(Eigen::VectorXf &sequence,
                          unsigned int offset,
                          const Eigen::VectorXf &x);

        /**
         * @brief Return the value of an action given the current state of PSR
         */
        float valueOfAction(Episode *episode, unsigned int action);

    private:
        std::mutex _mutex;

        unsigned int _history_length;
        unsigned int _test_length;
        unsigned int _rank;
        unsigned int _random_features;

        PSR * _psr;

        unsigned int _last_episode_length;
        Episode *_last_episode;

        std::vector<Eigen::VectorXf> _features;
};

#endif
