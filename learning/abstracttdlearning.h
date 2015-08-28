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

#ifndef __ABSTRACTTDLEARNING_H__
#define __ABSTRACTTDLEARNING_H__

#include "abstractlearning.h"

/**
 * @brief Base class for temporal-difference learning
 */
class AbstractTDLearning : public AbstractLearning
{
    public:
        /**
         * @param discount_factor Discount factor used when computing cumulative rewards
         * @param eligbility_factor Discount factor used when computing eligibility traces
         * @param learning_rate Rate at which learning occurs
         */
        AbstractTDLearning(float discount_factor, float eligibility_factor, float learning_rate);

        virtual void actions(Episode *episode, std::vector<float> &probabilities, float &td_error);

        /**
         * @brief Compute the temporal-difference error between @p timestep - 1
         *        and @p timestep.
         *
         * @param episode Episode that contains the experiences of the agent
         */
        virtual float tdError(const Episode *episode, unsigned int timestep) = 0;

    protected:
        float _discount_factor;
        float _eligibility_factor;
        float _learning_rate;
};

#endif
