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

#ifndef __QLEARNING_H__
#define __QLEARNING_H__

#include "abstracttdlearning.h"

/**
 * @brief Well-known Q-Learning algorithm
 */
class QLearning : public AbstractTDLearning
{
    public:
        /**
         * @param discount_factor Discount factor used when computing cumulative rewards
         * @param learning_rate Rate at which learning occurs
         */
        QLearning(float discount_factor, float eligibility_factor, float learning_rate);

        virtual float tdError(const Episode *episode, unsigned int timestep);

    private:
        // Lists so that memory does not need to be continuously reallocated
        std::vector<float> _last_values;
        std::vector<float> _current_values;
};

#endif
