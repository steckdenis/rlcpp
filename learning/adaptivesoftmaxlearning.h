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
#ifndef __ADAPTIVESOFTMAXLEARNING_H__
#define __ADAPTIVESOFTMAXLEARNING_H__

#include "softmaxlearning.h"

class Network;

/**
 * @brief Softmax that adjusts its temperature depending on the TD-error usually
 *        received at the current state.
 */
class AdaptiveSoftmaxLearning : public SoftmaxLearning
{
    public:
        /**
         * @param learning Learning algorithm that is wrapped by this Softmax
         * @param discount_factor Discount factor applied to future TD errors
         */
        AdaptiveSoftmaxLearning(AbstractLearning *learning,
                                float discount_factor);

        /**
         * @brief Let a subclass adjust the temperature of Softmax before an action
         *        is selected.
         */
        virtual float adjustTemperature(Episode *episode, float td_error);

        /**
         * @brief Require one more value, so that each state-action has information
         *        about its expected td-error.
         */
        virtual unsigned int valueSize(unsigned int num_actions) const;

    private:
        std::vector<float> _values;
        float _discount_factor;
};

#endif
