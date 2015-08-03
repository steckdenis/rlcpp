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

#ifndef __ABSTRACTLEARNING_H__
#define __ABSTRACTLEARNING_H__

#include <model/episode.h>

/**
 * @brief Learning algorithm
 *
 * The learning algorithm returns a probability distribution over the possible
 * actions given an episode. It can also update the action values of the episode.
 */
class AbstractLearning
{
    public:
        AbstractLearning() {}
        virtual ~AbstractLearning() {}

        /**
         * @brief Populate @p probabilities with the probability that each action
         *        is taken. @p episode can be updated if the learning algorithm
         *        has to learn new state-action values.
         */
        virtual void actions(Episode *episode, std::vector<float> &probabilities, float &td_error) = 0;

        /**
         * @brief Return the number of value elements to be stored in an episode
         *        given the number of possible actions.
         *
         * This allows learning algorithms to store more data in the episode, if
         * statistics about rewards or td-errors are needed, for instance.
         */
        virtual unsigned int valueSize(unsigned int num_actions) const
        {
            return num_actions;
        }
};

#endif
