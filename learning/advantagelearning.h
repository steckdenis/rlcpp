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

#ifndef __ADVANTAGELEARNING_H__
#define __ADVANTAGELEARNING_H__

#include "abstracttdlearning.h"

/**
 * @brief Advantage learning as detailed in "Reinforcement Learning using LSTM"
 */
class AdvantageLearning : public AbstractTDLearning
{
    public:
        /**
         * @param kappa The smaller this factor is, the strongest the bias for
         *              better actions is.
         */
        AdvantageLearning(float discount_factor, float eligibility_factor, float learning_rate, float kappa);

        virtual float tdError(const Episode *episode, unsigned int timestep);

    private:
        float _inv_kappa;

        // Lists so that memory does not need to be continuously reallocated
        std::vector<float> _last_values;
        std::vector<float> _current_values;
};

#endif
