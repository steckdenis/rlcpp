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

#ifndef __SCALEWORLD_H__
#define __SCALEWORLD_H__

#include "postprocessworld.h"

/**
 * @brief World that wraps another ones and encodes scales its observations.
 *
 * This can be used for normalization (for neural networks), or for multiplying
 * some state variables by 0.0, hence making the world partially observable.
 */
class ScaleWorld : public PostProcessWorld
{
    public:
        /**
         * @param world World to be wrapped
         * @param weights Values by which the state variables are multiplied
         */
        ScaleWorld(AbstractWorld *world,
                    const std::vector<float> &weights);
        virtual ~ScaleWorld();

    protected:
        /**
         * @brief Scale a state according to the weight vector
         */
        virtual void processState(std::vector<float> &state) override;

    private:
        AbstractWorld *_world;
        std::vector<float> _weights;
};

#endif