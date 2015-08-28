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

#ifndef __POLARGRIDWORLD_H__
#define __POLARGRIDWORLD_H__

#include "gridworld.h"

/**
 * @brief Same as GridWorld, except that the agent can only sense its orientation
 *        and the distance between it and the wall in front of it.
 */
class PolarGridWorld : public GridWorld
{
    public:
        enum class Action {
            Forward,
            Backward,
            TurnLeft,
            TurnRight
        };

        PolarGridWorld(unsigned int width,
                       unsigned int height,
                       Point initial,
                       Point obstacle,
                       Point goal,
                       bool stochastic);

        virtual void reset();
        virtual void step(unsigned int action,
                          bool &finished,
                          float &reward,
                          std::vector<float> &state);

    private:
        /**
         * @brief Encode the current position into a 2-variables (rho, sigma) state
         */
        virtual void encodeState(const Point &point, std::vector<float> &state);

    private:
        unsigned int _direction;
};

#endif
