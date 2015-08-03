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

#ifndef __TMAZEWORLD_H__
#define __TMAZEWORLD_H__

#include "abstractworld.h"

/**
 * @brief Corridor leading to a T junction, where the agent has to choose the
 *        correct direction.
 */
class TMazeWorld : public AbstractWorld
{
    public:
        enum class Action {
            Up = 1,
            Down,
            Left,
            Right
        };

        /**
         * @param length Length of the corridor, T junction included
         * @param info_time Number of time steps during which the agent can observe
         *                  which way it has to go at the T junction.
         */
        TMazeWorld(unsigned int length,
                   unsigned int info_time);

        virtual void initialState(std::vector<float> &state);
        virtual void reset();
        virtual void step(unsigned int action,
                          bool &finished,
                          float &reward,
                          std::vector<float> &state);

    private:
        /**
         * @brief Encode the current position into a 2-variables (x, hint) state
         */
        virtual void encodeState(unsigned int pos, std::vector<float> &state);

    protected:
        unsigned int _length;
        unsigned int _info_time;

        unsigned int _timesteps;
        unsigned int _pos;
        Action _target;
};

#endif
