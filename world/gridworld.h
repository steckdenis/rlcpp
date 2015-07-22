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

#ifndef __GRIDWORLD_H__
#define __GRIDWORLD_H__

#include "abstractworld.h"

/**
 * @brief Gridworld in which there is an obstacle and a goal
 */
class GridWorld : public AbstractWorld
{
    public:
        struct Point {
            int x;
            int y;
        };

        enum class Action {
            Up,
            Right,
            Down,
            Left
        };

        GridWorld(unsigned int width,
                  unsigned int height,
                  Point initial,
                  Point obstacle,
                  Point goal,
                  bool stochastic);

        virtual void initialState(std::vector<float> &state);
        virtual void reset();
        virtual void step(unsigned int action,
                          bool &finished,
                          float &reward,
                          std::vector<float> &state);

    private:
        /**
         * @brief Encode the current position into a 2-variables (x, y) state
         */
        void encodeState(const Point &point, std::vector<float> &state);

    private:
        unsigned int _width;
        unsigned int _height;
        Point _initial;
        Point _obstacle;
        Point _goal;
        Point _current_pos;

        bool _stochastic;
};

#endif