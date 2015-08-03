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

#include "tmazeworld.h"

#include <cstdlib>

TMazeWorld::TMazeWorld(unsigned int length,
                       unsigned int info_time)
: AbstractWorld(4),
  _length(length),
  _info_time(info_time),
  _timesteps(0),
  _target(Action::Up)
{
}

void TMazeWorld::initialState(std::vector<float> &state)
{
    encodeState(0, state);
}

void TMazeWorld::reset()
{
    _timesteps = 0;
    _pos = 0;

    // Choose a target
    if (std::rand() & 1) {
        _target = Action::Up;
    } else {
        _target = Action::Down;
    }
}

void TMazeWorld::step(unsigned int action,
                      bool &finished,
                      float &reward,
                      std::vector<float> &state)
{
    unsigned int pos_x = _pos;
    unsigned int pos_y = 0;

    // Count the number of timesteps
    ++_timesteps;

    // Perform the action
    switch ((Action)(action + 1)) { // Action starts at 1, while action is in the range 0..3
        case Action::Up:
            pos_y += 1;
            break;

        case Action::Down:
            pos_y -= 1;
            break;

        case Action::Left:
            pos_x -= 1;
            break;

        case Action::Right:
            pos_x += 1;
            break;
    }

    // Check the validity of the position
    if (pos_x == _length - 1 && pos_y == -1) {
        // Down part of the junction
        reward = (_target == Action::Down ? 10.0f : 0.0f);
        finished = true;

        _pos = pos_x;
    } else if (pos_x == _length - 1 && pos_y == 1) {
        // Up part of the junction
        reward = (_target == Action::Up ? 10.0f : 0.0f);
        finished = true;

        _pos = pos_x;
    } else if (pos_y == -1 || pos_y == 1 || pos_x < 0 || pos_x >= _length) {
        // Overflow
        reward = -2.0f;
        finished = false;
    } else {
        // Simple move
        reward = 0.0f;
        finished = false;

        _pos = pos_x;
    }

    encodeState(_pos, state);
}

void TMazeWorld::encodeState(unsigned int pos, std::vector<float> &state)
{
    state.resize(2);

    state[0] = float(_timesteps <= _info_time ? (int)_target : 0);
    state[1] = float(pos);
}
