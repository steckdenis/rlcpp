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

#include "polargridworld.h"

PolarGridWorld::PolarGridWorld(unsigned int width,
                               unsigned int height,
                               Point initial,
                               Point obstacle,
                               Point goal,
                               bool stochastic)
: GridWorld(width, height, initial, obstacle, goal, stochastic),
  _direction((unsigned int)GridWorld::Action::Right)
{
}

void PolarGridWorld::step(unsigned int action,
                          bool &finished,
                          float &reward,
                          std::vector<float> &state)
{
    // Map a PolarGridWorld action to a GridWorld action
    unsigned int mapped_action;

    switch ((Action)action) {
        case Action::Forward:
            mapped_action = _direction;
            break;

        case Action::Backward:
            mapped_action = (_direction + 2) % 4;
            break;

        case Action::TurnLeft:
            _direction = (_direction - 1) % 4;
            mapped_action = 100;
            break;

        case Action::TurnRight:
            _direction = (_direction + 1) % 4;
            mapped_action = 100;
            break;
    }

    if (mapped_action != 100) {
        // Forward or backward: let GridWorld perform the action
        GridWorld::step(mapped_action, finished, reward, state);
    } else {
        // Provide the reward and the other signals
        finished = false;
        reward = -1.0f;
        encodeState(_current_pos, state);
    }
}

void PolarGridWorld::encodeState(const Point &point, std::vector<float> &state)
{
    // Measure the distance between the agent and the closest wall
    unsigned int distance;

    switch ((GridWorld::Action)_direction) {
        case GridWorld::Action::Up:
            distance = point.y;
            break;

        case GridWorld::Action::Right:
            distance = _width - point.x - 1;
            break;

        case GridWorld::Action::Down:
            distance = _height - point.y - 1;
            break;

        case GridWorld::Action::Left:
            distance = point.x;
            break;
    }

    state.resize(2);
    state[0] = float(_direction);
    state[1] = float(distance);
}
