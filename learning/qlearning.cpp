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

#include "qlearning.h"

#include <algorithm>

QLearning::QLearning(float discount_factor, float eligibility_factor, float learning_rate)
: AbstractTDLearning(discount_factor, eligibility_factor, learning_rate)
{
}

float QLearning::tdError(const Episode *episode, unsigned int timestep)
{
    unsigned int last_action = episode->action(timestep - 1);
    float last_reward = episode->reward(timestep - 1);

    episode->values(timestep - 1, _last_values);
    episode->values(timestep, _current_values);

    float Q = _last_values[last_action];

    return
        last_reward +
        _discount_factor * *std::max_element(_current_values.begin(), _current_values.end())
        - Q;
}
