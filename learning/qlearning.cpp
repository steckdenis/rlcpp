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

QLearning::QLearning(float discount_factor, float learning_rate)
: _discount_factor(discount_factor),
  _learning_rate(learning_rate)
{
}


void QLearning::actions(Episode *episode, std::vector<float> &probabilities, float &td_error)
{
    // Update the Q-value of the last action that was taken
    std::vector<float> &current_values = probabilities;             // Reuse temporary vectors

    if (episode->length() >= 2) {
        unsigned int last_t = episode->length() - 2;
        unsigned int last_action = episode->action(last_t);
        float last_reward = episode->reward(last_t);

        episode->values(last_t, _last_values);
        episode->values(last_t + 1, current_values);

        float Q = _last_values[last_action];
        td_error =
            last_reward +
            _discount_factor * *std::max_element(current_values.begin(), current_values.end())
            - Q;

        episode->updateValue(last_t, last_action, Q + _learning_rate * td_error);
    }

    // probabilities (alias current_values) contains the values of the last state
    // in the episode.
}
