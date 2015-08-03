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

#include "advantagelearning.h"

#include <algorithm>
#include <iostream>

AdvantageLearning::AdvantageLearning(float discount_factor, float learning_rate, float kappa)
: _discount_factor(discount_factor),
  _learning_rate(learning_rate),
  _inv_kappa(1.0f / kappa)
{
}


void AdvantageLearning::actions(Episode *episode, std::vector<float> &probabilities, float &td_error)
{
    // Update the Q-value of the last action that was taken
    std::vector<float> &current_values = probabilities;             // Reuse temporary vectors

    if (episode->length() >= 2) {
        unsigned int last_t = episode->length() - 2;
        unsigned int last_action = episode->action(last_t);
        float last_reward = episode->reward(last_t);

        episode->values(last_t, _last_values);
        episode->values(last_t + 1, current_values);

        float advantage = _last_values[last_action];
        float last_value = *std::max_element(_last_values.begin(), _last_values.end());
        float current_value = *std::max_element(current_values.begin(), current_values.end());
        td_error =
            last_value +
            (last_reward + _discount_factor * current_value - last_value) * _inv_kappa -
            advantage;

        episode->updateValue(last_t, last_action, advantage + _learning_rate * td_error);
    } else {
        td_error = 0;
    }

    // probabilities (alias current_values) contains the values of the last state
    // in the episode. Truncate it to the number of actions.
    probabilities.resize(episode->numActions());
}
