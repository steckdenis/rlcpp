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

#include "abstracttdlearning.h"

AbstractTDLearning::AbstractTDLearning(float discount_factor, float eligibility_factor, float learning_rate)
: _discount_factor(discount_factor),
  _eligibility_factor(eligibility_factor),
  _learning_rate(learning_rate)
{
}


void AbstractTDLearning::actions(Episode *episode, std::vector<float> &probabilities, float &td_error)
{
    // Update the action values using the TD errors
    float eligibility = 1.0f;

    if (episode->length() >= 2) {
        for (unsigned int current_t = episode->length() - 1; current_t > 0; --current_t) {
            unsigned int last_action = episode->action(current_t - 1);

            // Compute the TD-error at this time-step
            float error = tdError(episode, current_t);

            // Update the action value using this error and an eligibility trace
            episode->addValue(current_t - 1, last_action, _learning_rate * eligibility * error);

            // Update the eligbility trace and set td_error to the TD-error of the
            // last action (td_error is used to tune exploration/exploitation in
            // AdaptiveSoftmax).
            eligibility *= _eligibility_factor;

            if (current_t == episode->length() - 1) {
                td_error = error;
            } else if (eligibility < 1e-2) {
                break;
            }
        }
    } else {
        td_error = 0.0f;
    }

    // probabilities (alias current_values) contains the values of the last state
    // in the episode. Truncate it to the number of actions.
    episode->values(episode->length() - 1, probabilities);
    probabilities.resize(episode->numActions());
}
