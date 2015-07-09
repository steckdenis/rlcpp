#include "advantagelearning.h"

#include <algorithm>
#include <iostream>

AdvantageLearning::AdvantageLearning(float discount_factor, float learning_rate, float kappa)
: _discount_factor(discount_factor),
  _learning_rate(learning_rate),
  _inv_kappa(1.0f / kappa)
{
}


void AdvantageLearning::actions(Episode *episode, std::vector<float> &probabilities)
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
        float error =
            last_value +
            (last_reward + _discount_factor * current_value - last_value) * _inv_kappa -
            advantage;

        episode->updateValue(last_t, last_action, advantage + _learning_rate * error);
    }

    // probabilities (alias current_values) contains the values of the last state
    // in the episode.
}
