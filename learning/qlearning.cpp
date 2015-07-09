#include "qlearning.h"

#include <algorithm>

QLearning::QLearning(float discount_factor, float learning_rate)
: _discount_factor(discount_factor),
  _learning_rate(learning_rate)
{
}


void QLearning::actions(Episode *episode, std::vector<float> &probabilities)
{
    // Update the Q-value of the last action that was taken
    std::vector<float> &current_values = probabilities;             // Reuse temporary vectors

    if (episode->length() >= 2) {
        unsigned int last_t = episode->length() - 1;
        unsigned int last_action = episode->action(last_t);
        float last_reward = episode->reward(last_t);

        episode->values(last_t - 1, _last_values);
        episode->values(last_t, current_values);

        float Q = _last_values[last_action];
        float error =
            last_reward +
            _discount_factor * *std::max_element(current_values.begin(), current_values.end())
            - Q;

        episode->updateValue(last_t - 1, last_action, Q + _learning_rate * error);
    }

    // probabilities (alias current_values) contains the values of the last state
    // in the episode.
}
