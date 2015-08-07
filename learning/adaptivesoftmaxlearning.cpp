#include "adaptivesoftmaxlearning.h"

#include <cmath>

AdaptiveSoftmaxLearning::AdaptiveSoftmaxLearning(AbstractLearning *learning,
                                                 float discount_factor)
: SoftmaxLearning(learning, 1.0f),
  _discount_factor(discount_factor)
{
}

unsigned int AdaptiveSoftmaxLearning::valueSize(unsigned int num_actions) const
{
    return num_actions + 1;
}

float AdaptiveSoftmaxLearning::adjustTemperature(Episode *episode, float td_error)
{
    /* Compute the new temperature : y(t) = |td_error| + beta*y(t+1)
     *
     * Formula given in "Reinforcement Learning with Long Short-Term Memory",
     * Bram Bakker, 2001
     *
     * Because t+1 is not yet known, another formula is used: the model
     * is used to predict y(t), and y(t-1) = |td_error| + beta*y(t) is used
     * to train the model for the previous observation
     */
    unsigned int current_t = episode->length() - 1;
    unsigned int temp_index = episode->valueSize() - 1;

    episode->values(current_t, _values);

    float current_temperature = _values[temp_index];
    float prev_temperature = std::abs(td_error) + _discount_factor * current_temperature;

    // Update the prediction for the last state
    if (episode->length() > 1) {
        episode->updateValue(current_t - 1, temp_index, prev_temperature);
    }

    // Use the new temperature
    return std::max(0.2f, current_temperature);
}
