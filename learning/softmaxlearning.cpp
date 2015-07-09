#include "softmaxlearning.h"

#include <cmath>
#include <numeric>

SoftmaxLearning::SoftmaxLearning(AbstractLearning *learning, float temperature)
: _learning(learning),
  _temperature(temperature)
{
}

void SoftmaxLearning::actions(Episode *episode, std::vector<float> &probabilities)
{

    // Let the wrapped learning algorithm compute the premilinary values
    _learning->actions(episode, probabilities);

    // Take the exponentials of all those values
    for (float &v : probabilities) {
        v = std::exp(v / _temperature);
    }

    float sum = std::accumulate(probabilities.begin(), probabilities.end(), 0.0f);

    // Compute exp(v / T) / sum(vi / T) for all v
    for (float &v : probabilities) {
        v /= sum;
    }
}
