#include "egreedylearning.h"

#include <algorithm>
#include <numeric>

EGreedyLearning::EGreedyLearning(AbstractLearning *learning, float epsilon)
: _learning(learning),
  _epsilon(epsilon)
{
}

EGreedyLearning::~EGreedyLearning()
{
    delete _learning;
}

void EGreedyLearning::actions(Episode *episode, std::vector<float> &probabilities)
{

    // Let the wrapped learning algorithm compute the premilinary values
    _learning->actions(episode, probabilities);

    // Iterator to the best value
    auto it = std::max_element(probabilities.begin(), probabilities.end());

    // All the elements have a probability epsilon/(N - 1) of being taken
    float proba = _epsilon / float(probabilities.size() - 1);

    for (float &v : probabilities) {
        v = proba;
    }

    // The best element has a probability of 1-epsilon of being taken
    *it = 1.0f - _epsilon;
}
