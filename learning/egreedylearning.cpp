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

void EGreedyLearning::actions(Episode *episode, std::vector<float> &probabilities, float &td_error)
{

    // Let the wrapped learning algorithm compute the premilinary values
    _learning->actions(episode, probabilities, td_error);

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
