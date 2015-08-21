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

#include "episode.h"

#include <algorithm>
#include <numeric>

template<typename T>
void extend(std::vector<T> &dest, const std::vector<T> &src)
{
    // Make room in dest for src
    std::size_t offset = dest.size();

    dest.resize(offset + src.size());

    // Append src to dest
    std::copy(src.begin(), src.end(), dest.begin() + offset);
}

static void extract(const std::vector<float> &vector,
                    unsigned int size,
                    unsigned int t,
                    std::vector<float> &rs)
{
    rs.resize(size);

    // Compute the positions between which the desired state values are stored
    unsigned int from = t * size;
    unsigned int to = from + size;

    // Copy the desired values into the output
    std::copy(vector.begin() + from, vector.begin() + to, rs.begin());
}

Episode::Episode(unsigned int value_size, unsigned int num_actions, Encoder encoder)
: _encoder(encoder),
  _state_size(0),
  _value_size(value_size),
  _num_actions(num_actions),
  _aborted(false)
{
}

void Episode::addState(const std::vector<float> &state)
{
    // Update the state size, used to split the values stored in _states by state
    _state_size = state.size();

    extend(_states, state);
}

void Episode::addValues(const std::vector<float> &values)
{
    extend(_values, values);
}

void Episode::addReward(float reward)
{
    _rewards.push_back(reward);
}

void Episode::addAction(int action)
{
    _actions.push_back(action);
}

void Episode::setAborted(bool aborted)
{
    _aborted = aborted;
}

unsigned int Episode::stateSize() const
{
    return _state_size;
}

unsigned int Episode::encodedStateSize() const
{
    std::vector<float> state;

    encodedState(0, state);

    return state.size();
}

unsigned int Episode::valueSize() const
{
    return _value_size;
}

unsigned int Episode::numActions() const
{
    return _num_actions;
}

unsigned int Episode::length() const
{
    return _states.size() / _state_size;
}

bool Episode::wasAborted() const
{
    return _aborted;
}

void Episode::state(unsigned int t, std::vector<float> &rs) const
{
    extract(_states, _state_size, t, rs);
}

void Episode::encodedState(unsigned int t, std::vector<float> &rs) const
{
    state(t, rs);

    // Encode the state using the encoder
    if (_encoder) {
        _encoder(rs);
    }
}

void Episode::values(unsigned int t, std::vector<float> &rs) const
{
    extract(_values, _value_size, t, rs);
}

void Episode::updateValue(unsigned int t, unsigned int action, float value)
{
    _values[t * _value_size + action] = value;
}

float Episode::reward(unsigned int t) const
{
    return _rewards[t];
}

float Episode::cumulativeReward() const
{
    return std::accumulate(_rewards.begin(), _rewards.end(), 0.0f);
}

float Episode::action(unsigned int t) const
{
    return _actions[t];
}
