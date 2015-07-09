#include "episode.h"

#include <algorithm>
#include <numeric>

template<typename T>
void extend(std::vector<T> &dest, const std::vector<T> &src)
{
    // Make room in dest for src
    dest.resize(dest.size() + src.size());

    // Append src to dest
    std::copy(src.begin(), src.end(), dest.end() - src.size());
}

static std::vector<float> extract(const std::vector<float> &vector,
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

Episode::Episode(unsigned int value_size)
: _state_size(0),
  _value_size(value_size)
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

unsigned int Episode::stateSize() const
{
    return _state_size;
}

unsigned int Episode::valueSize() const
{
    return _value_size;
}

unsigned int Episode::length() const
{
    return _values.size() / _value_size;
}

void Episode::state(unsigned int t, std::vector<float> &rs) const
{
    extract(_states, _state_size, t, rs);
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
