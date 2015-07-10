#include "oneofnworld.h"

#include <cmath>
#include <iostream>

OneOfNWorld::OneOfNWorld(AbstractWorld *world,
                         const std::vector<int> &minimums,
                         const std::vector<int> &maximums)
: AbstractWorld(world->numActions()),
  _world(world),
  _minimums(minimums),
  _maximums(maximums)
{
    // Compute the state size
    _postprocessed_state_size = 0;

    for (std::size_t i=0; i<minimums.size(); ++i) {
        _postprocessed_state_size += 1 + maximums[i] - minimums[i];
    }
}

OneOfNWorld::~OneOfNWorld()
{
    delete _world;
}

void OneOfNWorld::initialState(std::vector <float> &state)
{
    _world->initialState(state);
    processState(state);
}

void OneOfNWorld::reset()
{
    _world->reset();
}

void OneOfNWorld::step(unsigned int action,
                       bool &finished,
                       float &reward,
                       std::vector<float> &state)
{
    // Let the wrapped world compute the step, then post-process it
    _world->step(action, finished, reward, state);
    processState(state);
}

void OneOfNWorld::processState(std::vector<float> &state)
{
    // Resize state to its new size, that will be bigger than the original size
    // because one-hot expands the state space
    int original_size = state.size();

    state.resize(_postprocessed_state_size);

    // Adjust the state
    int index = original_size - 1;
    int offset = _postprocessed_state_size;

    for (; index >= 0; --index) {
        float value = state[index] - _minimums[index];
        int encoded_length = 1 + _maximums[index] - _minimums[index];

        // Compute the starting position in the encoded state where the value
        // has to be placed
        offset -= encoded_length;

        // Put a one at the position that corresponds the better to the state
        for (int i=0, f=0.0f; i < encoded_length; i += 1, f += 1.0f) {
            state[offset + i] = std::abs(f - value) < 0.5 ? 1.0f : 0.0f;
        }
    }
}
