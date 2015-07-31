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

#include "oneofnworld.h"

#include <cmath>
#include <assert.h>

OneOfNWorld::OneOfNWorld(AbstractWorld *world,
                         const std::vector<int> &minimums,
                         const std::vector<int> &maximums)
: PostProcessWorld(world),
  _minimums(minimums),
  _maximums(maximums)
{
    // Compute the state size
    _postprocessed_state_size = 0;

    for (std::size_t i=0; i<minimums.size(); ++i) {
        _postprocessed_state_size += 1 + maximums[i] - minimums[i];
    }
}

void OneOfNWorld::processState(std::vector<float> &state)
{
    assert(state.size() == _minimums.size());

    // Resize state to its new size, that will be bigger than the original size
    // because one-hot expands the state space
    state.resize(_postprocessed_state_size);

    // Adjust the state
    int index = _minimums.size() - 1;
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
