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

#include "deviceworld.h"

DeviceWorld::DeviceWorld(AbstractWorld *world,
                         unsigned int device_actions)
: PostProcessWorld(world, world->numActions() + device_actions),
  _first_action(world->numActions())
{
}

void DeviceWorld::initialState(std::vector<float> &state)
{
    // Store the initial unprocessed state in _last_state so that device actions
    // can immediately be issued, then postprocess that state.
    _world->initialState(state);
    _last_state = state;

    processState(state);
}

void DeviceWorld::step(unsigned int action,
                       bool &finished,
                       float &reward,
                       std::vector<float> &state)
{
    if (action < _first_action) {
        // Normal world action, let the world execute it
        _world->step(action, finished, reward, state);

        // Save the world state so that it can be used again if a device action
        // is performed
        _last_state = state;
    } else {
        // Perform the device action
        reward = performAction(action - _first_action);
        state = _last_state;
        finished = false;
    }

    // Post-process the state. This allows the device to add its observations
    // to the state.
    processState(state);
}
