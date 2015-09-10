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

#include "freezedeviceworld.h"

FreezeDeviceWorld::FreezeDeviceWorld(AbstractWorld *world)
: DeviceWorld(world, 1)
{
}

void FreezeDeviceWorld::reset()
{
    DeviceWorld::reset();

    // The original frozen observation is the initial state of the wrapped world
    _world->initialState(_frozen);
}

float FreezeDeviceWorld::performAction(unsigned int action)
{
    (void) action;      // Only one action supported

    // Copy the current observation to the frozen observation
    _frozen = _last_state;

    // Give a small penalty when the device is used
    return -1.0f;
}

void FreezeDeviceWorld::processState(std::vector<float> &state)
{
    // Add the value to the state
    state.insert(state.end(), _frozen.begin(), _frozen.end());
}
