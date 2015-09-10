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

#include "integratordeviceworld.h"

IntegratorDeviceWorld::IntegratorDeviceWorld(AbstractWorld *world, float min, float max)
: DeviceWorld(world, 2),
  _min(min),
  _max(max),
  _value(0.0f)
{
}

void IntegratorDeviceWorld::reset()
{
    DeviceWorld::reset();

    _value = 0.0f;
}

float IntegratorDeviceWorld::performAction(unsigned int action)
{
    float old_value = _value;

    switch (action) {
        case 0:
            _value = std::min(_max, _value + 1.0f);
            break;

        case 1:
            _value = std::max(_min, _value - 1.0f);
            break;
    }

    return (old_value == _value ? -2.0f : -1.0f);        // Give a penalty when the agent does something useless
}

void IntegratorDeviceWorld::processState(std::vector<float> &state)
{
    // Add the value to the state
    state.push_back(_value);
}
