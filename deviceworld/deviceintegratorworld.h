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

#ifndef __DEVICEINTEGRATORWORLD_H__
#define __DEVICEINTEGRATORWORLD_H__

#include "deviceworld.h"

/**
 * @brief World that wraps another one and adds an "integrator" device
 *
 * This world adds two possible actions and one observation. The actions are
 * "increment" and "decrement", and the observation is the value of the integrator.
 * Its initial value is zero.
 *
 * The agent can learn to use this world to count events or set flags, which allows
 * an normally MDP-only to solve some simple POMDPs.
 */
class DeviceIntegratorWorld : public DeviceWorld
{
    public:
        DeviceIntegratorWorld(AbstractWorld *world, float min, float max);

        virtual void reset() override;

    protected:
        /**
         * @brief Increment or decrement the integrator
         */
        virtual float performAction(unsigned int action) override;

        /**
         * @brief Add the observation to the state returned by the wrapped world
         */
        virtual void processState(std::vector<float> &state) override;

    private:
        float _min;
        float _max;
        float _value;
};

#endif
