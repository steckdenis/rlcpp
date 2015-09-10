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

#ifndef __FREEZEDEVICEWORLD_H__
#define __FREEZEDEVICEWORLD_H__

#include "deviceworld.h"

/**
 * @brief World that wraps another one and adds an "freeze" device
 *
 * This world duplicates its observation. The wrapped world produces O(w), and this
 * world produces O(w)||O(f), with || the concatenation operator. Whenever the
 * freeze action is triggered, O(w) is copied to O(f). Otherwise, O(f) is copied
 * from time step to time step. This allows an agent to learn when an observation
 * will be interesting in the future.
 */
class FreezeDeviceWorld : public DeviceWorld
{
    public:
        FreezeDeviceWorld(AbstractWorld *world);

        virtual void reset() override;

    protected:
        /**
         * @brief Freeze the observation when requested
         */
        virtual float performAction(unsigned int action) override;

        /**
         * @brief Add the frozen observation to @p state
         */
        virtual void processState(std::vector<float> &state) override;

    private:
        std::vector<float> _frozen;
};

#endif
