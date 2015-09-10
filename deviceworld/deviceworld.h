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

#ifndef __DEVICEWORLD_H__
#define __DEVICEWORLD_H__

#include "world/postprocessworld.h"

/**
 * @brief World that wraps another one and adds a device to it.
 *
 * A device extends a world with actions, observations and rewards. The agent can
 * learn to use the actions in order to change the internal state of the device,
 * from which one or several observations are built. The device adds those observations
 * to the observations returned by the wrapped world. It can also change the
 * reward returned by the wrapped world.
 */
class DeviceWorld : public PostProcessWorld
{
    public:
        /**
         * @param world World to be wrapped
         * @param device_actions Number of actions added by the device
         */
        DeviceWorld(AbstractWorld *world, unsigned int device_actions);

        virtual void initialState(std::vector<float> &state);

        /**
         * @brief Detect actions of this device and perform them instead of sending
         *        them to the wrapped world.
         */
        virtual void step(unsigned int action,
                          bool &finished,
                          float &reward,
                          std::vector<float> &state);

    protected:
        /**
         * @brief Perform a device action
         * @return Reward produced by the action, that is added to the reward
         *         returned by the world.
         */
        virtual float performAction(unsigned int action) = 0;

    private:
        unsigned int _first_action;

        std::vector<float> _last_state;         /*!< @brief When a device action is performed, the state of the wrapped world does not change and must therefore be preserved */
};

#endif
