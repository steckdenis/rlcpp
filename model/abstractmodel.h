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

#ifndef __ABSTRACTMODEL_H__
#define __ABSTRACTMODEL_H__

#include <vector>

class Episode;

/**
 * @brief Model associating episodes (histories) to state values
 */
class AbstractModel
{
    public:
        AbstractModel() {}
        virtual ~AbstractModel() {}

        /**
         * @brief Update the model using the action values of the @p episodes
         *
         * @note Learn might be called in another thread, while values() is used.
         *       Learning should not interfere with prediction. For instance, learning
         *       can occur in a separate internal model, that will be "swapped"
         *       and made fore-ground by a call to swapModels().
         */
        virtual void learn(const std::vector<Episode *> &episodes) = 0;

        /**
         * @brief Return the action values corresponding to the last state of
         *        @p episode.
         */
        virtual void values(Episode *episode, std::vector<float> &rs) = 0;

        /**
         * @brief Tell the model to swap its training and prediction internal models
         *
         * Learning and prediction typically occur in this order :
         *
         * @code
         * learn() | values()  (called in independent threads)
         * a mutex gets locked, synchronizing all the threads
         * swapModels()        (does not need to lock any mutex)
         * the mutex gets unlocked
         * learn() | values()
         * @endcode
         */
        virtual void swapModels() {};

        /**
         * @brief Faster variant of values, used when plotting the model.
         *
         * Some models can do statistics or other things when values() is called,
         * this method is available for fast plotting without any bookkeeping.
         */
        virtual void valuesForPlotting(Episode *episode, std::vector<float> &rs)
        {
            values(episode, rs);
        }

        /**
         * @brief Called at the beginning of an episode, before the first
         *        time step of the episode is predicted.
         *
         * This can be used by time-series models to start a new episode. For
         * instance, a time-step counter can be reset to 0. values() will then
         * use counter..episode->length() values for prediction.
         */
        virtual void nextEpisode() {}
};

#endif
