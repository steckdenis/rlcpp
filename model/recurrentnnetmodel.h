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

#ifndef __RECURRENTNNETMODEL_H__
#define __RECURRENTNNETMODEL_H__

#include "abstractmodel.h"

#include <nnetcpp/network.h>

/**
 * @brief Base class for recurrent neural networks.
 *
 * Recurrent neural networks are trained on input sequences, not just simple inputs.
 * This model takes care of the proper initialization and reinitialization of
 * the neural network between sequences (during training and prediction).
 */
class RecurrentNnetModel : public AbstractModel
{
    public:
        RecurrentNnetModel();
        virtual ~RecurrentNnetModel();

        virtual void values(Episode *episode, std::vector<float> &rs);
        virtual void learn(const std::vector<Episode *> &episodes);

        /**
         * @brief Create a neural network having a number of input and output
         *        neurons adapted to @p first_episode.
         */
        virtual Network *createNetwork(Episode *first_episode) const = 0;

    private:
        Network *_network;
        unsigned int _last_episode_length;
};

#endif