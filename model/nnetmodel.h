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

#ifndef __NNETMODEL_H__
#define __NNETMODEL_H__

#include "abstractmodel.h"

#include <nnetcpp/network.h>
#include <mutex>

/**
 * @brief Base class for non-recurrent neural networks.
 *
 * In this network, every state-action-value tuple is considered independent from
 * any history. This hypothesis is valid when a neural network has no recurrence,
 * but recurrent networks require histories to be kept in order (use
 * RecurrentNnetModel for that).
 */
class NnetModel : public AbstractModel
{
    public:
        NnetModel();
        virtual ~NnetModel();

        virtual void values(Episode *episode, std::vector<float> &rs);
        virtual void learn(const std::vector<Episode *> &episodes);
        virtual void swapModels();

        /**
         * @brief Create a neural network having a number of input and output
         *        neurons adapted to @p first_episode
         */
        virtual Network *createNetwork(Episode *first_episode) const = 0;

        static void vectorToVector(const std::vector<float> &stl, Vector &eigen);
        static void vectorToCol(const std::vector<float> &stl, Matrix &matrix, int col);
        static void getNodeOutput(AbstractNode *node, std::vector<float> &rs);

    private:
        Network *_network;
        Network *_learn_network;

        std::mutex _mutex;
};

#endif
