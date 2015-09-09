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

#ifndef __FUSIONARTMODEL_H__
#define __FUSIONARTMODEL_H__

#include "abstractmodel.h"
#include "functionapproximators/fusionart.h"

#include <mutex>

/**
 * @brief Model built on FusionART.
 */
class FusionARTModel : public AbstractModel
{
    public:
        /**
         * @brief Constructor
         *
         * @param mask_actions True if actions that were not taken at a given time
         *                     step should be ignored (not learned).
         */
        FusionARTModel(bool mask_actions);
        virtual ~FusionARTModel();

        virtual void values(Episode *episode, std::vector<float> &rs);
        virtual void learn(const std::vector<Episode *> &episodes);
        virtual void swapModels();

    private:
        void vectorToArrayXf(const std::vector<float> &stl, Eigen::ArrayXf &eigen);

    private:
        struct Model {
            FusionART model;
            FusionART::Port state;
            FusionART::Port action;
            FusionART::Port value;
        };

        std::mutex _mutex;

        bool _mask_actions;
        std::vector<float> _state;

        Model *_prediction_model;
        Model *_learning_model;
};

#endif
