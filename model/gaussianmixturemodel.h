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

#ifndef __GAUSSIANMIXTUREMODEL_H__
#define __GAUSSIANMIXTUREMODEL_H__

#include "abstractmodel.h"

#include <Eigen/Dense>
#include <vector>
#include <random>
#include <mutex>

class GaussianMixture;

/**
 * @brief Model that can handle continuous states and uses a GaussianMixture as
 *        backing store.
 */
class GaussianMixtureModel : public AbstractModel
{
    public:
        /**
         * @param var_initial Initial variance of the gaussian clusters
         * @param novelty Minimum probability that x appears in a cluster for the
         *                clusters to be updated instead of a new cluster being created
         * @param noise Variance of the noise added to the inputs. If this model
         *              is used in a discrete world, some noise has to be added
         *              in order to prevent the gaussians from degenerating to
         *              diracs (and vanishing due to rounding errors)
         * @param mask_actions Only learn values associated with the action that
         *                     has been taken, instead of learning all the values.
         */
        GaussianMixtureModel(float var_initial, float novelty, float noise, bool mask_actions);
        virtual ~GaussianMixtureModel();

        virtual void values(Episode *episode, std::vector<float> &rs);
        virtual void learn(const std::vector<Episode *> &episodes);
        virtual void swapModels();

    private:
        void vectorToVectorXf(const std::vector<float> &stl, Eigen::VectorXf &eigen);

    private:
        float _var_initial;
        float _novelty;
        bool _mask_actions;

        std::normal_distribution<float> _noise_distribution;
        std::default_random_engine _random_engine;

        std::vector<GaussianMixture *> _models;     /*!< @brief One model per action */
        std::vector<GaussianMixture *> _learn_models;

        std::mutex _models_mutex;
};

#endif
