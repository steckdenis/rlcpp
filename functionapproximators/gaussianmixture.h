/*
 * Copyright (c) 2015 Denis Steckelmacher
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

#ifndef __GAUSSIANMIXTURE_H__
#define __GAUSSIANMIXTURE_H__

#include <Eigen/Dense>
#include <vector>

/**
 * @brief Function approximator based on an incremental gaussian mixture model
 */
class GaussianMixture
{
    public:
        /**
         * @param var_initial Initial variance of a gaussian cluster
         * @param novelty Minimum probability that a point is in a cluster in
         *                order for it to be added to the cluster.
         */
        GaussianMixture(float var_initial, float novelty);

        /**
         * @brief Set the value of a point
         */
        void setValue(const Eigen::VectorXf &input, float value);

        /**
         * @brief Get the value of a point
         */
        float value(const Eigen::VectorXf &input) const;

        /**
         * @brief Number of clusters in the model (for statistics)
         */
        unsigned int numberOfClusters() const;

    private:
        /**
         * @brief Compute p(input|cluster)
         */
        float probabilityOfInput(unsigned int cluster, const Eigen::VectorXf &input) const;

        /**
         * @brief Probabilities of all the inputs
         */
        void probabilitiesOfInputs(float *out, const Eigen::VectorXf &input, float value) const;

        /**
         * @brief Compute p(cluster|input) for all the clusters
         *
         * Computing the probabilities of all the clusters takes only marginally
         * more time than computing the probability of only one cluster, hence
         * this function that returns all the probabilities.
         */
        void probabilitiesOfClusters(float *out, const Eigen::VectorXf &input, float value) const;
        void probabilitiesOfClusters(float *out, float *input_probabilities) const;

    private:
        float _var_initial;
        float _novelty;

        float _inv_2pi_d;                                                       /*!< @brief 1/(2pi ^ (D/2)), used to normalize the gaussians */
        std::vector<float> _gaussian_normalizations;                            /*!< @brief Each cluster has a normalization factor equal to _inv_2pi_d / sqrt(|covariance(i)|) */
        std::vector<float> _probabilities;                                      /*!< @brief Probabilities of all the clusters */
        std::vector<float> _sprobabilities;                                     /*!< @brief sp(i) values of the clusters, used to compute p(i) = sp(i) / sum(sp(*)) */
        std::vector<float> _weights;                                            /*!< @brief Weights of all the clusters */
        std::vector<Eigen::MatrixXf> _covariances;                              /*!< @brief Covariance matrices for all the clusters */
        std::vector<Eigen::MatrixXf> _inv_covariances;                          /*!< @brief Inverse of covariance matrices for all the clusters, used to speed up value() */
        std::vector<Eigen::VectorXf> _means;                                    /*!< @brief Centroids of all the gaussians */
};

#endif