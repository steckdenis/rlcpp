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

#include "gaussianmixture.h"

#include <algorithm>
#include <cmath>

#include <alloca.h>

#define NEW_VEC (float *)alloca(numberOfClusters() * sizeof(float))

GaussianMixture::GaussianMixture(float var_initial, float novelty)
: _var_initial(var_initial),
  _novelty(novelty)
{
}

unsigned int GaussianMixture::numberOfClusters() const
{
    return _weights.size();
}


float GaussianMixture::value(const Eigen::VectorXf &input) const
{
    float *probabilities = NEW_VEC;
    float rs = 0.0f;

    probabilitiesOfClusters(probabilities, input, NAN);

    for (std::size_t cluster=0; cluster<numberOfClusters(); ++cluster) {
        rs += _weights[cluster] * probabilities[cluster];
    }

    return rs;
}

void GaussianMixture::setValue(const Eigen::VectorXf &input, float value)
{
    int D = input.rows();
    float sum_sp = numberOfClusters() == 0 ? 1.0f : std::accumulate(_sprobabilities.begin(), _sprobabilities.end(), 0.0f);

    // If the probability of one cluster is above its novelty, an existing
    // cluster can be reused.
    bool create_new_cluster = true;

    for (std::size_t cluster=0; cluster<numberOfClusters(); ++cluster) {
        if (probabilityOfInput(cluster, input) > _gaussian_normalizations[cluster] * _novelty) {
            create_new_cluster = false;
            break;
        }
    }

    if (create_new_cluster) {
        Eigen::MatrixXf covariance = Eigen::MatrixXf::Identity(D, D) * _var_initial;

        _inv_2pi_d = 1.0f / (std::pow(2 * 3.14159265, float(D) * 0.5f));

        // Create a new cluster
        _means.push_back(input);
        _covariances.push_back(covariance);
        _inv_covariances.push_back(covariance.inverse());
        _weights.push_back(value);
        _probabilities.push_back(1.0f / sum_sp);
        _sprobabilities.push_back(1.0f);
        _gaussian_normalizations.push_back(_inv_2pi_d / std::sqrt(covariance.norm()));

        // Adjust the probabilities of the other clusters
        float inv_sum_sp = 1.0f / (sum_sp + 1.0f);

        for (std::size_t cluster=0; cluster<numberOfClusters()-1; ++cluster) {
            _probabilities[cluster] = _sprobabilities[cluster] * inv_sum_sp;
        }
    } else {
        // Probabilities of the clusters given the inputs
        float *cluster_probabilities = NEW_VEC;
        float *input_probabilities = NEW_VEC;

        probabilitiesOfInputs(input_probabilities, input, value);
        probabilitiesOfClusters(cluster_probabilities, input_probabilities);

        // Find the cluster with the greatest probability
        auto it = std::max_element(input_probabilities, input_probabilities + numberOfClusters());
        int cluster = std::distance(input_probabilities, it);
        float proba = cluster_probabilities[cluster];

        // Update the cluster according to
        // "An Incremental Probabilistic Neural Network for Regression and Reinforcement Learning Tasks"
        float new_sproba = _sprobabilities[cluster] + proba;
        float learning_factor = proba / new_sproba;
        Eigen::VectorXf delta_mean = input - _means[cluster];
        Eigen::VectorXf delta_mean_factor = learning_factor * delta_mean;
        Eigen::VectorXf delta_prev_mean = delta_mean - delta_mean_factor;

        _sprobabilities[cluster] = new_sproba;
        _probabilities[cluster] = new_sproba / (sum_sp + proba);
        _means[cluster] += delta_mean_factor;
        _weights[cluster] += learning_factor * (value - _weights[cluster]);
        _covariances[cluster] = _covariances[cluster] +
                                delta_mean_factor * delta_mean_factor.transpose() +
                                learning_factor * (
                                    delta_prev_mean * delta_prev_mean.transpose() -
                                    _covariances[cluster]
                                );
        _inv_covariances[cluster] = _covariances[cluster].inverse();
        _gaussian_normalizations[cluster] = _inv_2pi_d / std::sqrt(_covariances[cluster].norm());
    }
}

float GaussianMixture::probabilityOfInput(unsigned int cluster, const Eigen::VectorXf &input) const
{
    return _gaussian_normalizations[cluster] * std::exp(
        (-0.5f * (
            (input - _means[cluster]).transpose() *
            _inv_covariances[cluster] *
            (input - _means[cluster])
        ))(0)
    );
}

void GaussianMixture::probabilitiesOfInputs(float *out, const Eigen::VectorXf &input, float value) const
{
    // Probability of the input for all the clusters
    for (std::size_t cluster=0; cluster<numberOfClusters(); ++cluster) {
        out[cluster] = probabilityOfInput(cluster, input) * _probabilities[cluster];
    }
}

void GaussianMixture::probabilitiesOfClusters(float *out, const Eigen::VectorXf &input, float value) const
{
    float *input_probas = NEW_VEC;

    probabilitiesOfInputs(input_probas, input, value);
    probabilitiesOfClusters(out, input_probas);
}

void GaussianMixture::probabilitiesOfClusters(float *out, float *input_probabilities) const
{
    // Compute, for each cluster, p(input|cluster)*p(cluster) / sum(probabilities)
    float proba_x = std::accumulate(input_probabilities, input_probabilities + numberOfClusters(), 0.0f);
    float inv_proba_x = 1 / proba_x;

    for (std::size_t cluster=0; cluster<numberOfClusters(); ++cluster) {
        out[cluster] = input_probabilities[cluster] * inv_proba_x;
    }
}
