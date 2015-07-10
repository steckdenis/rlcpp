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