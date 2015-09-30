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

#ifndef __PSR_H__
#define __PSR_H__

#include <Eigen/Dense>
#include <Eigen/Jacobi>
#include <Eigen/Householder>
#include <vector>

/**
 * @brief Predict observations based on actions
 *
 * This model allows to model sequences. It is trained by being given histories,
 * action-observations and futures (the fact that a future is required makes
 * this model learn off-line).
 *
 * Histories and futures are sequences of action-observations. An action-observation
 * pair is a vector of floats, each float representing the probability of a
 * "feature". This is like one-hot encoding continuous values. The action-observation
 * {4, 0.4, 0.77} can be encoded as {0.1, 0.1, 0.0, 0.12, 0.16, 0.56, 0.01, 0.9}
 * for instance. The probabilities don't need to sum to one.
 *
 * Encoding the actions-observations is the responsibility of the user of this
 * model.
 *
 * Boots and Gordon, An Online Spectral Learning Algorithm for Partially Observable
 * Nonlinear Dynamical Systems, 2011
 */
class PSR
{
    public:
        /**
         * @brief Constructor
         *
         * @param history_length Length of a history, in the number of features
         *                       used to represent an history.
         * @param ao_length Number of features of an action-observation
         * @param test_length Number of features of a test (sequence of action-observations)
         * @param rank Number of singular values used to express the system.
         *             rank-N models can express POMDPs with at least N hidden
         *             variables.
         */
        PSR(unsigned int history_length,
            unsigned int ao_length,
            unsigned int test_length,
            unsigned int rank);

        /**
         * @brief Incrementally train the model with an history, present and future
         *
         * @param history Concatenated sequence of action-observations pairs that
         *                happened just before @p ao
         * @param ao      Features of an action-observation pair
         * @param future  Concatenated sequence of action-observations pairs that
         *                happen just after @p ao
         *
         * @warning The first element of @p history must be a 1.
         */
        void train(const Eigen::VectorXf &history,
                   const Eigen::VectorXf &ao,
                   const Eigen::VectorXf &future);

        /**
         * @brief Reset the model to its initial state
         */
        void reset();

        /**
         * @brief Advance the model one step by providing an action-observation
         *        pair.
         *
         * @return The probability that the observation was observed
         */
        float update(const Eigen::VectorXf &ao);

        /**
         * @brief Return the probability that a given observation is observed if
         *        a given action is performed.
         */
        float predict(const Eigen::VectorXf &ao);

    private:
        /**
         * @brief Fix U and V so that they remain orthogonal
         */
        void fixUV();

        /**
         * @brief Update the Bao matrix given the internal state of the model and
         *        an action-observation pair.
         */
        void computeBao(const Eigen::VectorXf &ao);

    private:
        // Many of the following matrices are temporary values kept in the class
        // so that memory can be reused between calls to train() and update().
        Eigen::VectorXf _mu_h;                  /*!< @brief Average features over all the histories */
        Eigen::MatrixXf _inv_sigma_ao;          /*!< @brief Inverse of the Sigma_AO,AO covariance matrix */
        Eigen::MatrixXf _u, _s, _inv_s, _v;     /*!< @brief U, S and V matrices of the SVD decomposition of _inv_sigma_ao */
        Eigen::MatrixXf _ct, _c, _dt, _d;       /*!< @brief C and D vectors used for computing K */
        Eigen::MatrixXf _uct, _hvd;             /*!< @brief Vectors with one more element than C and D used for computing K */
        Eigen::MatrixXf _k;                     /*!< @brief K matrix used for updating U, S and V */
        Eigen::MatrixXf _b1;                    /*!< @brief Belief after a reset (approximately the expected belief over all the histories) */
        Eigen::MatrixXf _binf;                  /*!< @brief Normalization factor over the beliefs */
        Eigen::MatrixXf _pre_belief;            /*!< @brief Vector used when computing the new belief */
        Eigen::MatrixXf _belief;                /*!< @brief Belief at the current time step */

        // SVD solver used for computing K so that its internal matrices are cached
        Eigen::JacobiSVD<Eigen::MatrixXf> _k_jacobi;

        // size1*size2 slices, there are size3 slices
        std::vector<Eigen::MatrixXf> _candidate_bao;    /*!< @brief Bao computed from S, U, V and others */
        Eigen::MatrixXf _bao_onesecond_dim;             /*!< @brief Matrix of coefficients used to compute the first and second dimensions of Bao */
        Eigen::VectorXf _bao_third_dim;                 /*!< @brief Vector of coefficients used to compute the third dimension of Bao */
        Eigen::MatrixXf _bao;                           /*!< @brief Final Bao, computed from _candidate_bao and adjustment factors. This is a matrix, not a 3D tensor. */

        unsigned int _history_length;
        unsigned int _ao_length;
        unsigned int _test_length;
        unsigned int _rank;

        unsigned int _num_train;
};

#endif
