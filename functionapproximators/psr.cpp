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

#include "psr.h"

#include <assert.h>

PSR::PSR(unsigned int history_length,
         unsigned int ao_length,
         unsigned int test_length,
         unsigned int rank)
: _history_length(history_length),
  _ao_length(ao_length),
  _test_length(test_length),
  _rank(rank),
  _num_train(0)
{
    assert(_rank <= _history_length);
    assert(_rank <= _test_length);

    // No history seen, average is currently null
    _mu_h = Eigen::VectorXf::Zero(_history_length);

    // Create the inverted Sigma_ao_ao matrix, that is the identity (the inverse
    // of the identity is the identity)
    _inv_sigma_ao = Eigen::MatrixXf::Identity(_ao_length, _ao_length);

    // Sigma_th, never computed in this model, is a observation_length * history_length
    // matrix (in height * width notation). U and V are identities (of the proper
    // dimension), S is zero (same dimension as Sigma_th). Because we
    // use thin rank-n SVD decomposition, the width of these matrices is _rank,
    // and _s is a square _rank*_rank matrix.
    _u = Eigen::MatrixXf::Identity(_test_length, _rank);
    _s = Eigen::MatrixXf::Zero(_rank, _rank);
    _inv_s = Eigen::MatrixXf::Zero(_rank, _rank);
    _v = Eigen::MatrixXf::Identity(_history_length, _rank);

    // Initial estimates of b1 and binf. Bao is computed from the null _candidate_bao
    // and is therefore also zero
    _b1 = Eigen::MatrixXf::Zero(_rank, 1);
    _pre_belief = _b1;
    _belief = _b1;
    _binf = Eigen::MatrixXf::Zero(1, _rank);
    _bao = Eigen::MatrixXf::Zero(_rank, _rank);

    // The candidate Bao is a _rank * _ao_length * _rank cube (height * width * depth).
    // At first, this cube is uninitialized
    _candidate_bao.reserve(_rank);

    for (unsigned int i=0; i<_rank; ++i) {
        _candidate_bao.push_back(Eigen::MatrixXf::Zero(_rank, _ao_length));
    }

    // Allocate memory for the other matrices (without initialization)
    _uct.resize(_rank + 1, 1);
    _hvd.resize(1, _rank + 1);
    _k.resize(_rank + 1, _rank + 1);

    _bao_third_dim.resize(_rank);
    _bao_onesecond_dim.resize(_rank, _ao_length);

    // _k, _c and _d don't need to be initialized. They are member fields only
    // so that memory can be reused between updates
}

void PSR::train(const Eigen::VectorXf &history,
                const Eigen::VectorXf &ao,
                const Eigen::VectorXf &future)
{
    // Update the average history
    _mu_h += history;

    // Update _inv_sigma_ao using the Sherman-Morrison formula
    _inv_sigma_ao.array() -=
        ((_inv_sigma_ao * ao) * (ao.transpose() * _inv_sigma_ao)).array() /
        ((ao.transpose() * _inv_sigma_ao * ao).value() + 1.0f);

    // Compute the C and D matrices
    _ct = (
        Eigen::MatrixXf::Identity(_test_length, _test_length) -
        _u * _u.transpose()
    ) * future;
    _dt = (
        Eigen::MatrixXf::Identity(_history_length, _history_length) -
        _v * _v.transpose()
    ) * history;
    _c = _ct / _ct.norm();
    _d = _dt / _dt.norm();

    // Updating K requires the construction of two temporary vectors with one
    // more element that C and D
    _uct.block(0, 0, _rank, 1) = _u.transpose() * future;
    _uct.block(_rank, 0, 1, 1) = _c.transpose() * future;

    _hvd.block(0, 0, 1, _rank) = history.transpose() * _v;
    _hvd.block(0, _rank, 1, 1) = history.transpose() * _d;

    // K can now be updated easily
    _k.setZero();
    _k.block(0, 0, _rank, _rank) = _s;
    _k.noalias() += _uct * _hvd;

    // SVD decomposition of K
    _k_jacobi.compute(_k, Eigen::ComputeThinU | Eigen::ComputeThinV);

    // Update U and V with the equation U = [U C] * _k_jacobi.matrixU(). The update
    // is done in two times to that U and C don't need to be concatenated
    const auto &matrixU = _k_jacobi.matrixU();
    const auto &matrixV = _k_jacobi.matrixV();

    _u = _u * matrixU.block(0, 0, _rank, _rank) + _c * matrixU.block(_rank, 0, 1, _rank);
    _v = _v * matrixV.block(0, 0, _rank, _rank) + _d * matrixV.block(_rank, 0, 1, _rank);
    _s.diagonal() = _k_jacobi.singularValues().block(0, 0, _rank, 1);

    if (_s.diagonal().minCoeff() < 1e-10) {
        // Not yet enough data, skip learning here
        return;
    } else {
        _inv_s.diagonal() = _s.diagonal().cwiseInverse();                       // Inverting a diagonal matrix is as simple as inverting the terms on the diagonal
    }

    // Keep U and V orthogonal (this correction is done every once and then)
    if (++_num_train == 200) {
        _num_train = 0;

        fixUV();
    }

    // U and V are orthogonal matrices, so UU' and VV' (and S⁻¹V'VS) are
    // identity matrices. Furthermore, U and V have not grown since last call to
    // learn(), so the computation of Bupdate is not needed (no added zero, only
    // identities). Add "Bnew" to Bao.
    _bao_third_dim = _inv_s * (_v.transpose() * history);
    _bao_onesecond_dim = _u.transpose() * future * ao.transpose();

    for (unsigned int i=0; i<_rank; ++i) {
        // NOTE: A lambda has been removed ("weight vector", Bao(i, i, i) = lambda_i)
        _candidate_bao[i] += _bao_onesecond_dim * _bao_third_dim(i);
    }

    // Compute the updated PSR parameters
    _b1 = (_s * _v.transpose()).col(0);                 // NOTE: Taking the first column is equivalent to the multiplication by "e" in the article, with the definition of e given in "Closing the loop with PSR"
    _binf = _mu_h.transpose() * _v * _inv_s;

    // _bao is updated at each action-observation
}

void PSR::reset()
{
    // Reset the current belief to the initial belief
    _belief = _b1;
}

float PSR::update(const Eigen::VectorXf &ao)
{
    computeBao(ao);

    // Update the current belief
    _pre_belief = _bao * _belief;
    _belief = _pre_belief / (_binf * _pre_belief).value();
}

float PSR::predict(const Eigen::VectorXf &ao)
{
    computeBao(ao);

    // Probability that the observation has been observed
    return (_binf * _bao * _belief).value();
}

void PSR::fixUV()
{
    Eigen::HouseholderQR<Eigen::MatrixXf> uQR(_u);
    Eigen::HouseholderQR<Eigen::MatrixXf> vQR(_v);
    Eigen::JacobiSVD<Eigen::MatrixXf> svd(
        uQR.matrixQR().triangularView<Eigen::Upper>() * _s * vQR.matrixQR().triangularView<Eigen::Upper>().transpose(),
        Eigen::ComputeThinU | Eigen::ComputeThinV
    );

    _u = (uQR.householderQ() * svd.matrixU()).block(0, 0, _test_length, _rank);
    _v = (vQR.householderQ() * svd.matrixV()).block(0, 0, _history_length, _rank);

    _s.diagonal() = svd.singularValues().block(0, 0, _rank, 1);
    _inv_s.diagonal() = _s.diagonal().cwiseInverse();
}

void PSR::computeBao(const Eigen::VectorXf &ao)
{
    // Compute the _bao corresponding to this action-observation
    for (unsigned int i=0; i<_rank; ++i) {
        // This computes rank * rank Bao column by column, each rank*AO * AO*1
        // will give a rank*1 column.
        _bao.col(i) = _candidate_bao[i] * (_inv_sigma_ao.transpose() * ao);
    }
}
