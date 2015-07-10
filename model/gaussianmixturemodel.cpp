#include "gaussianmixturemodel.h"
#include "episode.h"
#include "functionapproximators/gaussianmixture.h"

#include <algorithm>
#include <random>
#include <iostream>

GaussianMixtureModel::GaussianMixtureModel(float var_initial, float novelty, float noise)
: _var_initial(var_initial),
  _novelty(novelty),
  _noise_distribution(0.0f, noise)
{
}

GaussianMixtureModel::~GaussianMixtureModel()
{
    // Delete all the models
    for (GaussianMixture *model : _models) {
        delete model;
    }
}

void GaussianMixtureModel::values(Episode *episode, std::vector<float> &rs)
{
    if (_models.size() == 0) {
        // No model available, clear out rs
        rs.resize(episode->valueSize());
        std::fill(rs.begin(), rs.end(), 0.0f);
    } else {
        // Convert the last state to an Eigen vector
        Eigen::VectorXf input(episode->stateSize());

        episode->state(episode->length() - 1, rs);
        vectorToVectorXf(rs, input);

        // Pass this input to all the models
        rs.resize(episode->valueSize());

        for (std::size_t i=0; i<rs.size(); ++i) {
            rs[i] = _models[i]->value(input);
        }
    }
}

void GaussianMixtureModel::learn(const std::vector<Episode *> &episodes)
{
    std::vector<float> state;
    std::vector<float> values;

    for (Episode *episode : episodes) {
        Eigen::VectorXf input(episode->stateSize());

        // Create the models if needed
        if (_models.size() == 0) {
            for (unsigned int a=0; a<episode->valueSize(); ++a) {
                _models.push_back(new GaussianMixture(_var_initial, _novelty));
            }
        }

        // Train the models on this episode
        for (unsigned int t=0; t < episode->length() - 1; ++t) {
            unsigned int action = episode->action(t);

            episode->state(t, state);
            episode->values(t, values);

            // Update the model of the selected action
            vectorToVectorXf(state, input);

            _models[action]->setValue(input, values[action]);
        }
    }

    // Print the number of clusters in the models
    std::cout << "[Gaussian mixture model] Number of clusters:";

    for (GaussianMixture *model : _models) {
        std::cout << ' ' << model->numberOfClusters();
    }

    std::cout << std::endl;
}

void GaussianMixtureModel::vectorToVectorXf(const std::vector<float> &stl, Eigen::VectorXf &eigen)
{
    for (std::size_t i=0; i<stl.size(); ++i) {
        // Add a bit of noise in order to avoid having vectors too close to each
        // other, and hence having a null variance.
        eigen(i) = stl[i] + _noise_distribution(_random_engine);
    }
}

