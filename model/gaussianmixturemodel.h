#ifndef __GAUSSIANMIXTUREMODEL_H__
#define __GAUSSIANMIXTUREMODEL_H__

#include "abstractmodel.h"

#include <Eigen/Dense>
#include <vector>
#include <random>

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
         */
        GaussianMixtureModel(float var_initial, float novelty, float noise);
        virtual ~GaussianMixtureModel();

        virtual void values(Episode *episode, std::vector<float> &rs);
        virtual void learn(const std::vector<Episode *> &episodes);

    private:
        void vectorToVectorXf(const std::vector<float> &stl, Eigen::VectorXf &eigen);

    private:
        float _var_initial;
        float _novelty;

        std::normal_distribution<float> _noise_distribution;
        std::default_random_engine _random_engine;
        std::vector<GaussianMixture *> _models;     /*!< @brief One model per action */
};

#endif