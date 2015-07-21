#ifndef __SCALEWORLD_H__
#define __SCALEWORLD_H__

#include "postprocessworld.h"

/**
 * @brief World that wraps another ones and encodes scales its observations.
 *
 * This can be used for normalization (for neural networks), or for multiplying
 * some state variables by 0.0, hence making the world partially observable.
 */
class ScaleWorld : public PostProcessWorld
{
    public:
        /**
         * @param world World to be wrapped
         * @param weights Values by which the state variables are multiplied
         */
        ScaleWorld(AbstractWorld *world,
                    const std::vector<float> &weights);
        virtual ~ScaleWorld();

    protected:
        /**
         * @brief Scale a state according to the weight vector
         */
        virtual void processState(std::vector<float> &state) override;

    private:
        AbstractWorld *_world;
        std::vector<float> _weights;
};

#endif