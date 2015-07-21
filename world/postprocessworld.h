#ifndef __POSTPROCESSWORLD_H__
#define __POSTPROCESSWORLD_H__

#include "abstractworld.h"

/**
 * @brief World that wraps another ones and postprocesses the states it produces
 */
class PostProcessWorld : public AbstractWorld
{
    public:
        /**
         * @param world World to be wrapped
         * @param weights Values by which the state variables are multiplied
         */
        PostProcessWorld(AbstractWorld *world);
        virtual ~PostProcessWorld();

        virtual void initialState(std::vector<float> &state);
        virtual void reset();
        virtual void step(unsigned int action,
                          bool &finished,
                          float &reward,
                          std::vector<float> &state);

    protected:
        /**
         * @brief Post-process a state
         */
        virtual void processState(std::vector<float> &state) = 0;

    private:
        AbstractWorld *_world;
};

#endif