#ifndef __ONEOFNWORLD_H__
#define __ONEOFNWORLD_H__

#include "abstractworld.h"

/**
 * @brief World that wraps another ones and encodes its (discrete) states using
 *        a one-hot notation.
 */
class OneOfNWorld : public AbstractWorld
{
    public:
        /**
         * @param world World to be wrapped
         * @param minimums Minimum value that each state variable can take
         * @param maximums Maximum value that each state variable can take
         */
        OneOfNWorld(AbstractWorld *world,
                    const std::vector<int> &minimums,
                    const std::vector<int> &maximums);
        virtual ~OneOfNWorld();

        virtual void initialState(std::vector<float> &state);
        virtual void reset();
        virtual void step(unsigned int action,
                          bool &finished,
                          float &reward,
                          std::vector<float> &state);

    private:
        /**
         * @brief Postprocess a state so that it is encoded in a one-host form
         */
        void processState(std::vector<float> &state);

    private:
        AbstractWorld *_world;
        std::vector<int> _minimums;
        std::vector<int> _maximums;

        int _postprocessed_state_size;
};

#endif