#include "scaleworld.h"

#include <assert.h>

ScaleWorld::ScaleWorld(AbstractWorld *world,
                       const std::vector<float> &weights)
: PostProcessWorld(world),
  _weights(weights)
{
}

ScaleWorld::~ScaleWorld()
{
    delete _world;
}

void ScaleWorld::processState(std::vector<float> &state)
{
    assert(state.size() == _weights.size());

    for (std::size_t i=0; i<state.size(); ++i) {
        state[i] *= _weights[i];
    }
}
