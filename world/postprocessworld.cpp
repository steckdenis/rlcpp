#include "postprocessworld.h"

#include <cmath>
#include <iostream>

PostProcessWorld::PostProcessWorld(AbstractWorld *world)
: AbstractWorld(world->numActions()),
  _world(world)
{
}

PostProcessWorld::~PostProcessWorld()
{
    delete _world;
}

void PostProcessWorld::initialState(std::vector <float> &state)
{
    _world->initialState(state);
    processState(state);
}

void PostProcessWorld::reset()
{
    _world->reset();
}

void PostProcessWorld::step(unsigned int action,
                       bool &finished,
                       float &reward,
                       std::vector<float> &state)
{
    // Let the wrapped world compute the step, then post-process it
    _world->step(action, finished, reward, state);
    processState(state);
}
