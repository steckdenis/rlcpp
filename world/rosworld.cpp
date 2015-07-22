#include "rosworld.h"

#include <sched.h>

/**
 * @brief Return the number of possible actions based on a list of producers
 *
 * This method is required because AbstractWorld needs a num_actions parameter,
 * that cannot be computed in the body of the RosWorld constructor.
 */
static unsigned int possibleActions(const std::vector<RosWorld::Producer *> &publications)
{
    unsigned int rs = 0;

    for (RosWorld::Producer *producer : publications) {
        rs += producer->values.size();
    }

    return rs;
}

RosWorld::RosWorld(const std::vector<RosWorld::Parser *> &subscriptions,
                   const std::vector<RosWorld::Producer *> &publications)
: AbstractWorld(possibleActions(publications)),
  _subscriptions(subscriptions),
  _publications(publications),
  _state(subscriptions.size()),
  _updated(false)
{
    // Create the list of actions
    Action a;

    for (RosWorld::Producer *producer : publications) {
        for (float value : producer->values) {
            a.producer = producer;
            a.value = value;

            _actions.push_back(a);
        }
    }

    // Tell all the parsers which state variable they correspond to
    for (std::size_t i=0; i<subscriptions.size(); ++i) {
        subscriptions[i]->var = &_state[i];
        subscriptions[i]->updated = &_updated;
    }
}

RosWorld::~RosWorld()
{
    // Delete all the parsers and producers
    for (Parser *p : _subscriptions) {
        delete p;
    }
    
    for (Producer *p : _publications) {
        delete p;
    }
}

void RosWorld::initialState(std::vector<float> &state)
{
    // Null state for the first episode, then this is the last state of the previous
    // episode (reset() does nothing)
    state = _state;
}

void RosWorld::reset()
{
    // Do nothing
}

void RosWorld::step(unsigned int action, bool &finished, float &reward, std::vector<float> &state)
{
    // Publish the action
    Action &a = _actions[action];
    
    a.producer->publishValue(a.value);

    // Wait for at least one observation to arrive
    while (!_updated) {
        ros::spinOnce();
        sched_yield();
    }

    _updated = false;

    // The parsers have updated the state, just return it
    std::size_t len = _state.size() - 1;

    state.resize(len);
    std::copy(_state.begin(), _state.begin() + len, state.begin());
    
    reward = _state[len];
    finished = false;
}
