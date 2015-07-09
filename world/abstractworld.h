#ifndef __ABSTRACTWORLD_H__
#define __ABSTRACTWORLD_H__

#include <vector>

class AbstractModel;
class AbstractLearning;
class Episode;

/**
 * @brief Provide states and rewards in response to actions
 */
class AbstractWorld
{
    public:
        AbstractWorld(unsigned int num_actions);
        virtual ~AbstractWorld() {}

        /**
         * @brief Number of actions that can be taken in this world.
         *
         * Actions are numeroted from 0 to numActions() - 1.
         */
        unsigned int numActions() const;

        /**
         * @brief Reset the environment to its initial state
         */
        virtual void reset() = 0;

        /**
         * @brief Return the initial state of this world
         *
         * This initial state is used as the first state of an episode that is
         * just started.
         */
        virtual void initialState(std::vector<float> &state) = 0;

        /**
         * @brief Execute an action in the world and observe a reward, a new state,
         *        and whether the state is a termination state.
         */
        virtual void step(unsigned int action,
                          bool &finished,
                          float &reward,
                          std::vector<float> &state) = 0;

        /**
         * @brief Produce a file of any format that represents the contents of
         *        the given model mapped to this world.
         */
        virtual void plotModel(AbstractModel *model) {}

        /**
         * @brief Run an agent in the world for a given number of episodes
         * 
         * @param model Model used for learning
         * @param learning Learning algorithm
         * @param num_episodes Number of episodes run
         * @param max_episode_length Maximum number of time steps per episode
         * @param batch_size Number of episodes to run between model updates
         *
         * @return A list of episodes. The caller must delete the episodes.
         */
        std::vector<Episode *> run(AbstractModel *model,
                                   AbstractLearning *learning,
                                   unsigned int num_episodes,
                                   unsigned int max_episode_length,
                                   unsigned int batch_size);
    private:
        unsigned int _num_actions;
};

#endif