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

#ifndef __MODELWORLD_H__
#define __MODELWORLD_H__

#include <world/abstractworld.h>

/**
 * @brief World that approximates a model of another world
 */
class ModelWorld : public AbstractWorld
{
    public:
        /**
         * @brief Create a new model world
         *
         * @param world World for which a model is built.
         * @param model Model used to approximate the world.
         * @param encoder Encoder to use to encode the states of the world. For
         *                instance, neural networks work best when states are
         *                normalized or one-hot encoded.
         */
        ModelWorld(AbstractWorld *world, AbstractModel *model, Episode::Encoder encoder);
        ~ModelWorld();

        virtual void initialState(std::vector<float> &state);
        virtual void reset();
        virtual void step(unsigned int action,
                          bool &finished,
                          float &reward,
                          std::vector<float> &state);
        virtual void stepSupervised(unsigned int action,
                                    const std::vector<float> &target_state,
                                    float reward);

        /**
         * @brief Use a list of episodes in order to learn the model of the world
         *
         * This function uses the sequence of states encountered in the episode
         * and the rewards received.
         */
        void learn(const std::vector<Episode *> episodes);

    protected:
        /**
         * @brief Encode a world state and an action to a model state (that
         *        contains the action so that the model can predict
         *        state,action -> state',reward)
         */
        virtual void makeModelState(const std::vector<float> &world_state,
                                    unsigned int action,
                                    std::vector<float> &model_state);

    protected:
        AbstractWorld *_world;
        AbstractModel *_model;
        Episode::Encoder _encoder;

        Episode *_episode;

        std::vector<float> _world_state;
        std::vector<float> _model_state;
        std::vector<float> _values;
};

#endif
