#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 11 12:08:45 2025

@author: Travis Czechorski

Email: travis.czechorski@gmail.com

"""

# Has multi-dimensional arrays and matrices.
# Has a large collection of mathematical functions to operate on these arrays.
import numpy as np

# Data manipulation and analysis.
import pandas as pd

# Data visualization tools.
import seaborn as sns

import mesa

from numpy.random import choice

lang_dict = {"English": 0.2, "French": 0.35, "Dutch": 0.35, "German": 0.1}

draw = choice(list(lang_dict.keys()), 2, p=list(lang_dict.values()))

class MoneyAgent(mesa.Agent):
    """An agent with fixed initial wealth."""

    def __init__(self, model):
        # Pass the parameters to the parent class.
        super().__init__(model)

        # Create the agent's attribute and set the initial values.
        self.wealth = 1
        self.lang = choice(list(lang_dict.keys()), 2, p=list(lang_dict.values()), replace=False)
        
    def exchange(self):
        # Verify agent has some wealth
        if self.wealth > 0:
            can_talk = False
            other_agent = self.random.choice(self.model.agents)
            langs_1 = self.lang
            langs_2 = other_agent.lang
            
            for i in langs_1:
                if i in langs_2:
                    can_talk = True
                else:
                    pass
            
            if other_agent is not None and can_talk is True:
                other_agent.wealth += 1
                self.wealth -= 1
                
            elif can_talk is False:
                print("Exchange not possible. Agent 1 speaks {} and {}. Agent 2 speaks {} and {}".format(langs_1[0], langs_1[1], langs_2[0], langs_2[1]))

    def say_hi(self):
        # The agent's step will go here.
        # For demonstration purposes we will print the agent's unique_id
        print("Hi, I am an agent, you can call me {}. I have {} wealth. LOL I speak {} and {}".format(self.unique_id, self.wealth, self.lang[0], self.lang[1]))


class MoneyModel(mesa.Model):
    """A model with some number of agents."""

    def __init__(self, n):
        super().__init__()
        self.num_agents = n

        # Create agents
        MoneyAgent.create_agents(model=self, n=n)

    def step(self):
        """Advance the model by one step."""
        # This function psuedo-randomly reorders the list of agent objects and
        # then iterates through calling the function passed in as the parameter
        self.agents.shuffle_do("exchange")
        
model = MoneyModel(10)  # Tells the model to create 10 agents
for _ in range(30):  # Runs the model for 30 steps;
    model.step()

# Note: An underscore is common convention for a variable that is not used.

agent_wealth = [a.wealth for a in model.agents]
# Create a histogram with seaborn
g = sns.histplot(agent_wealth, discrete=True)
g.set(title="Wealth distribution", xlabel="Wealth", ylabel="number of agents");  # The semicolon is just to avoid printing the object representation