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
import matplotlib.pyplot as plt  # Add this with your other imports
# Data manipulation and analysis.
import pandas as pd
# Data visualization tools.
import seaborn as sns
import mesa
from numpy.random import choice
from mesa.discrete_space import CellAgent, OrthogonalMooreGrid

lang_dict_flanders = {"English": 0.3, "French": 0.2, "Dutch": 0.4, "German": 0.1}
lang_dict_wallonia = {"English": 0.1, "French": 0.7, "Dutch": 0.1, "German": 0.1}

#draw = choice(list(lang_dict.keys()), 2, p=list(lang_dict.values()))

class MoneyAgent(CellAgent):
    """An agent with fixed initial wealth."""

    def __init__(self, model, cell):
        # Pass the parameters to the parent class.
        super().__init__(model)
        
        self.cell = cell  # Instantiate agent with location (x,y)
        
        # Get grid width (columns) and height (rows)
        grid_width = model.grid.width  # Number of columns (x-axis)
        grid_height = model.grid.height  # Number of rows (y-axis)

        # print(f"Grid dimensions: {grid_width} x {grid_height}")
        
        
        self.pos = cell.coordinate  # Access the 'coordinate' attribute
        self.x = self.pos[0]
        self.y = self.pos[1]
        # print(f"Agent created at position: X = {self.x}, Y = {self.y} ")  # e.g., (4, 9)

        # Create the agent's attribute and set the initial values.
        self.wealth = 1
        boundary = (grid_height / 2) - 1
        
        if self.y <= boundary:
            self.lang = choice(list(lang_dict_wallonia.keys()), 2, p=list(lang_dict_wallonia.values()), replace=False)
        else:
            self.lang = choice(list(lang_dict_flanders.keys()), 2, p=list(lang_dict_flanders.values()), replace=False)
        
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
                pass #print("Exchange not possible. Agent 1 speaks {} and {}. Agent 2 speaks {} and {}".format(langs_1[0], langs_1[1], langs_2[0], langs_2[1]))

    def say_hi(self):
        # The agent's step will go here.
        # For demonstration purposes we will print the agent's unique_id
        print("Hi, I am an agent, you can call me {}. I have {} wealth. LOL I speak {} and {}".format(self.unique_id, self.wealth, self.lang[0], self.lang[1]))
        
    # Move Function
    def move(self):
        self.cell = self.cell.neighborhood.select_random_cell()

    def give_money(self):
        cellmates = [a for a in self.cell.agents if a is not self]  # Get all agents in cell
        
        

        if self.wealth > 0 and cellmates:
            other_agent = self.random.choice(cellmates)
            
            can_talk = False
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
                pass #print("Exchange not possible. Agent 1 speaks {} and {}. Agent 2 speaks {} and {}".format(langs_1[0], langs_1[1], langs_2[0], langs_2[1]))

            


class MoneyModel(mesa.Model):
    """A model with some number of agents."""

    def __init__(self, n, width, height, seed=None):
        super().__init__()
        self.num_agents = n
        
        
        # Instantiate an instance of Moore neighborhood space
        self.grid = OrthogonalMooreGrid((width, height), torus=True, capacity=10, random=self.random)

        # Create agents
        agents = MoneyAgent.create_agents(self, self.num_agents, 
                                          self.random.choices(self.grid.all_cells.cells, k=self.num_agents))

    def step(self):
        """Advance the model by one step."""
        # This function psuedo-randomly reorders the list of agent objects and
        # then iterates through calling the function passed in as the parameter
        self.agents.shuffle_do("move")
        self.agents.do("give_money")
        
    def count_language_per_cell(self, language):
        """Returns a grid where each cell contains the count of agents speaking 'language'."""
        lang_counts = np.zeros((self.grid.width, self.grid.height))
        
        for cell in self.grid.all_cells:
            x, y = cell.coordinate
            # Count agents in this cell who speak 'language'
            count = sum(
                1 for agent in cell.agents 
                if language in agent.lang  # Check if the agent speaks the language
            )
            lang_counts[x, y] = count
        
        return lang_counts
        
model = MoneyModel(100, 10, 10)  # Tells the model to create 10 agents
for _ in range(20):  # Runs the model for 30 steps;
    model.step()

# Note: An underscore is common convention for a variable that is not used.

agent_counts = np.zeros((model.grid.width, model.grid.height))

for cell in model.grid.all_cells:
    agent_counts[cell.coordinate] = len(cell.agents)
# Plot using seaborn, with a visual size of 5x5
g = sns.heatmap(agent_counts, cmap="viridis", annot=True, cbar=False, square=True)
g.figure.set_size_inches(5, 5)
g.set(title="Number of agents on each cell of the grid");
plt.savefig("/home/travis/Desktop/Python/Agents.png")
plt.close()
# After running the model...
dutch_counts = model.count_language_per_cell("Dutch")
french_counts = model.count_language_per_cell("French")
english_counts = model.count_language_per_cell("English")
german_counts = model.count_language_per_cell("German")

# # Plot
# plt.figure(figsize=(5, 5))
# sns.heatmap(
#     dutch_counts, 
#     cmap="viridis", 
#     annot=True, 
#     cbar=False, 
#     square=True,
#     fmt=".0f"  # Display counts as integers
# )
# plt.title("Number of Dutch-speaking agents per cell")
# plt.close()

# Plot Dutch
sns.heatmap(dutch_counts, cmap="Blues", annot=True, square=True)
plt.title("Dutch Speakers")
plt.savefig("/home/travis/Desktop/Python/DutchSpeakers.png")
plt.close()

# Plot French
sns.heatmap(french_counts, cmap="Reds", annot=True, square=True)
plt.title("French Speakers")
plt.savefig("/home/travis/Desktop/Python/FrenchSpeakers.png")
plt.close()

# Plot English
sns.heatmap(english_counts, cmap="Greens", annot=True, square=True)
plt.title("English Speakers")
plt.savefig("/home/travis/Desktop/Python/EnglishSpeakers.png")
plt.close()

# Plot German
sns.heatmap(german_counts, cmap="Purples", annot=True, square=True)
plt.title("German Speakers")
plt.savefig("/home/travis/Desktop/Python/GermanSpeakers.png")
plt.close()