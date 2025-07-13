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
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
# Data manipulation and analysis.
import pandas as pd
# Data visualization tools.
import seaborn as sns
import mesa
from numpy.random import choice
from mesa.discrete_space import CellAgent, OrthogonalMooreGrid
import random

lang_dict_flanders = {"English": 0.8, "French": 0.1, "German": 0.1}
lang_dict_wallonia = {"English": 0.5, "Dutch": 0.1, "German": 0.4}

def plot_language_wealth_heatmap(model, step_num):
    # Initialize dictionaries to track language counts and wealth per cell
    cell_data = {}
    for x in range(model.grid.width):
        for y in range(model.grid.height):
            cell_data[(x, y)] = {'Dutch': 0, 'French': 0, 'German': 0, 'English': 0,
                                'Dutch_wealth': 0, 'French_wealth': 0, 
                                'German_wealth': 0, 'English_wealth': 0}

    # Populate cell data
    for agent in model.agents:
        x, y = agent.pos
        native_lang = agent.native
        cell_data[(x, y)][native_lang] += 1
        cell_data[(x, y)][f"{native_lang}_wealth"] += agent.wealth

    # Prepare grids
    lang_grid = np.zeros((model.grid.height, model.grid.width), dtype=object).T
    wealth_grid = np.zeros((model.grid.height, model.grid.width)).T
    
    for (x, y), data in cell_data.items():
        # Find most prevalent language in cell
        counts = {lang: data[lang] for lang in ['Dutch', 'French', 'German', 'English']}
        dominant_lang = max(counts.keys(), key=lambda k: counts[k])
        lang_grid[y, x] = dominant_lang
        
        # Get total wealth for dominant language in cell
        wealth_grid[y, x] = data[f"{dominant_lang}_wealth"]

    # Create colormap
    lang_codes = {'Dutch': 0, 'French': 1, 'German': 2, 'English': 3}
    lang_num_grid = np.vectorize(lang_codes.get)(lang_grid)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create custom colormap for languages
    lang_cmap = LinearSegmentedColormap.from_list('lang_cmap', 
                                                ['blue', 'red', 'green', 'black'])
    
    # Plot language dominance
    lang_plot = ax.imshow(lang_num_grid, cmap=lang_cmap, 
                         vmin=0, vmax=3, alpha=0.7)
    
    # Plot wealth as color intensity
    wealth_norm = plt.Normalize(vmin=wealth_grid.min(), vmax=wealth_grid.max())
    wealth_plot = ax.imshow(wealth_grid, cmap='YlOrRd', 
                           alpha=0.5, norm=wealth_norm)
    
    # Add colorbars
    cbar_lang = fig.colorbar(lang_plot, ax=ax, ticks=[0.375, 1.125, 1.875, 2.625])
    cbar_lang.ax.set_yticklabels(['Dutch', 'French', 'German', 'English'])
    cbar_lang.set_label('Dominant Language')
    
    cbar_wealth = fig.colorbar(wealth_plot, ax=ax)
    cbar_wealth.set_label('Total Wealth of Dominant Language')
    
    # Add dividing line for Flanders/Wallonia
    ax.axhline(y=model.grid.width//2 - 0.5, color='black', 
               linestyle='--', linewidth=2)
    
    ax.set_title(f'Dominant Language by Cell with Wealth (Step {step_num})')
    ax.set_xlabel('Left → Right')
    ax.set_ylabel('Wallonia (South) → Flanders (North)')
    
    plt.tight_layout()
    plt.gca().invert_yaxis()
    plt.savefig(f"/home/travis/Desktop/Python/LangWealthHeatmap_Step_{step_num}.png")
    plt.close()

def plot_wealth_language_heatmap(model, step_num):
    # Create grid arrays
    grid_width = max(agent.pos[0] for agent in model.agents) + 1
    grid_height = max(agent.pos[1] for agent in model.agents) + 1
    
    # Initialize wealth and language grids
    wealth_grid = np.zeros((grid_height, grid_width))
    language_grid = np.zeros((grid_height, grid_width))  # Will store numerical codes
    
    # Language encoding (for color mapping)
    lang_codes = {'Dutch': 0, 'French': 1, 'German': 2, 'English': 3}
    
    # Populate grids
    for agent in model.agents:
        x, y = agent.pos
        wealth_grid[y, x] += agent.wealth
        language_grid[y, x] = lang_codes[agent.lang[0]]  # Use primary language
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Wealth heatmap
    wealth_plot = ax1.imshow(wealth_grid, cmap='YlOrRd', interpolation='nearest')
    fig.colorbar(wealth_plot, ax=ax1, label='Total Wealth')
    ax1.set_title('Wealth Distribution')
    ax1.set_xlabel('Flanders (Left) → Wallonia (Right)')
    ax1.set_ylabel('North → South')
    
    # Language region heatmap
    custom_cmap = LinearSegmentedColormap.from_list('lang_cmap', 
                                                  ['blue', 'red', 'green', 'purple'])
    lang_plot = ax2.imshow(language_grid, cmap=custom_cmap, 
                          vmin=0, vmax=3, interpolation='nearest')
    
    # Language colorbar
    cbar = fig.colorbar(lang_plot, ax=ax2, ticks=[0.375, 1.125, 1.875, 2.625])
    cbar.ax.set_yticklabels(['Dutch', 'French', 'German', 'English'])
    ax2.set_title('Language Distribution')
    
    # Add dividing line for Flanders/Wallonia
    for ax in (ax1, ax2):
        ax.axvline(x=grid_width//2 - 0.5, color='black', linestyle='--', linewidth=2)
    
    plt.tight_layout()
    plt.savefig("/home/travis/Desktop/Python/HeatMap_Step_{}.png".format(step_num))
    plt.close()

def lang_plot(model, step_numb):
    agent_counts = np.zeros((model.grid.width, model.grid.height))

    for cell in model.grid.all_cells:
        agent_counts[cell.coordinate] = len(cell.agents)
    #agent_counts.T
    # Plot using seaborn, with a visual size of 5x5
    g = sns.heatmap(agent_counts, cmap="viridis", annot=True, cbar=False, square=True)
    g.figure.set_size_inches(5, 5)
    g.set(title="Number of agents on each cell of the grid");
    plt.savefig("/home/travis/Desktop/Python/Agents_Step_{}.png".format(step_numb))
    plt.close()
    # After running the model...
    dutch_counts = model.count_language_per_cell("Dutch")
    french_counts = model.count_language_per_cell("French")
    english_counts = model.count_language_per_cell("English")
    german_counts = model.count_language_per_cell("German")

    # Plot Dutch
    sns.heatmap(dutch_counts, cmap="Blues", annot=True, square=True)
    plt.title("Dutch Speakers Step: {}".format(step_numb))
    plt.gca().invert_yaxis()
    plt.savefig("/home/travis/Desktop/Python/DutchSpeakers_Step_{}.png".format(step_numb))
    plt.close()

    # Plot French
    sns.heatmap(french_counts, cmap="Reds", annot=True, square=True)
    plt.title("French Speakers Step: {}".format(step_numb))
    plt.gca().invert_yaxis()
    plt.savefig("/home/travis/Desktop/Python/FrenchSpeakers_Step_{}.png".format(step_numb))
    plt.close()

    # Plot English
    sns.heatmap(english_counts, cmap="Greens", annot=True, square=True)
    plt.title("English Speakers Step: {}".format(step_numb))
    plt.gca().invert_yaxis()
    plt.savefig("/home/travis/Desktop/Python/EnglishSpeakers_Step_{}.png".format(step_numb))
    plt.close()

    # Plot German
    sns.heatmap(german_counts, cmap="Purples", annot=True, square=True)
    plt.title("German Speakers Step: {}".format(step_numb))
    plt.gca().invert_yaxis()
    plt.savefig("/home/travis/Desktop/Python/GermanSpeakers_Step_{}.png".format(step_numb))
    plt.close()

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
        
        boundary = (grid_height / 2) - 1
        
        if self.y <= boundary:
            self.wealth = 100
            self.native = "French"
            other_langs = choice(list(lang_dict_wallonia.keys()), random.randint(1, 2), p=list(lang_dict_wallonia.values()), replace=False).tolist()
            total_langs = [self.native]
            total_langs.extend(other_langs)
            self.lang = total_langs
        else:
            self.wealth = 150
            self.native = "Dutch"
            other_langs = choice(list(lang_dict_flanders.keys()), random.randint(1, 3), p=list(lang_dict_flanders.values()), replace=False).tolist()
            total_langs = [self.native]
            total_langs.extend(other_langs)
            self.lang = total_langs
        
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
                if self.native == other_agent.native:
                    exchange_int = random.randint(10, 25)
                    other_agent.wealth += exchange_int
                    self.wealth -= exchange_int
                else:
                    exchange_int_2 = random.randint(1, 5)
                    other_agent.wealth += exchange_int_2
                    self.wealth -= exchange_int_2
                    
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
step_number = 0
for _ in range(1):  # Runs the model for 1 step;
    model.step()
    step_number += 1
    
plot_wealth_language_heatmap(model, step_number)
lang_plot(model, step_number)
plot_language_wealth_heatmap(model, step_number)

for _ in range(9):  # Runs the model for 9 steps; Total = 10
    model.step()
    step_number += 1
    
plot_wealth_language_heatmap(model, step_number)
lang_plot(model, step_number)
plot_language_wealth_heatmap(model, step_number)
  
for _ in range(90):  # Runs the model for 90 steps; Total = 100
    model.step()
    step_number += 1
    
plot_wealth_language_heatmap(model, step_number)
lang_plot(model, step_number)
plot_language_wealth_heatmap(model, step_number)

for _ in range(900):  # Runs the model for 900 steps; Total = 1000
    model.step()
    step_number += 1
    
plot_wealth_language_heatmap(model, step_number)
lang_plot(model, step_number)
plot_language_wealth_heatmap(model, step_number)