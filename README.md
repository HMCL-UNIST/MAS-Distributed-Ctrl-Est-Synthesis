# MAS-Distributed-Ctrl-Est-Synthesis
This repository contains the simulation source code for distributed control and estimation synthesis. It implements the proposed method alongside comparison baseline methods as described in our paper.

<p align="center">
  <img src="https://github.com/HMCL-UNIST/MAS-Distributed-Ctrl-Est-Synthesis/assets/32535170/86fd1f66-2679-436f-aff7-ee7b81753d65" alt="Fig1_v2" style="width:45%;"/>
  <img src="https://github.com/HMCL-UNIST/MAS-Distributed-Ctrl-Est-Synthesis/assets/32535170/36d362d9-e8ef-4af5-933f-69c900dc5487" alt="Fig2_5" style="width:45%;"/>
</p>


## Overview
The project focuses on Multi-Agent Systems (MAS) and aims to provide a comprehensive framework for the synthesis of distributed control and estimation. 

## Getting Started
### Prerequisites

Ensure you have Python 3.x installed on your system. 
```
pip install seaborn matplotlib pandas scipy
```

### Installation

Clone this repository to your local machine using:

```
git clone https://github.com/HMCL-UNIST/MAS-Distributed-Ctrl-Est-Synthesis.git
```

### Running the Simulation

To run the simulation, execute the following command in the terminal:
```
python3 Comparison_MCMC.py
```

#### Configuration Options

Before running the simulation, you can configure several parameters within `Comparison_MCMC.py` to tailor the simulation to your needs:

- **Number of MCMC Tests**: Adjust the number of Monte Carlo Markov Chain (MCMC) simulations by setting the `num_simulations` variable. For example: `num_simulations = 5`.

- **Number of Agents**: Specify the number of agents for each MCMC test by defining the `num_agent_list`. For instance: `num_agent_list = [5, 10]`.

- **Gamma Values**: For the comparison baseline method, set a list of gamma values with `gamma_list`. The length of this list should correspond to the number of agents. A higher gamma value might be necessary to ensure the stability of the Multi-Agent System (MAS). Example: `gamma_list = [3, 20]`.



    
    

