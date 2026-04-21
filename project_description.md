# Reinforcement Learning (RL) (T2)
**Prof. Anderson Rocha**
**Deadline: April 30, 2026**

---

### Important Observation
The implementation of the algorithms (Bellman Equations, Value Iteration/Policy Evaluation, and Q-learning) must be the group's own authorship. Auxiliary libraries are permitted for visualization, structures of data, and simulation, but the central algorithms must be implemented by the teams.

### General Objective
Develop an intelligent agent capable of learning to make sequential decisions in an unknown environment, utilizing:
* Formal modeling of the problem as an MDP;
* Implementation of Bellman equations;
* Implementation of Q-learning;
* Exploration strategies;
* Experimental evaluation of the learning process.

### Experimental Protocol (Mandatory)
* Fixed number of training episodes;
* Maximum horizon per episode;
* Standardized initialization;
* Mandatory instrumentation.

### Group Formation
* Groups of four or five people.

---

### Problem Description: Gridworld with Hazards
Two-dimensional grid environment in which the agent must navigate to an objective.

**Characteristics:**
* **States:** positions in the grid;
* **Actions:** move (up, down, left, right);
* **Rewards:**
    * Objective: positive reward;
    * Traps: negative reward;
    * Movement: small negative cost;
* **Transitions:** Possibility of stochastic transitions (e.g., wind);
* **Terminal States:** Success or failure.

**Challenge:** Learn the optimal policy in an environment with sparse rewards and possible uncertainties.

---

### Deliverables
1. Technical report of up to 5/6 pages (PDF);
2. Oral presentation or video (30 minutes);
3. Practical demonstration of the agent in operation;
4. Mandatory visualizations of the results.

---

### Expected Structure of the Report

**1. Problem Modeling**
* Formal definition of the MDP (S, A, T, R);
* State representation;
* Definition of actions;
* Environment dynamics;
* Reward function.

**2. Bellman Equations**
* Implementation of the value update:
  $$V(s) \leftarrow \max_{a} \sum_{s'} T(s,a,s') [R(s,a,s') + \gamma V(s')]$$
* Description of the algorithm (Value Iteration or Policy Evaluation);
* Convergence criteria;
* Analysis of the behavior of the value throughout the iterations.

**3. Q-learning**
* Implementation of the update:
  $$Q(s,a) \leftarrow Q(s,a) + \alpha [r + \gamma \max_{a'} Q(s',a') - Q(s,a)]$$
* Exploration strategy (e.g., e-greedy);
* Initialization of values;
* Impact of $\alpha$, $\gamma$, and $\epsilon$.

**4. Experimental Evaluation**
* Evolution of the accumulated reward per episode;
* Comparison between different values of $\gamma$;
* Comparison between exploration strategies;
* Convergence time;
* Stability of the learned policy;
* Comparison between Bellman (planning) and Q-learning (learning).

**5. Visualizations**
Must obligatorily include:
* Value map $V(s)$;
* Learned policy;
* Agent trajectories;
* Learning curves.

**6. Discussion**
* Difficulties encountered;
* Observed trade-offs;
* Impact of exploration;
* Model limitations;
* Relationship between theory and practice.

---

### Evaluation Criteria
* Technical quality of the agent: 30%
* Clarity and depth of the report: 30%
* Presentation and demonstration: 20%
* Experimental analysis: 20%

### Extras (Optional)
* Double Q-learning;
* $\epsilon$ decay;
* Stochastic environments;
* Comparison with known optimal policy;
* Extension to larger spaces.

---

### Final Message
Unlike TP1, where the agent plans before acting, here the agent learns by acting. The objective is to understand how sequential decisions can emerge from the interaction with the environment.