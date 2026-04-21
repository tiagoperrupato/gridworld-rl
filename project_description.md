# Project 2: Reinforcement Learning (RL)
**Course:** Artificial Intelligence
**Professor:** Anderson Rocha
**Deadline:** April 30, 2026

---

## 1. General Objective
Develop an intelligent agent capable of learning to make sequential decisions in an unknown environment using:
* Formal modeling of the problem as an MDP;
* Implementation of Bellman equations;
* Implementation of Q-learning;
* Exploration strategies;
* Experimental evaluation of the learning process.

---

## 2. Selected Theme: Gridworld with Hazards
A two-dimensional grid environment where the agent must navigate to a target.

### Problem Characteristics:
* **States:** Positions within the grid;
* **Actions:** Move (up, down, left, right);
* **Rewards:**
    * **Goal:** Positive reward;
    * **Traps:** Negative reward;
    * **Movement:** Small negative cost;
* **Dynamics:** Potential for stochastic transitions (e.g., wind);
* **Terminal States:** Success or failure.
* **Challenge:** Learning the optimal policy in an environment with sparse rewards and uncertainty.

---

## 3. Implementation Requirements
**Important Note:** The implementation of core algorithms (Bellman Equations, Value Iteration/Policy Evaluation, and Q-learning) must be the group's original work. While auxiliary libraries for visualization and data structures are allowed, the central logic must be implemented by the team.

### Mandatory Algorithms:
1.  **Bellman Equations (Planning):**
    $$V(s) \leftarrow \max_{a} \sum_{s'} T(s,a,s') [R(s,a,s') + \gamma V(s')]$$
    * Implementation of Value Iteration or Policy Evaluation.
2.  **Q-learning (Learning):**
    $$Q(s,a) \leftarrow Q(s,a) + \alpha [r + \gamma \max_{a'} Q(s',a') - Q(s,a)]$$
    * Use of exploration strategies such as $\epsilon$-greedy.

---

## 4. Experimental Protocol and Deliverables
### Mandatory Protocol:
* Fixed number of training episodes;
* Maximum horizon per episode;
* Standardized initialization;
* Mandatory data instrumentation.

### Deliverables:
1. **Technical Report:** PDF format, between 5 and 6 pages;
2. **Presentation:** Oral or video (30 minutes);
3. **Practical Demonstration:** The agent in operation;
4. **Required Visualizations:** $V(s)$ value maps, learned policy, agent trajectories, and learning curves.

---

## 5. Evaluation Criteria
* **Agent Technical Quality:** 30%
* **Report Clarity and Depth:** 30%
* **Presentation and Demonstration:** 20%
* **Experimental Analysis:** 20%

---
**Group Formation:** Groups of four or five members.