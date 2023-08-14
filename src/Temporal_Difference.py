import numpy as np
import random
from tabulate import tabulate
from tqdm import tqdm
import ast

class TemporalDifference:
    def __init__(self):
        pass

    def generate_grid_world(self, length, width,path_lenght,holes_number,Random_State):
    
        random.seed(Random_State)
        #store all cells in a list
        Grid_Cells = []
        for row in range(length):
            for col in range(width):
                Grid_Cells.append([row,col])


        #specify the number of holes in the gridworld
        
        #specify the start point as a random cell
        start = [random.randint(0, length), random.randint(0, width)]

        #create a path from start point
        """instead of defining start and goal points,
        we define just a start point and a random path with a random lenght to
        another point and name it as goal point"""
        
        def random_path(Start, Path_Lenght,length, width):
            
            Path = []
            Path.append(Start)
            for i in range(Path_Lenght):
                
                #there are two moves that take us on a random cell named Goal [1,0], [0,1]
                
                move = random.choice([[1,0], [0,1]])
                
                #update the start cell/point by the above move
                Start = [x + y for x, y in zip(Start, move)]
                
                #if the movement take us out of our gridworld, we reverse the change in the start point
                if Start[0] < 0 or Start[1] < 0 or Start[0] > length-1 or Start[1] > width-1:

                    Start = [x - y for x, y in zip(Start, move)]

                else:
                    
                    #create a path history
                    Path.append(Start)

            Goal = Start

            return Goal,Path
        

        GoalPath = random_path(start, path_lenght,length, width)

        goal = GoalPath[0]
        path = GoalPath[1]

        #now we must eliminate the path cells from the Grid_Cells to choose hole cells from remaining cells

        FreeCells = [x for x in Grid_Cells if x not in path]

        Holes = random.sample(FreeCells, holes_number)

        #Also, we can visualize our gridworld in a simple way

        def mark_holes(holes):
            marked_data = [["Hole" if [row, col] in holes else [row, col] for col in range(width)] for row in range(length)]
            return marked_data
        
        marked_matrix = mark_holes(Holes)

        print(tabulate(marked_matrix, tablefmt="grid"))

        
        return length, width, start, goal, Holes, path,Grid_Cells
    
    def generate_probabilities(self, n):
            numbers = [random.random() for _ in range(n)]
            total_sum = sum(numbers)
            scaled_numbers = [num / total_sum for num in numbers]
            return scaled_numbers
    
    def probability_distribution(self, grid_size,randomness):
        random.seed(41)
        
        #by this function we generate probabilities which their sum is equal to 1
        
        
        cells_prob = {}
        if randomness == 'stochastic':
            for cell in range(grid_size):
                
                #we set the number of probs to 4 due to 4 possible action for each cell (go to its neighbors)
                probs = self.generate_probabilities(4)

                cells_prob[cell] = probs
        elif randomness == 'equal probable':

            for cell in range(grid_size):

                cells_prob[cell] = [0.25,0.25,0.25,0.25]
        
        elif randomness == 'deterministic':
            for cell in range(grid_size):

                cells_prob[cell] = [0,0,0,1]#[0.15,.15,0.1,0.6]


        #Note that we consider the correspondence between probabilities and actions as below:
        #probs = [p1, p2, p3, p4] ---> [[1,0],[-1,0],[0,1],[0,-1]]

        return cells_prob
    
    #this function specify the 4 neighbors cells around one arbitrary cell in the gridworld
    #we need this function to prevent redundant computstion in the Bellman equation
    def neighbor_cells(self, cell):

        grid_cells = environment[6]
        Actions = [[1,0],[-1,0],[0,1],[0,-1]]

        Neighbors = []

        for action in Actions:

            neighbor = [x + y for x, y in zip(cell, action)]
            #if neighbor not in environment[4]:
            Neighbors.append(neighbor)

        return Neighbors
    
    def arbitrary_policy(self, randomness):
        #random.seed(randomness)
        
        policy = {}
        policy_action = {}
        for state in environment[6]:

            if state not in environment[4]:

                neighbors = self.neighbor_cells(state)[0]
                Actions_Neighbors = self.neighbor_cells(state)[1]

                allowed_positions = []

                for neighbor in neighbors:
                    
                    if neighbor in environment[6] and neighbor not in environment[4]:
                        
                        allowed_positions.append(neighbor)
                
                if len(allowed_positions) > 0:
                    
                    next_state = random.choice(allowed_positions)
                    row = next_state[0] - state[0]
                    col = next_state[1] - state[1]
                    PolicyAction = [row, col]

                    policy['{}'.format(state)] = next_state
                    policy_action['{}'.format(state)] = PolicyAction



        return policy, policy_action
    def generate_trajectory(self, policy,randomnumber,state_prob_type):

        policy_action = policy[1]

        probs = self.probability_distribution(environment[0]*environment[1],state_prob_type)
        
        start = environment[2]

        terminate = start

        trajectory = [start]
        c = 0
        test = []
        while terminate != environment[3]:
            random.seed(randomnumber+c)
            Actions = [[1,0],[-1,0],[0,1],[0,-1]]

            action = policy_action[str(terminate)]
            Actions.remove(action)
            #sorted_actions = [action]
            sorted_actions = Actions + [action]
            #print(sorted_actions)
            state_indice = state_indice_dict[str(terminate)]
            actions_prob = probs[state_indice]
            actions_prob.sort()
            #print(actions_prob)
            #print(actions_prob)


            selected_action = random.choices(sorted_actions, actions_prob)[0]
            
            next_state = [x + y for x, y in zip(terminate, selected_action)]
            
            #if the agent goes out of the gridworld, it stays in its current state
            if next_state not in environment[6]:

                next_state = terminate
            
            #if it drops into the holes, it goes to the start points
            elif next_state in environment[4]:

                next_state = start

            
            terminate = next_state

            trajectory.append(terminate)
            c = c+1

        return trajectory
    

    def TD_zero(self, num_trials, policy, alpha, gamma,environment_stochasticity):

        policy_state = policy[0]
        policy_action = policy[1]

        grid_size = environment[0]*environment[1]
        
        V = {}
        for state in environment[6]:
        
            if state not in environment[4] and state != environment[3]:

                V[str(state)] = 0
        
        for trial in tqdm(range(num_trials)):

            trajectory = self.generate_trajectory(policy,trial,environment_stochasticity)
            
            #state start from start point to terminal point
            for state in trajectory[:-1]:
                
                #s_prime in the algorithm pseudocode
                next_state = policy_state[str(state)]

                if next_state == environment[3]:
                    
                    V[str(next_state)] = 0

                reward = self.state_reward(policy,state)

                V[str(state)] = V[str(state)] +\
                    alpha * (reward + gamma * V[str(next_state)] - V[str(state)])
        
    
        return V


    def state_reward_policy_free(self, state, Final_action):

        next_state = [x + y for x, y in zip(state, Final_action)]

        if next_state in environment[4]:
            r = -3
        elif next_state == environment[3]:
            r = 100
        elif next_state not in environment[6]:
            r = -2
        else:
            r = -1 
        return r

    def argmax_policy(self,q_values):

        policy = {}
        for state in list(q_values.keys()):

            value_action_state = self.reverse_dictionary(q_values[state])
            Max_val = max(list(value_action_state.keys()))
            best_action = value_action_state[Max_val]
            policy[state] = ast.literal_eval(best_action)
        
        return policy

    def reverse_dictionary(self,dict):
        reverse_dict = {}
        for key in list(dict.keys()):
            val = dict[key]
            reverse_dict[val] = key
        return reverse_dict


    def sarsa(self, num_trials, alpha, gamma,environment_stochasticity, epsilon):

        grid_size = environment[0]*environment[1]

        probs = self.probability_distribution(grid_size,environment_stochasticity)
        
        Q = {}
        for state in environment[6]:

            if state not in environment[4]:
                
                Q[str(state)] = {}

                for action in ["[1, 0]","[-1, 0]","[0, 1]","[0, -1]"]:

                    #next_state = [x + y for x, y in zip(state, ast.literal_eval(action))]

                    #if (next_state in environment[6]) and next_state not in environment[4]:
                        
                    Q[str(state)][action] = random.uniform(1e-9, 1e-8)

        def state_action_nextstate(current_state):

            #probs = probability_distribution(grid_size,42)

            if type(current_state) == str:

                state = ast.literal_eval(current_state)
            else:
                state = current_state
            #Choose action using policy derived from Q===================================
            value_action_state = self.reverse_dictionary(Q[str(state)])
            Max_val = max(list(value_action_state.keys()))
            best_action = value_action_state[Max_val]
            best_action = ast.literal_eval(best_action)

            #============================================================================
            #Epsilon Greedy
            if random.uniform(0, 1) > epsilon:

                selected_action = best_action
            
            else:
                Actions = [[1,0],[-1,0],[0,1],[0,-1]]
                Actions.remove(best_action)
                epsilon_action = random.choice(Actions)

                selected_action = epsilon_action
            #============================================================================
            
            Actions = [[1,0],[-1,0],[0,1],[0,-1]]
            Actions.remove(selected_action)
            sorted_actions = Actions + [selected_action]
            state_indice = state_indice_dict[str(state)]
            actions_prob = probs[state_indice]
            actions_prob.sort()
            #due to stochasticity of the environment
            Final_action = random.choices(sorted_actions, actions_prob)[0]
            #print(type(state), type(Final_action))
            
            next_state = [x + y for x, y in zip(state, Final_action)]

            if next_state not in environment[6] or next_state in environment[4]:

                next_state = current_state

            return Final_action, next_state
        

        
        for trial in tqdm(range(num_trials)):

            next_state = environment[2]

            while next_state != environment[3]:
                
                state = next_state
                
                action_nextstate = state_action_nextstate(state)

                action = action_nextstate[0]
                next_state = action_nextstate[1]

                next_action = state_action_nextstate(next_state)[0]


                if next_state == environment[3]:

                    for action in [[1,0],[-1,0],[0,1],[0,-1]]:
                    
                        Q[str(next_state)][str(action)] = 0

                reward = state_reward_policy_free(state, action)

                Q[str(state)][str(action)] = Q[str(state)][str(action)] +\
                    alpha * (reward + gamma * Q[str(next_state)][str(next_action)] - Q[str(state)][str(action)])
            
        
        return Q


    def Q_learning(self,num_trials, alpha, gamma,environment_stochasticity, epsilon):

        grid_size = environment[0]*environment[1]

        probs = self.probability_distribution(grid_size,environment_stochasticity)

        Q = {}
        for state in environment[6]:

            if state not in environment[4]:
                
                Q[str(state)] = {}

                for action in ["[1, 0]","[-1, 0]","[0, 1]","[0, -1]"]:

                    Q[str(state)][action] = random.uniform(1e-9, 1e-8)


        def state_action_nextstate(current_state):

            #probs = probability_distribution(grid_size,42)

            if type(current_state) == str:

                state = ast.literal_eval(current_state)
            else:
                state = current_state
            #Choose action using policy derived from Q===================================
            value_action_state = self.reverse_dictionary(Q[str(state)])
            Max_val = max(list(value_action_state.keys()))
            best_action = value_action_state[Max_val]
            best_action = ast.literal_eval(best_action)

            #============================================================================
            #Epsilon Greedy
            if random.uniform(0, 1) > epsilon:

                selected_action = best_action
            
            else:
                Actions = [[1,0],[-1,0],[0,1],[0,-1]]
                Actions.remove(best_action)
                epsilon_action = random.choice(Actions)

                selected_action = epsilon_action
            #============================================================================
            
            Actions = [[1,0],[-1,0],[0,1],[0,-1]]
            Actions.remove(selected_action)
            sorted_actions = Actions + [selected_action]
            state_indice = state_indice_dict[str(state)]
            actions_prob = probs[state_indice]
            actions_prob.sort()
            #due to stochasticity of the environment
            Final_action = random.choices(sorted_actions, actions_prob)[0]
            #print(type(state), type(Final_action))
            
            next_state = [x + y for x, y in zip(state, Final_action)]

            if next_state not in environment[6] or next_state in environment[4]:

                next_state = current_state
            
            value_action_state = self.reverse_dictionary(Q[str(next_state)])
            #max Q(s',s)
            Max_q_val = max(list(value_action_state.keys()))
            best_action = value_action_state[Max_q_val]
            best_action = ast.literal_eval(best_action)

            return Final_action, next_state, Max_q_val
        
        policy = {}
        path = [environment[2]]
        
        for trial in tqdm(range(num_trials)):

            next_state = environment[2] #sorry for bad names

            while next_state != environment[3]:
                
                state = next_state
                
                action_nextstate = state_action_nextstate(state)

                action = action_nextstate[0]
                next_state = action_nextstate[1]

                next_action = state_action_nextstate(next_state)[0]


                if next_state == environment[3]:

                    for action in [[1,0],[-1,0],[0,1],[0,-1]]:
                    
                        Q[str(next_state)][str(action)] = 0

                reward = self.state_reward_policy_free(state, action)

                Max_q_val = action_nextstate[2]

                Q[str(state)][str(action)] = Q[str(state)][str(action)] +\
                    alpha * (reward + gamma * Max_q_val - Q[str(state)][str(action)])
                    
                
                if trial  == num_trials - 1: #the last trial

                    policy[str(state)] = [action, next_state]
                    path.append(next_state)

            
        
        return Q, policy, path



    def double_Q_learning(self,num_trials, alpha, gamma,environment_stochasticity, epsilon):

        grid_size = environment[0]*environment[1]

        probs = self.probability_distribution(grid_size,environment_stochasticity)

        Q1, Q2 = {}, {}
        for state in environment[6]:

            if state not in environment[4]:
                
                Q1[str(state)] = {}
                Q2[str(state)] = {}

                for action in ["[1, 0]","[-1, 0]","[0, 1]","[0, -1]"]:

                    Q1[str(state)][action] = random.uniform(1e-9, 1e-8)
                    Q2[str(state)][action] = random.uniform(1e-9, 1e-8)

        #Choose A from S using the policy epsilon-greedy in Q1 + Q2
        def state_action_nextstate(current_state,Q1, Q2):

            Q = {}

            for state in environment[6]:

                if state not in environment[4]:
                    
                    Q[str(state)] = {}
                    for action in ["[1, 0]","[-1, 0]","[0, 1]","[0, -1]"]:

                        Q[str(state)][action] = Q1[str(state)][action] + Q2[str(state)][action]

            #probs = probability_distribution(grid_size,42)

            if type(current_state) == str:

                state = ast.literal_eval(current_state)
            else:
                state = current_state
            #Choose action using policy derived from Q===================================
            value_action_state = self.reverse_dictionary(Q[str(state)])
            Max_val = max(list(value_action_state.keys()))
            best_action = value_action_state[Max_val]
            best_action = ast.literal_eval(best_action)

            #============================================================================
            #Epsilon Greedy
            if random.uniform(0, 1) > epsilon:

                selected_action = best_action
            
            else:
                Actions = [[1,0],[-1,0],[0,1],[0,-1]]
                Actions.remove(best_action)
                epsilon_action = random.choice(Actions)

                selected_action = epsilon_action
            #============================================================================
            
            Actions = [[1,0],[-1,0],[0,1],[0,-1]]
            Actions.remove(selected_action)
            sorted_actions = Actions + [selected_action]
            state_indice = state_indice_dict[str(state)]
            actions_prob = probs[state_indice]
            actions_prob.sort()
            #due to stochasticity of the environment
            Final_action = random.choices(sorted_actions, actions_prob)[0]
            #print(type(state), type(Final_action))
            
            next_state = [x + y for x, y in zip(state, Final_action)]

            if next_state not in environment[6] or next_state in environment[4]:

                next_state = current_state
            
            value_action_state = self.reverse_dictionary(Q[str(next_state)])
            #max Q(s',s)
            Max_q_val = max(list(value_action_state.keys()))
            next_best_action = value_action_state[Max_q_val]
            next_best_action = ast.literal_eval(next_best_action)

            return Final_action, next_state, Max_q_val, next_best_action
        
        policy = {}
        path = [environment[2]]
        
        for trial in tqdm(range(num_trials)):

            next_state = environment[2] #sorry for bad names

            while next_state != environment[3]:
                
                state = next_state
                
                action_nextstate = state_action_nextstate(state, Q1, Q2)

                action = action_nextstate[0]
                next_state = action_nextstate[1]
                next_best_action =  action_nextstate[3]

                if next_state == environment[3]:

                    for action in [[1,0],[-1,0],[0,1],[0,-1]]:
                    
                        Q1[str(next_state)][str(action)] = 0
                        Q2[str(next_state)][str(action)] = 0

                reward = self.state_reward_policy_free(state, action)


                if random.random() >= 0.5:

                    Q1[str(state)][str(action)] = Q1[str(state)][str(action)] +\
                        alpha * (reward + gamma * Q2[str(next_state)][str(next_best_action)] - Q1[str(state)][str(action)])
                
                else:

                    Q2[str(state)][str(action)] = Q2[str(state)][str(action)] +\
                        alpha * (reward + gamma * Q1[str(next_state)][str(next_best_action)] - Q2[str(state)][str(action)])

                
                if trial  == num_trials - 1: #the last trial

                    policy[str(state)] = [action, next_state]
                    path.append(next_state)

            
        
        return Q1,Q2, policy, path
