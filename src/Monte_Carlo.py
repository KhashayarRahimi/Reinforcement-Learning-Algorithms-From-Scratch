import numpy as np
import random
from tabulate import tabulate
from tqdm import tqdm
import ast
class MonteCarlo:
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
    
    
    
    def state_reward(self, policy,state):

        policy_state = policy[0]
        
        next_state = policy_state[str(state)]

        if next_state in environment[4]:

            r = -3
        
        elif next_state == environment[3]:

            r = 100
        
        elif next_state not in environment[6]:

            r = -2
        
        else:

            r = -1
        
        return r

    #Note that here we want to evaluate just a fixed policy
    # and so we are not trying to optimize it 
    def monte_carlo_prediction(self, num_trials, policy, gamma, state_prob_type):

        #V = np.zeros((environment[6],1))

        #store returns of each trajectory
        Returns = {} #np.zeros((environment[6],1))
        #Lens = []
        #Loop for ever (for each episode)
        for trial in tqdm(range(num_trials)):
            
            #generate an episode
            trajectory = self.generate_trajectory(policy,trial, state_prob_type)
            #Lens.append(trajectory)

            #limit the lenght of trajectory

            #total reward
            G = 0

            trajectory.reverse()
            
            
            returns = {}

            for state in environment[6]:
                
                if state not in environment[4] and state != environment[3]:

                    returns[str(state)] = 0

            first_visit = []
            for step in trajectory[1:]:

                if step not in first_visit:

                    first_visit.append(step)

                    r = self.state_reward(policy,step)

                    G = gamma * G + r

                    returns[str(step)] = returns[str(step)] + G
            
            #Returns[trial] = returns
        
        V = {}
        for step in list(returns.keys()):

            V[step] = returns[step]/num_trials
        

        return V,returns
    def state_action_reward(self, policy,state):

        policy_action = policy[str(state)]
        next_state = [x + y for x, y in zip(state, policy_action)]
        

        if next_state in environment[4]:

            r = -3
        
        elif next_state == environment[3]:

            r = 100
        
        elif next_state not in environment[6]:

            r = -2
        
        else:

            r = -1
        
        return r


    def generate_trajectory_probability_based(self, policy,randomness,epsilon,traj_len,action_prob_type):

        probs = self.probability_distribution(environment[0]*environment[1],action_prob_type)
    
        start = environment[2]
        terminate = start
        trajectory = [start]
        c = 0
        test = []
        while terminate != environment[3]:
            random.seed(randomness+c)
            Actions = [[1, 0],[-1, 0],[0, 1],[0, -1]]

            #we have two probabilities for epsilon-greedy action selection
            #It's a kind of exploration-exploitation balancing
            
            #probability for exploration on not best action values
            low_prob = epsilon/len(Actions)
            high_prob = 1 - epsilon #+ (epsilon/len(Actions))

            #this random action selection is for balancing exploration-exploitation trade-off

            exex_probs = [low_prob,low_prob,low_prob,high_prob]
            if type(policy) == tuple:
                policy = policy[1]
            
            best_action_value = policy[str(terminate)]
            #print(type(best_action_value))
            Actions_copy = Actions.copy()
            #print(Actions_copy)
            Actions_copy.remove(best_action_value)
            exex_actions = Actions_copy + [best_action_value]
            #print(exex_actions)
            #print(exex_probs)
            
            action = random.choices(exex_actions, exex_probs)[0]

            #second part of action selection
            Actions.remove(action)
            sorted_actions = Actions + [action]
            state_indice = state_indice_dict[str(terminate)]
            actions_prob = probs[state_indice]
            actions_prob.sort()

            #print(sorted_actions)
            #print(actions_prob)
            #this random action selection is due to the randomness of the environment
            selected_action = random.choices(sorted_actions, actions_prob)[0]
            #print(selected_action)
            #print('=====')
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

            if c >traj_len:
                break
                #c = traj_len + 1
        
        if c > traj_len:
            
            return False
            
        else:
            return trajectory
    

    def OnPolicy_MC_prediction(self, num_trials, policy, gamma, epsilon,traj_len, action_prob_type):
        
        def reverse_dictionary(dict):
            reverse_dict = {}
            for key in list(dict.keys()):
                val = dict[key]
                reverse_dict[val] = key
            return reverse_dict

        Q = {}
        for state in environment[6]:

            if state not in environment[4]:
                
                Q[str(state)] = {}

                for action in ["[1, 0]","[-1, 0]","[0, 1]","[0, -1]"]:

                    #next_state = [x + y for x, y in zip(state, ast.literal_eval(action))]

                    #if (next_state in environment[6]) and next_state not in environment[4]:
                        
                    Q[str(state)][action] = random.uniform(1e-9, 1e-8)
        
        counter = {}
        for state in environment[6]:

            if state not in environment[4]:
                
                counter[str(state)] = {}

                for action in ["[1, 0]","[-1, 0]","[0, 1]","[0, -1]"]:

                    #next_state = [x + y for x, y in zip(state, ast.literal_eval(action))]

                    #if (next_state in environment[6]) and next_state not in environment[4]:
                        
                    counter[str(state)][action] = random.uniform(1e-9, 1e-8)
        
        done_trials = 0
        Policies = [policy]
        cp = 0
        for trial in tqdm(range(1,num_trials)):
            #print(policy['[3,3]'])
            policy = Policies[cp]
            trajectory = self.generate_trajectory_probability_based(policy, trial, epsilon,traj_len, action_prob_type)
            #print(len(trajectory))

            #if len(trajectory) < 100:
            #print(trajectory)
            
            if trajectory:
                #print(len(trajectory))

                done_trials +=1 
            

                G = 0
                returns = {}
                first_visit = []

                for state in environment[6]:

                    if state not in environment[4]:# and state != environment[3]:

                        returns[str(state)] = {}

                for state in environment[6]:
                    
                    if state not in environment[4]:# and state != environment[3]:

                        for action in ["[1, 0]","[-1, 0]","[0, 1]","[0, -1]"]:

                            #next_state = [x + y for x, y in zip(state, ast.literal_eval(action))]

                            #if (next_state in environment[6]) and next_state not in environment[4]:

                            returns[str(state)][action] = random.uniform(1e-9, 1e-8)

                    
                #print(returns)

                for i in range(len(trajectory[1:])):
                    step = trajectory[1:][i]

                    if step not in first_visit:
                        
                        """state_str = str(step)
                        if state_str not in returns:
                            returns[state_str] = {}
                            for action in ["[1,0]", "[-1,0]", "[0,1]", "[0,-1]"]:
                                returns[state_str][action] = 0"""  # Initialize all actions with value 0
                                
                        first_visit.append(step)
                        #action = derive_action(trajectory[1:][i + 1], trajectory[1:][i])
                        last_step = str(trajectory[1:][i])
                        if type(policy) == tuple:
                            policy = policy[1]
                            
                        action = policy[last_step]
                        #if action == [0,0]:
                        r = state_action_reward(policy, step)
                        G = gamma * G + r
                        #print(G)
                        #action_str = str(action)
                        #print(action_str)
                        #print(returns[str(step)])
                        returns[str(step)][str(action)] += G
                        #print(returns[str(step)][action_str])

                


                for state in list(returns.keys()):
                    for action in ["[1, 0]","[-1, 0]","[0, 1]","[0, -1]"]:
                        #print(state,action)
                        #print('q',Q[state][action])
                        #print(returns[state][action])
                        #if returns[state]["[-1, 0]"] != 0:
                        #    print(returns[state]["[-1, 0]"])
                        #Q[state][action] = returns[state][action] / trial

                        if abs(returns[state][action]) > 1e-3:

                            counter[state][action] = counter[state][action] + 1

                            Q[state][action] = Q[state][action] + returns[state][action]

                            Q[state][action] = Q[state][action] / round(counter[state][action])
                            #print('f')
                        
                        #else:

                        #    Q[state][action] = Q[state][action] + returns[state][action]

                policy = {}
                for state in list(Q.keys()):
                    #print('d')
                    if Q[state] != {}:
                        value_action_state = reverse_dictionary(Q[state])
                        #print('value_action_state:',value_action_state)
                        #print(state)
                        #print(value_action_state)
                        Max_val = max(list(value_action_state.keys()))
                        best_action = value_action_state[Max_val]
                        policy[state] = ast.literal_eval(best_action)
                #print(policy)
                #if policy != policy_0:
                #    print('f')

                Policies.append(policy)
                cp = cp + 1
                if cp == 100:
                    print(cp)
        
            
        return policy, Q, done_trials, Policies
      
if __name__ == "__main__":
    MC = MonteCarlo()

    # Generate the environment using your generate_grid_world function
    environment = MC.generate_grid_world(5, 4,4,4,39)

    state_indice_dict = {}
    counter = 0
    for state in environment[6]:

        state = str(state)
        state_indice_dict[state] = counter
        counter = counter + 1

    policy_0 = MC.arbitrary_policy(41)

    # Obtain the optimal value and policy using policy_iteration
    value_functions, returns = MC.monte_carlo_prediction(1000,policy_0,0.9,'stochastic')

    # Print the results
    print("Value Functions:")
    print(value_functions)
    print("Returns:")
    print(returns)