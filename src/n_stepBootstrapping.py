import numpy as np
import random
from tabulate import tabulate
from tqdm import tqdm
import ast

class n_stepBootstrapping:
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
    
    def state_reward(self,next_state):

        if next_state in environment[4]:

            r = -3
        
        elif next_state == environment[3]:

            r = 100
        
        elif next_state not in environment[6]:

            r = -2
        
        else:

            r = -1
    
        return r
        
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
    
    def reverse_dictionary(self, dict):
        reverse_dict = {}
        for key in list(dict.keys()):
            val = dict[key]
            reverse_dict[val] = key
        return reverse_dict

    def state_to_indice(self, environment):
        state_indice_dict = {}
        counter = 0
        for state in environment[6]:

            state = str(state)
            state_indice_dict[state] = counter
            counter = counter + 1
        
        return state_indice_dict

    def generate_trajectory(self,policy,randomness,environment_stochasticity):

        policy_action = policy[1]
        probs = self.probability_distribution(environment[0]*environment[1],environment_stochasticity)
        start = environment[2]
        terminate = start
        trajectory = []
        pure_trajectory = [start]
        c = 0
        while terminate != environment[3]:
            random.seed(randomness+c)
            Actions = [[1,0],[-1,0],[0,1],[0,-1]]
            action = policy_action[str(terminate)]
            Actions.remove(action)
            sorted_actions = Actions + [action]
            state_indice_dict = self.state_to_indice(environment)
            state_indice = state_indice_dict[str(terminate)]
            actions_prob = probs[state_indice]
            actions_prob.sort()

            selected_action = random.choices(sorted_actions, actions_prob)[0]
            current_state = terminate
            next_state = [x + y for x, y in zip(terminate, selected_action)]
            pure_trajectory.append(next_state)
            
            #if the agent goes out of the gridworld, it stays in its current state
            if next_state not in environment[6]:
                next_state = terminate
            
            #if it drops into the holes, it goes to the start points
            elif next_state in environment[4]:
                next_state = start  

            terminate = next_state
            trajectory.append((current_state,selected_action))
            c = c+1
        
        trajectory.append((environment[3],[0,0]))
        pure_trajectory.append(environment[3])

        return trajectory,pure_trajectory


    def n_step_TD(self,num_trials, n, policy, gamma, alpha, environment_stochasticity):

        V = {}
        for state in environment[6]:
        
            if state not in environment[4]:

                V[str(state)] = 0
        
        indice_state_dict = {}
        counter = 0
        for state in environment[6]:

            #state = str(state)
            indice_state_dict[counter] = state
            counter = counter + 1
        
        state_policy = policy[0]
        action_policy = policy[1]

        for trial in tqdm(range(num_trials)):

            TRAJECTORY = self.generate_trajectory(policy,trial,environment_stochasticity)

            trajectory = TRAJECTORY[0]
            pure_trajectory = TRAJECTORY[1]
            
            T = float('inf')
            tau = 0
            t = -1
            while tau != T - 1:

                t = t + 1

                if t < T:
                    
                    #t_state = indice_state_dict[t]
                    #next_state = state_policy[str(t_state)]
                    

                    if pure_trajectory[t+1] == environment[3]:
                        T = t + 1
                
                tau = t - n + 1

                if tau >= 0:

                    G = 0

                    for i in range(tau+1, min(tau+n,T)+1):

                        r = self.state_reward(pure_trajectory[i+1])

                        G = G + (gamma ** (i-tau-1)) * r

                    if tau + n < T:

                        #print(tau , n)
                        
                        tau_n_state = trajectory[tau + n] #indice_state_dict[tau + n]

                        if tau_n_state not in environment[4] and tau_n_state != environment[3]:

                            G = G + (gamma ** n) * V[str(tau_n_state)]
                    
                    tau_state = trajectory[tau] #indice_state_dict[tau]
                    #tau
                    if tau_state not in environment[4]:
                        #print(type(tau_state))


                        V[str(tau_state)] = V[str(tau_state)] + alpha * (G - V[str(tau_state)])
            

        
        return V



    def n_step_sarsa(self,num_trials, n, gamma, alpha, environment_stochasticity,epsilon):
    
        grid_size = environment[0]*environment[1]

        probs = self.probability_distribution(grid_size,environment_stochasticity)

        Q = {}
        for state in environment[6]:

            if state not in environment[4]:
                
                Q[str(state)] = {}

                for action in ["[1, 0]","[-1, 0]","[0, 1]","[0, -1]"]:

                    Q[str(state)][action] = random.uniform(1e-9, 1e-8)
        
        def state_action_nextstate(current_state):

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
            state_indice_dict = self.state_to_indice(environment)
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


        state_t  = environment[2]

        for trial in tqdm(range(num_trials)):
            
            T = float('inf')
            tau = 0
            t = -1
            trajectory = []
            state_t  = environment[2]
            while tau != T - 1:

                t = t + 1

                if t < T:
                    trajectory.append(state_t)
                    next_state = state_action_nextstate(state_t)[1]
                    #last_state = state_t
                    state_t = next_state
                    

                    if next_state == environment[3]:
                        T = t + 1
                    
                    #else:
                    #    next_action = 

                
                tau = t - n + 1

                if tau >= 0:

                    G = 0
                    
                    state_tau = trajectory[tau]
                    state_i = state_tau
                    action_tau = state_action_nextstate(state_tau)[0]

                    action_ii = 0
                    for i in range(tau+1, min(tau+n,T)+1):

                        #actionTN = action_ii

                        state_ii = state_action_nextstate(state_i)[1]
                        action_ii = state_action_nextstate(state_i)[0]

                        #stateTN = state_i #s_(t+n) this is store for the next part
                        

                        #n_state_action.append((state_ii, action_ii))

                        #state_i = state_ii
                        r = state_reward(state_ii)

                        G = G + (gamma ** (i-tau-1)) * r

                        if i < min(tau+n,T):

                            state_i = state_ii

                    if tau + n < T:

                        stateTN = state_action_nextstate(state_ii)[1]
                        actionTN = state_action_nextstate(state_ii)[0]

                        #print(tau , n)
                        
                        #tau_n_state = trajectory[tau + n] #indice_state_dict[tau + n]

                        #if state_ii not in environment[4] and state_ii != environment[3]:

                        G = G + (gamma ** n) * Q[str(stateTN)][str(actionTN)]

                    if state_tau not in environment[4] and state_tau != environment[3]:
                        #print(type(tau_state))


                        Q[str(state_tau)][str(action_tau)] = Q[str(state_tau)][str(action_tau)] + alpha * (G - Q[str(state_tau)][str(action_tau)])
        
        del Q[str(environment[3])]
        return Q

    def generate_nonzero_probabilities(self,n,RandomSeed):
        
        Seed = RandomSeed
        status = 'Not Done'
        while status != 'Done':
            random.seed(Seed)
            numbers = [random.random() for _ in range(n)]
            total_sum = sum(numbers)
            scaled_numbers = [num / total_sum for num in numbers]

            if min(scaled_numbers) > 0:

                status = 'Done'
            
            else:
                Seed = Seed + 1
        
        return scaled_numbers

    def state_action_probs_policy(self,RandomSeed):
            
        policy = {}
        policy_action = {}

        state_action_probs = {}
        
        for state in environment[6]:

            if state not in environment[4]:
                
                state_action_probs[str(state)] = {}
                #state_probs = generate_nonzero_probabilities(4,RandomSeed + c)
                
                #prob = 0
                for action in ["[1, 0]","[-1, 0]","[0, 1]","[0, -1]"]:

                    state_action_probs[str(state)][action] = 0
                        
                    #state_action_probs[str(state)][action] = state_probs[prob]
                    ##prob = prob + 1
        
            #c =  c + 1
        
        c = 0

        for state in environment[6]:

            if state not in environment[4]:

                neighbors = neighbor_cells(state)[0]
                Actions_Neighbors = neighbor_cells(state)[1]

                allowed_positions = []

                for neighbor in neighbors:
                    
                    if neighbor in environment[6] and neighbor not in environment[4]:
                        
                        allowed_positions.append(neighbor)
            
                next_state = random.choice(allowed_positions)

                row = next_state[0] - state[0]
                col = next_state[1] - state[1]
                PolicyAction = [row, col]

                policy['{}'.format(state)] = next_state
                policy_action['{}'.format(state)] = PolicyAction
        
                state_probs = generate_nonzero_probabilities(4,RandomSeed + c)

                state_action_probs[str(state)][str(PolicyAction)] = max(state_probs)

                state_probs.remove(max(state_probs))

                Actions = [[1,0],[-1,0],[0,1],[0,-1]]

                Actions.remove(PolicyAction)

                state_action_probs[str(state)][str(Actions[0])] = state_probs[0]
                state_action_probs[str(state)][str(Actions[1])] = state_probs[1]
                state_action_probs[str(state)][str(Actions[2])] = state_probs[2]

                c = c + 1

        return policy, policy_action,state_action_probs

    def off_policy_n_step_sarsa(self, num_trials, n,  policy_b, gamma, alpha, environment_stochasticity,epsilon):
        
        grid_size = environment[0]*environment[1]

        probs = self.probability_distribution(grid_size,environment_stochasticity)

        

        policy_b_state = policy_b[0]
        policy_b_action = policy_b[1]
        policy_b_probs = policy_b[2]

        Q = {}
        for state in environment[6]:

            if state not in environment[4]:
                
                Q[str(state)] = {}

                for action in ["[1, 0]","[-1, 0]","[0, 1]","[0, -1]"]:

                    Q[str(state)][action] = random.uniform(1e-9, 1e-8)

        
        def greedy_policy_pi(current_state):

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
                selection_prob = 1 - epsilon
            
            else:
                Actions = [[1,0],[-1,0],[0,1],[0,-1]]
                Actions.remove(best_action)
                epsilon_action = random.choice(Actions)

                selected_action = epsilon_action
                selection_prob = epsilon
            #============================================================================
            
            Actions = [[1,0],[-1,0],[0,1],[0,-1]]
            Actions.remove(selected_action)
            sorted_actions = Actions + [selected_action]
            state_indice_dict = self.state_to_indice(environment)
            state_indice = state_indice_dict[str(state)]
            actions_prob = probs[state_indice]
            actions_prob.sort()
            #due to stochasticity of the environment
            Final_action = random.choices(sorted_actions, actions_prob)[0]
            #print(type(state), type(Final_action))
            
            next_state = [x + y for x, y in zip(state, Final_action)]

            if next_state not in environment[6] or next_state in environment[4]:

                next_state = current_state

            return Final_action, next_state, selection_prob


        state_t  = environment[2]
        
        for trial in tqdm(range(num_trials)):

            trajectory_path = self.generate_trajectory(policy_b,trial,environment_stochasticity)[0]
            
            T = float('inf')
            tau = 0
            t = -1
            trajectory = []
            state_t  = environment[2]
            while tau != T - 1:

                t = t + 1

                if t < T:
                    trajectory.append(state_t)
                    next_state = trajectory_path[t][0]
                    #last_state = state_t
                    state_t = next_state

                    #print(next_state)
                    

                    if next_state == environment[3]:
                        T = t + 1


                
                tau = t - n + 1

                if tau >= 0:

                    G = 0
                    
                    state_tau = trajectory_path[tau][0]
                    state_i = state_tau
                    action_tau =  trajectory_path[tau][1]

                    action_ii = 0

                    ratio = 1

                    for i in range(tau+1, min(tau+n-1,T-1)):

                        r_state_ii = trajectory_path[i][0]
                        r_action_ii = trajectory_path[i][1]

                        greedy_pi = greedy_policy_pi(trajectory_path[i-1][0])

                        if r_action_ii == greedy_pi[0]:

                            pi = greedy_pi[2]
                        else:
                            pi = 1 - greedy_pi[2]
                        
                        b_prob = policy_b_probs[str(r_state_ii)][str(r_action_ii)]

                        ratio = ratio * (pi/b_prob)

                        state_i = r_state_ii
                    
                    

                    #===============

                    state_i = state_tau
                    action_tau = trajectory_path[tau][1]

                    action_ii = 0

                    for i in range(tau+1, min(tau+n,T)):

                        #actionTN = action_ii
                        r_state_ii = trajectory_path[i][0]
                        r_action_ii = trajectory_path[i][1]

                        #stateTN = state_i #s_(t+n) this is store for the next part
                        

                        #n_state_action.append((state_ii, action_ii))

                        #state_i = state_ii
                        r = self.state_reward(r_state_ii)

                        G = G + (gamma ** (i-tau-1)) * r

                        if i < min(tau+n,T):

                            state_i = r_state_ii

                    if tau + n < T:

                        stateTN = trajectory_path[tau + n - 1][0]
                        actionTN = trajectory_path[tau + n - 1][1]


                        #print(tau , n)
                        
                        #tau_n_state = trajectory[tau + n] #indice_state_dict[tau + n]

                        #if state_ii not in environment[4] and state_ii != environment[3]:

                        G = G + (gamma ** n) * Q[str(stateTN)][str(actionTN)]

                    if state_tau not in environment[4] and state_tau != environment[3]:
                        #print(type(tau_state))


                        Q[str(state_tau)][str(action_tau)] = Q[str(state_tau)][str(action_tau)] + alpha * ratio * (G - Q[str(state_tau)][str(action_tau)])
        
        del Q[str(environment[3])]
        return Q

    def n_step_Tree_Backup(self,num_trials, n, gamma, alpha, environment_stochasticity):
    
        grid_size = environment[0]*environment[1]

        probs = self.probability_distribution(grid_size,environment_stochasticity)

        indice_state_dict = {}
        counter = 0
        for state in environment[6]:

            #state = str(state)
            indice_state_dict[counter] = state
            counter = counter + 1

        Q = {}
        for state in environment[6]:

            if state not in environment[4]:
                
                Q[str(state)] = {}

                for action in ["[1, 0]","[-1, 0]","[0, 1]","[0, -1]"]:

                    Q[str(state)][action] = random.uniform(1e-9, 1e-8)
        
        def greedy_policy_pi(current_state):

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
            
            Actions = [[1,0],[-1,0],[0,1],[0,-1]]
            Actions.remove(best_action)
            sorted_actions = Actions + [best_action]
            state_indice_dict = self.state_to_indice(environment)
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


        state_t  = environment[2]

        for trial in tqdm(range(num_trials)):
            
            T = float('inf')
            tau = 0
            t = -1
            trajectory = []
            state_t  = environment[2]

            state_action_frequency  = {}
            for state in environment[6]:

                if state not in environment[4]:
                    
                    state_action_frequency[str(state)] = {}

                    for action in ["[1, 0]","[-1, 0]","[0, 1]","[0, -1]"]:

                        state_action_frequency[str(state)][action] = 1
            
            while tau != T - 1:

                t = t + 1

                if t < T:
                    trajectory.append(state_t)
                    next_state = greedy_policy_pi(state_t)[1]
                    next_action = greedy_policy_pi(state_t)[0]

                    state_action_frequency[str(next_state)][str(next_action)] += 1

                    state_t = next_state
                    

                    if next_state == environment[3]:
                        T = t + 1
                        final_state = next_state

                
                tau = t - n + 1

                if tau >= 0:

                    G = 0

                    if t+1 >= T:

                        G = self.state_reward(final_state)
                    
                    else:

                        action_tau = greedy_policy_pi(state_t)[0]
                        next_tau = greedy_policy_pi(state_t)[1]

                        #def Sigma(state):
                        numberofvisitstplus1 = 0
                        for action in ["[1, 0]","[-1, 0]","[0, 1]","[0, -1]"]:

                            numberofvisitstplus1 = numberofvisitstplus1 + state_action_frequency[str(next_tau)][action]
                        
                        sigma = 0
                        for action in ["[1, 0]","[-1, 0]","[0, 1]","[0, -1]"]:

                            pi_a_s_tplus1 = state_action_frequency[str(next_tau)][action] / numberofvisitstplus1

                            sigma = sigma + pi_a_s_tplus1 * Q[str(next_tau)][str(action)]


                        G = self.state_reward(state_t) + gamma * sigma

                        action_ii = 0
                        state_i = next_tau
                        for i in range(tau+1, min(t,T-1)+1):

                            state_ii = greedy_policy_pi(state_i)[1]
                            action_ii = greedy_policy_pi(state_i)[0]

                            numberofvisitii = 0
                            for action in ["[1, 0]","[-1, 0]","[0, 1]","[0, -1]"]:

                                if action != action_ii:

                                    numberofvisitii = numberofvisitii + state_action_frequency[str(state_ii)][action]
                            
                            sigma = 0
                            for action in ["[1, 0]","[-1, 0]","[0, 1]","[0, -1]"]:

                                if action != action_ii:

                                    pi_a_s_ii = state_action_frequency[str(state_ii)][action] / numberofvisitii

                                    sigma = sigma + pi_a_s_ii * Q[str(state_ii)][str(action)]

                                r = self.state_reward(state_i)

                                G = r + (gamma ** sigma) +\
                                    gamma * (state_action_frequency[str(state_ii)][str(action_ii)] / numberofvisitii) * G

                            state_i = state_ii


                        if state_t not in environment[4] and state_t != environment[3]:
                            #print(type(tau_state))


                            Q[str(state_t)][str(next_action)] = Q[str(state_t)][str(next_action)] + alpha * (G - Q[str(state_t)][str(next_action)])
        
        del Q[str(environment[3])]
        return Q

    def Q_sigma(self, num_trials, n, policy_b, gamma, alpha, environment_stochasticity, epsilon):
        
        grid_size = environment[0]*environment[1]

        probs = self.probability_distribution(grid_size,environment_stochasticity)

        indice_state_dict = {}
        counter = 0
        for state in environment[6]:

            #state = str(state)
            indice_state_dict[counter] = state
            counter = counter + 1
        
        policy_b_state = policy_b[0]
        policy_b_action = policy_b[1]
        policy_b_probs = policy_b[2]

        def choose_sigma(epsiode_length):

            sigma = []

            for t in range(epsiode_length):

                sigma.append(random.random())
            
            return sigma


        Q = {}
        for state in environment[6]:

            if state not in environment[4]:
                
                Q[str(state)] = {}

                for action in ["[1, 0]","[-1, 0]","[0, 1]","[0, -1]"]:

                    Q[str(state)][action] = random.uniform(1e-9, 1e-8)
        
        def greedy_policy_pi(current_state):

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
                selection_prob = 1 - epsilon
            
            else:
                Actions = [[1,0],[-1,0],[0,1],[0,-1]]
                Actions.remove(best_action)
                epsilon_action = random.choice(Actions)

                selected_action = epsilon_action
                selection_prob = epsilon
            #============================================================================
            
            Actions = [[1,0],[-1,0],[0,1],[0,-1]]
            Actions.remove(selected_action)
            sorted_actions = Actions + [selected_action]
            state_indice_dict = self.state_to_indice(environment)
            state_indice = state_indice_dict[str(state)]
            actions_prob = probs[state_indice]
            actions_prob.sort()
            #due to stochasticity of the environment
            Final_action = random.choices(sorted_actions, actions_prob)[0]
            #print(type(state), type(Final_action))
            
            next_state = [x + y for x, y in zip(state, Final_action)]

            if next_state not in environment[6] or next_state in environment[4]:

                next_state = current_state

            return Final_action, next_state, selection_prob


        state_t  = environment[2]

        for trial in tqdm(range(num_trials)):

            trajectory_path = self.generate_trajectory(policy_b,trial,environment_stochasticity)[0]

            
            T = float('inf')
            tau = 0
            t = -1
            trajectory = []
            state_t  = environment[2]

            state_action_frequency  = {}
            for state in environment[6]:

                if state not in environment[4]:
                    
                    state_action_frequency[str(state)] = {}

                    for action in ["[1, 0]","[-1, 0]","[0, 1]","[0, -1]"]:

                        state_action_frequency[str(state)][action] = 1
            
            ratio = {}
            for state in environment[6]:

                if state not in environment[4]:
                    
                    ratio[str(state)] = {}

                    for action in ["[1, 0]","[-1, 0]","[0, 1]","[0, -1]"]:

                        ratio[str(state)][action] = 1
            

            while tau != T - 1:

                t = t + 1

                if t < T:
                    #trajectory.append(state_t)
                    next_state = trajectory_path[t][0]
                    next_action = trajectory_path[t][1]

                    if next_state != environment[3]:

                        state_action_frequency[str(next_state)][str(next_action)] += 1

                        numberofvisitstplus1 = 0
                        for action in ["[1, 0]","[-1, 0]","[0, 1]","[0, -1]"]:

                            numberofvisitstplus1 = numberofvisitstplus1 + state_action_frequency[str(next_state)][action]
                        
                        pi_A_S = state_action_frequency[str(next_state)][str(next_action)] / numberofvisitstplus1
                        b_A_S = policy_b_probs[str(next_state)][str(next_action)]

                        ratio_A_S = pi_A_S / b_A_S

                        ratio[str(next_state)][str(next_action)] = ratio_A_S


                        state_t = next_state

                    if next_state == environment[3]:
                        T = t + 1
                        final_state = next_state

                
                tau = t - n + 1

                if tau >= 0:

                    G = 0

                    action_ii = 0
                    #state_i = next_tau

                    for i in range(tau+1, min(t+1,T)+1):

                        if i == T:

                            G = state_reward(final_state)
                        
                        else:

                            action_i = trajectory_path[i][1]
                            next_i = trajectory_path[i][0]
                            if next_i not in environment[4] and next_i != environment[3]:

                                #def Sigma(state):
                                numberofvisitstplus1 = 0
                                for action in ["[1, 0]","[-1, 0]","[0, 1]","[0, -1]"]:

                                    numberofvisitstplus1 = numberofvisitstplus1 + state_action_frequency[str(next_i)][action]
                                
                                sigma = 0

                                for action in ["[1, 0]","[-1, 0]","[0, 1]","[0, -1]"]:

                                    pi_a_s_tplus1 = state_action_frequency[str(next_i)][action] / numberofvisitstplus1

                                    sigma = sigma + pi_a_s_tplus1 * Q[str(next_i)][str(action)]

                                V_bar = sigma

                                for action in ["[1, 0]","[-1, 0]","[0, 1]","[0, -1]"]:

                                    numberofvisitstplus1 = numberofvisitstplus1 + state_action_frequency[str(next_i)][action]
                                
                                pi_Ai_Si = state_action_frequency[str(next_i)][str(action_i)] / numberofvisitstplus1
                                
                                R_k = self.state_reward(next_i)
                                #print(len(trajectory_path))
                                sigma_k = choose_sigma(len(trajectory_path))

                                G = R_k + \
                                    gamma * (sigma_k[i] * ratio[str(next_i)][str(action_i)] + (1-sigma_k[i]) * pi_Ai_Si) * (G -  Q[str(next_i)][str(action_i)])+\
                                        gamma * V_bar

                                    #state_i = state_ii

                                next_state = trajectory_path[t][0]
                                next_action = trajectory_path[t][1]

                                Q[str(next_state)][str(next_action)] = Q[str(next_state)][str(next_action)] + alpha * (G - Q[str(next_state)][str(next_action)])
                
        del Q[str(environment[3])]
        return Q