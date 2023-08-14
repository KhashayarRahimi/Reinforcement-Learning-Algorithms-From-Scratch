import numpy as np
import random
from tabulate import tabulate
from tqdm import tqdm
import ast

class PlanningLearning:
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
    
    def sample_model(self,Environment):

        """ As we are working on the gridworld, we know the reward function in general.
            Also, we know the start and terminal point.
            we dedicate -1 reward to each state transition except when the next state is the terminal.
            As we do not have any information about the environmen's details, detecting holes
            and returning high negative rewsard is not possible. 
        """
        #grid_size = Environment[0]*Environment[1]
        #start = Environment[2]
        terminal = Environment[3]

        state_action_reward_nextstate = {}

        for state in Environment[6]:

            state_action_reward_nextstate[str(state)] = {}

            for action in [[1, 0],[-1, 0],[0, 1],[0, -1]]:

                state_action_reward_nextstate[str(state)][str(action)] = []

        for state in Environment[6]:

            for action in [[1, 0],[-1, 0],[0, 1],[0, -1]]:

                next_state = [x + y for x, y in zip(state, action)]

                if next_state == terminal:

                    state_action_reward_nextstate[str(state)][str(action)] = [0, next_state]
                
                else:

                    if next_state in Environment[6]:

                        state_action_reward_nextstate[str(state)][str(action)] = [-1, next_state]
                    
                    else:

                        state_action_reward_nextstate[str(state)][str(action)] = [-1, state]
        

        return state_action_reward_nextstate    

    def random_sample_Q_planning(self,num_trials, gamma, alpha, environment_stochasticity, sample_model):

        grid_size = environment[0]*environment[1]

        probs = self.probability_distribution(grid_size,environment_stochasticity)

        Q = {}
        for state in environment[6]:
                
            Q[str(state)] = {}

            for action in ["[1, 0]","[-1, 0]","[0, 1]","[0, -1]"]:

                Q[str(state)][action] = random.uniform(1e-9, 1e-8)
        
        next_state = environment[2] #start state

        for trial in tqdm(range(num_trials)):

            current_state = next_state

            Actions = [[1, 0],[-1, 0],[0, 1],[0, -1]]

            random_action = random.choice(Actions)
            
            Actions.remove(random_action)
            sorted_actions = Actions + [random_action]
            state_indice_dict = self.state_to_indice(environment)
            state_indice = state_indice_dict[str(state)]
            actions_prob = probs[state_indice]
            actions_prob.sort()
            #due to stochasticity of the environment
            Final_action = random.choices(sorted_actions, actions_prob)[0]

            new_state = [x + y for x, y in zip(current_state, Final_action)]

            if new_state in environment[6]:

                next_state = new_state
            
            else:

                next_state = current_state

            reward = sample_model[str(next_state)][str(Final_action)][0]

            value_action_state = self.reverse_dictionary(Q[str(next_state)])
            Max_val = max(list(value_action_state.keys()))

            Q[str(current_state)][str(Final_action)] = Q[str(current_state)][str(Final_action)] + alpha * (reward + gamma * Max_val - Q[str(current_state)][str(Final_action)])


        return Q


    def Dyna_Q(self, num_trials, n, gamma, alpha, sample_model, epsilon):

        Q = {}
        for state in environment[6]:
                
            Q[str(state)] = {}

            for action in ["[1, 0]","[-1, 0]","[0, 1]","[0, -1]"]:

                Q[str(state)][action] = random.uniform(1e-9, 1e-8)
        
        next_state = environment[2] #start state

        for trial in tqdm(range(num_trials)):

            Observation = {}

            current_state = next_state

            Actions = [[1, 0],[-1, 0],[0, 1],[0, -1]]

            value_action_state = self.reverse_dictionary(Q[str(state)])
            Max_val = max(list(value_action_state.keys()))
            best_action = value_action_state[Max_val]
            best_action = ast.literal_eval(best_action)
            #Epsilon Greedy
            if random.uniform(0, 1) > epsilon:

                selected_action = best_action
            
            else:
                Actions = [[1,0],[-1,0],[0,1],[0,-1]]
                Actions.remove(best_action)
                epsilon_action = random.choice(Actions)

                selected_action = epsilon_action 

            #As the book mentioned that "assuming deterministic environment"
            new_state = [x + y for x, y in zip(current_state, selected_action)]

            if new_state in environment[6]:

                next_state = new_state
            
            else:

                next_state = current_state

            reward = self.state_reward(next_state)

            value_action_state = self.reverse_dictionary(Q[str(next_state)])
            Max_val = max(list(value_action_state.keys()))

            Q[str(current_state)][str(selected_action)] = Q[str(current_state)][str(selected_action)] + alpha * (reward + gamma * Max_val - Q[str(current_state)][str(selected_action)])

            sample_model[str(next_state)][str(selected_action)] = [reward, next_state]
            
            if str(next_state) not in list(Observation.keys()):

                Observation[str(next_state)] = []


            Observation[str(next_state)] = Observation[str(next_state)] + [selected_action]

            #Loop repeat n times

            for i in range(n):

                rand_state = random.choice(list(Observation.keys()))

                observed_actions = Observation[str(rand_state)]

                rand_action = random.choice(observed_actions)

                reward = sample_model[rand_state][str(rand_action)][0]

                new_state = sample_model[rand_state][str(rand_action)][1]

                value_action_state = self.reverse_dictionary(Q[str(new_state)])
                Max_val = max(list(value_action_state.keys()))

                Q[str(rand_state)][str(rand_action)] = Q[str(rand_state)][str(rand_action)] + alpha * (reward + gamma * Max_val - Q[str(rand_state)][str(rand_action)])

        return Q