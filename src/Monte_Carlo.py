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

    def policy_evaluation(self, threshold,gamma,randomness):
        random.seed(42)

        grid_size = environment[0]*environment[1]
        
        grid_cells = environment[6]

        Actions = [[1,0],[-1,0],[0,1],[0,-1]]

        prob_dist = probability_distribution(environment[0]*environment[1],randomness)
        
        #this dictionary stores state indices with their linear number; [0,0] = 0 ,..., [m,n] = m.n 
        indice_state_dict = {}
        for num in range(grid_size):

            indice_state_dict[num] = grid_cells[num]
        

        state_indice_dict = {}
        counter = 0
        for state in grid_cells:

            state = str(state)
            state_indice_dict[state] = counter
            counter = counter + 1

        #initialize the state values with zeros - minus 1 is because of terminal state or goal
        V_old = np.zeros((grid_size , 1))
        V_new = np.zeros((grid_size , 1))
        
        Counter = 0
        delta = threshold + 0.2
        while delta >= threshold:
            Delta = []
            for state_num in range(grid_size):

                state = indice_state_dict[state_num]
                #state_indice = '{}{}'.format(row,col)

                if state not in environment[4]: 

                    first_sigma = 0
                    #print('come to loop',state)
                
                    for action in range(4):
                        #print(action)


                        #as we are working on random policy the pi(a|s) = 1/4; equal probable
                        pi = 1/4

                        neighbors = self.neighbor_cells(state)
                        #print(neighbors)
                        
                        #second sigma in the Bellman equation
                        
                        intended_state = [x + y for x, y in zip(state, Actions[action])]
                        #inverse_state = 
                        second_sigma = 0
                        second_sigma_list = []
                        for neighbor in neighbors:
                            #print('third loop')

                            prob_list = prob_dist[state_num].copy()

                            if neighbor == intended_state:

                                #print(prob_list)
                                p = max(prob_list)
                                prob_list.remove(p)
                                #print(p)

                            else:

                                p = random.choice(prob_dist[state_num]) #prob_dist[state_num][action-1]
                                #print(p)
                            
                            if neighbor in grid_cells:
                                indice = state_indice_dict[str(neighbor)]
                            
                            #if the agent reach the goal, we eliminate the -1 reward
                            if intended_state == environment[3]:

                                second_sigma = second_sigma + p*(100000+gamma*V_old[indice])
                            
                            #in this part we dedicated a very large negative reward if the agent drop on a hole
                            elif neighbor in environment[4]:

                                second_sigma = second_sigma + p*(-2 + gamma*V_old[indice])
                            #in other states, which are not the teriminal state or holes; reward = -1

                            elif neighbor not in grid_cells:

                                second_sigma = second_sigma + p*(-1)

                            else:
                                second_sigma = second_sigma + p*(-1 + gamma*V_old[indice])

                        second_sigma_list.append(pi * second_sigma)
                        
                    first_sigma = sum(second_sigma_list) #first_sigma + pi * second_sigma_list[action]
                    
                    V_new[state_num] = first_sigma
                    if state == [3,3]:

                        print('[3,3]:',first_sigma,second_sigma_list)
                
            
            
                #if Counter == 0:

                delta_ = max([0,np.abs(V_new[state_num] - V_old[state_num])])
                Delta.append(delta_)

                    #Counter = Counter + 1

                """else:

                    delta_ = max([delta_,np.abs(V_new[state_num] - V_old[state_num])])
                    Delta.append(delta_)

                    Counter = Counter + 1"""
                #print(V_old)
                #print('=====')
                #print(V_new)
                V_old[state_num] = V_new[state_num]
            
            delta = max(Delta)
            print(delta)
            #print(Delta)
            #print('delta:',delta)
            #print(Counter)
                
        Final_dict = {}
        for state in grid_cells:

            if state not in environment[4]:

                Final_dict[str(state)] = V_new[state_indice_dict[str(state)]]

        return Final_dict
    
    def find_optimal_path(self, StateValue):

        start = environment[2]
        goal = environment[3]

        path = []

        #neighbors = neighbor_cells(start)
        
        next_move = start
        ex_move = []
        counter = 0
        checked_states = []
        
        while goal not in DynamicProgramming.neighbor_cells(next_move):

            neighbor_values = {}
            """if counter != 0:
                Neighbors_ = neighbor_cells(next_move)
                Neighbors = Neighbors_.copy()
                for neighbor in Neighbors:

                    if neighbor in checked_states:

                        Neighbors.remove(neighbor)
                #print(counter, Neighbors)
            
            else:
                Neighbors = neighbor_cells(next_move).copy()"""
            
            #for neighbor in
            Neighbors = neighbor_cells(next_move)

            Allowed_Neighbors = Neighbors.copy()

            for neighbor in Allowed_Neighbors:

                if neighbor in checked_states:

                        Allowed_Neighbors.remove(neighbor)



            for neighbor in Allowed_Neighbors:
                
                if neighbor in environment[6] and neighbor not in environment[4]:

                    value = StateValue[str(neighbor)][0]

                    neighbor_values[value] = neighbor
                    #checked_states.append(neighbor)
            #ex_move = next_move
            """for state in Neighbors:

                if state in environment[6]:

                    checked_states.append(state)"""
            print(neighbor_values)

            maximum_value = max(list(neighbor_values.keys()))

            next_move = neighbor_values[maximum_value]

            checked_states.append(next_move)

            print(next_move)

            path.append(next_move)
            
            counter += 1
        path.append(environment[3])
        return path

        if goal in neighbor_cells(next_move):

            return "Just one step" #should be edited later
    
    def policy(self, state):

        Neighbors = neighbor_cells(state)

        neighbor_values = {}

        for neighbor in Neighbors:

            if neighbor in environment[6] and neighbor not in environment[4]:

                neighbor_values[states_values['{}'.format(neighbor)][0]] = '{}'.format(neighbor)
        
        best_val = max(list(neighbor_values.keys()))
        best_neighbor = neighbor_values[best_val]

        return best_neighbor
    
    def arbitrary_policy(self, randomness):
            grid_cells = environment[6]
            Actions = [[1, 0], [-1, 0], [0, 1], [0, -1]]
            policy = {}

            for state in grid_cells:
                if state not in environment[4]:
                    neighbors = self.neighbor_cells(state)
                    allowed_positions = []

                    for neighbor in neighbors:
                        if neighbor in grid_cells and neighbor not in environment[4]:
                            allowed_positions.append(neighbor)

                    next_state = random.choice(allowed_positions)
                    policy['{}'.format(state)] = next_state

            return policy
    
    def policy_iteration(self, threshold,gamma,randomness,state_prob_type):
    
        random.seed(randomness)
        
        policy_0 = self.arbitrary_policy(42)


        grid_size = environment[0]*environment[1]

        grid_cells = environment[6]

        prob_dist = self.probability_distribution(environment[0]*environment[1],state_prob_type)
        
        #this dictionary stores state indices with their linear number; [0,0] = 0 ,..., [m,n] = m.n 
        indice_state_dict = {}
        for num in range(grid_size):

            indice_state_dict[num] = grid_cells[num]
        

        state_indice_dict = {}
        counter = 0
        for state in grid_cells:

            state = str(state)
            state_indice_dict[state] = counter
            counter = counter + 1   

        def PolicyEvaluation(policy,threshold,gamma,randomness):
                
            random.seed(randomness)

            #initialize the state values with zeros - minus 1 is because of terminal state or goal
            V_old = np.zeros((grid_size , 1))
            V_new = np.zeros((grid_size , 1))
            
            delta = threshold + 0.2
            while delta >= threshold:
                Delta = []
                for state_num in range(grid_size):

                    state = indice_state_dict[state_num]
                    #state_indice = '{}{}'.format(row,col)

                    if state not in environment[4]: 

                        #first_sigma = 0
                        
                        neighbors = self.neighbor_cells(state)
                            
                        #intended_state = [x + y for x, y in zip(state, policy(state))]
                        
                        intended_state = policy['{}'.format(state)]
                        

                        second_sigma = 0
                        second_sigma_list = []
                        for neighbor in neighbors:

                            prob_list = prob_dist[state_num].copy()

                            maximum_p = max(prob_list)
                            prob_list.remove(maximum_p)

                            if neighbor == intended_state:

                                p = maximum_p
                                #prob_list.remove(p)

                                """else:

                                if len(prob_list) == 4:
                                    p = max(prob_list)
                                    prob_list.remove(p)

                                p = random.choice(prob_dist[state_num]) #prob_dist[state_num][action-1]"""
                            
                            else:
                            
                                p = random.choice(prob_list)

                                #print(p)
                            
                            if neighbor in grid_cells:
                                indice = state_indice_dict[str(neighbor)]
                                #print(indice)
                            
                            #if the agent reach the goal, we eliminate the -1 reward
                            if intended_state == environment[3]:

                                second_sigma = second_sigma + p*(10+gamma*V_old[indice])
                                #print('goal:',second_sigma)

                            
                            #in this part we dedicated a very large negative reward if the agent drop on a hole
                            elif neighbor in environment[4]:

                                second_sigma = second_sigma + p*(-3 + gamma*V_old[indice])
                                #print('hole:',second_sigma)
                            #in other states, which are not the teriminal state or holes; reward = -1

                            #elif neighbor not in grid_cells:

                            #    second_sigma = second_sigma + p*(-2)
                                #print('out:',second_sigma)

                            else:
                                second_sigma = second_sigma + p*(-1 + gamma*V_old[indice])
                                #print('in:',second_sigma)

                            #second_sigma_list.append(second_sigma)
                        
                        #first_sigma = sum(second_sigma_list) #first_sigma + pi * second_sigma_list[action]
                        
                        #V_new[state_num] = sum(second_sigma_list) #first_sigma
                        V_new[state_num] = second_sigma # max(V_new[state_num], second_sigma)

                    delta_ = max([0,np.abs(V_new[state_num] - V_old[state_num])])
                    Delta.append(delta_)

                    V_old[state_num] = V_new[state_num]
                    
                delta = max(Delta)
                #print('delta',delta)
                    
            Final_dict = {}
            for state in grid_cells:

                if state not in environment[4]:

                    Final_dict[str(state)] = V_new[state_indice_dict[str(state)]]

            return Final_dict

        def policy_improvement(policy,threshold,gamma,randomness):

            policy_stable = False
            policy = policy_0
            c = 0
            while policy_stable == False:

                State_Values = PolicyEvaluation(policy,threshold,gamma,randomness)
                #print('sv',State_Values)

                #print(max(list(State_Values.values())))
                #print(c)
                for state_num in range(grid_size):

                    state = indice_state_dict[state_num]
                    #state_indice = '{}{}'.format(row,col)

                    if state not in environment[4]:
                        string_state = '{}'.format(state)

                        first_sigma = 0
                        
                        neighbors = self.neighbor_cells(state)
                            
                        #intended_state = [x + y for x, y in zip(state, policy(state))]
                        old_policy = policy
                        intended_state = policy[string_state]

                        second_sigma = 0
                        second_sigma_dict = {}
                        best_value = float("-inf")  # Initialize the best value to negative infinity
                        best_neighbor = None

                        for neighbor in neighbors:

                            if neighbor in environment[6] and neighbor not in environment[4]:

                                prob_list = prob_dist[state_num].copy()
                                #print(prob_list)
                                #print(neighbor)
                                #print(State_Values)

                                maximum_p = max(prob_list)
                                prob_list.remove(maximum_p)

                                if neighbor == intended_state:

                                    p = maximum_p
                                    #prob_list.remove(p)

                                else:

                                    p = random.choice(prob_list) #prob_dist[state_num][action-1]
                                    #print(p)
                                
                                if neighbor in grid_cells:
                                    indice = state_indice_dict[str(neighbor)]
                                #print(intended_state , environment[3])
                                #if the agent reach the goal, we eliminate the -1 reward
                                if intended_state == str(environment[3]):
                                    #print(type(intended_state) , type(environment[3]))

                                    second_sigma = second_sigma + p*(10 + gamma*State_Values[string_state])
                                
                                #in this part we dedicated a very large negative reward if the agent drop on a hole
                                elif neighbor in environment[4]:

                                    second_sigma = second_sigma + p*(-3 + gamma*State_Values[string_state])
                                #in other states, which are not the teriminal state or holes; reward = -1

                                #elif neighbor not in grid_cells:

                                #    second_sigma = second_sigma + p*(-2)

                                else:
                                    second_sigma = second_sigma + p*(-1 + gamma*State_Values[string_state])

                                #print(second_sigma,neighbor)
                                #second_sigma = second_sigma[0]
                                #second_sigma_dict[second_sigma] = neighbor
                            
                                if type(second_sigma) == float:
                                    val = round(second_sigma,5)
                                else:
                                    val = round(second_sigma[0],5)
                                second_sigma_dict[val] = str(neighbor)

                        maximum_value = max(list(second_sigma_dict.keys()))
                        best_neighbor = second_sigma_dict[maximum_value]

                        if environment[3] in neighbors:

                            policy[string_state] = str(environment[3])
                        else:

                            policy[string_state] = best_neighbor

                        #if state == [4,2]:
                            #print(second_sigma_dict)
                            #print('policy:',policy)

                        if old_policy == policy:
                            policy_stable = True

                        if str(intended_state) != best_neighbor:
                            #print(intended_state,best_neighbor)
                            
                            policy_stable = False

                c +=1

            
            if policy_stable == True:

                return State_Values, policy
    
        optimals = policy_improvement(policy_0,threshold,gamma,randomness)
        optimal_value = optimals[0]
        optimal_value.pop(str(environment[3]))
        
        optimal_policy = optimals[1]
        optimal_policy.pop(str(environment[3]))

        return optimal_value, optimal_policy

        
if __name__ == "__main__":
    dp = DynamicProgramming()

    # Generate the environment using your generate_grid_world function
    environment = dp.generate_grid_world(5, 4,4,4,39)

    # Obtain the optimal value and policy using policy_iteration
    optimal_value, optimal_policy = dp.policy_iteration(0.9,.7,42,'stochastic')

    # Print the results
    print("Optimal Value:")
    print(optimal_value)
    print("Optimal Policy:")
    #print(optimal_policy)