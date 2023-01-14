# encoding=utf8
import copy
import logging

import numpy as np

from niapy.algorithms.algorithm import Algorithm, Individual, default_individual_init
from sklearn.metrics import jaccard_score
from gekko import GEKKO
import random

logging.basicConfig()
logger = logging.getLogger('niapy.algorithms.basic')
logger.setLevel('INFO')

__all__ = ['ArtificialBeeColonyAlgorithm']


class SolutionABC(Individual):
    r"""Representation of solution for Artificial Bee Colony Algorithm.

    Date:
        2018

    Author:
        Klemen Berkovič

    See Also:
        * :class:`niapy.algorithms.Individual`

    """

    pass


class ArtificialBeeColonyAlgorithm(Algorithm):
    r"""Implementation of Artificial Bee Colony algorithm.

    Algorithm:
        Artificial Bee Colony algorithm

    Date:
        2018

    Author:
        Uros Mlakar and Klemen Berkovič

    License:
        MIT

    Reference paper:
        Karaboga, D., and Bahriye B. "A powerful and efficient algorithm for numerical function optimization: artificial bee colony (ABC) algorithm." Journal of global optimization 39.3 (2007): 459-471.

    Arguments
        Name (List[str]): List containing strings that represent algorithm names
        limit (Union[float, numpy.ndarray[float]]): Maximum number of cycles without improvement.

    See Also:
        * :class:`niapy.algorithms.Algorithm`

    """

    Name = ['ArtificialBeeColonyAlgorithm', 'ABC']

    @staticmethod
    def info():
        r"""Get algorithms information.

        Returns:
            str: Algorithm information.

        See Also:
            * :func:`niapy.algorithms.Algorithm.info`

        """
        return r"""Karaboga, D., and Bahriye B. "A powerful and efficient algorithm for numerical function optimization: artificial bee colony (ABC) algorithm." Journal of global optimization 39.3 (2007): 459-471."""

    def __init__(self, population_size=20, limit=5, *args, **kwargs):
        """Initialize ArtificialBeeColonyAlgorithm.

        Args:
            population_size (Optional[int]): Population size.
            limit (Optional[int]): Maximum number of cycles without improvement.

        See Also:
            :func:`niapy.algorithms.Algorithm.__init__`

        """
        super().__init__(population_size, initialization_function=default_individual_init, individual_type=SolutionABC,
                         *args, **kwargs)
        self.limit = limit
        self.food_number = self.population_size 
        

    def set_parameters(self, population_size=20, limit=5, **kwargs):
        r"""Set the parameters of Artificial Bee Colony Algorithm.

        Args:
            population_size(Optional[int]): Population size.
            limit (Optional[int]): Maximum number of cycles without improvement.

        See Also:
            * :func:`niapy.algorithms.Algorithm.set_parameters`

        """
        super().set_parameters(population_size=population_size, initialization_function=default_individual_init,
                               individual_type=SolutionABC, **kwargs)
        self.food_number = self.population_size 
        self.limit = limit

    def get_parameters(self):
        """Get parameters."""
        params = super().get_parameters()
        params.update({
            'limit': self.limit
        })
        return params

    def calculate_probabilities(self, foods):
        r"""Calculate the probes.

        Args:
            foods (numpy.ndarray): Current population.

        Returns:
            numpy.ndarray: Probabilities.

        """
        probs = np.asarray([1.0 / (foods[i].f + 0.01) for i in range(self.food_number)])
        probs = probs / np.sum(probs)
        return np.cumsum(probs)

    def init_population(self, task):
        r"""Initialize the starting population.

        Args:
            task (Task): Optimization task

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray[float], Dict[str, Any]]:
                1. New population
                2. New population fitness/function values
                3. Additional arguments:
                    * trials (numpy.ndarray): Number of cycles without improvement.

        See Also:
            * :func:`niapy.algorithms.Algorithm.init_population`

        """
        foods, fpop, _ = super().init_population(task)
        trials = np.zeros(self.food_number, dtype=np.int32)
        return foods, fpop, {'trials': trials}
    

    def run_iteration(self, task, population, population_fitness, Gbest, best_f, **params):
        r"""Core function of  the algorithm.

        Args:
            task (Task): Optimization task
            population (numpy.ndarray): Current population
            population_fitness (numpy.ndarray[float]): Function/fitness values of current population
            Gbest (numpy.ndarray): Current best individual
            best_f (float): Current best individual fitness/function value
            params (Dict[str, Any]): Additional parameters

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, float, Dict[str, Any]]:
                1. New population
                2. New population fitness/function values
                3. New global best solution
                4. New global best fitness/objective value
                5. Additional arguments:
                    * trials (numpy.ndarray): Number of cycles without improvement.

        """
        trials = params.pop('trials')
        #print("*************************************************************")
        #print("Population: ")
        
        for i in range(len(population)):
            ix = np.where(population[i].x > 0.5, 1, 0)
            population[i] = SolutionABC(ix,task,True,self.rng)
            population[i].evaluate(task,self.rng)
            #print(population[i])
        
        #print("Gbest: ", Gbest)
        #print("best_f: ", best_f)
        
        #Faza zaposlenih pcela
        #print("Zaposlene pcele")
        for i in range(self.food_number):
            #print("iteracija: ", i)
            # ********************************

            xi = population[i].x
            i1 = np.random.randint(0,self.food_number)
            while i1 == i:
                i1 = np.random.randint(0,self.food_number)
            i2 = np.random.randint(0,self.food_number)
            while i2 == i1 or i2 == i:
                i2 = np.random.randint(0,self.food_number)
            i3 = np.random.randint(0,self.food_number)
            while i3 == i2 or i3 == i1 or i3 == i:
                i3 = np.random.randint(0,self.food_number)
            neighbor_first = population[i1]
            neighbor_second = population[i2]
            neighbor_third = population[i3]
            
            #print("Susjedi: ")
            #print("     ", neighbor_first, "    ", i1)
            #print("     ", neighbor_second, "    ", i2)
            #print("     ", neighbor_third, "    ", i3)
            
            xr1 = neighbor_first.x
            xr2 = neighbor_second.x
            xr3 = neighbor_third.x
            
            diss = 1 - jaccard_score(xr2,xr3)
            fi = 0.8 
            fi_diss = fi*diss
            
            ############ HILL CLIMBING
            
            m1 = np.sum(xr1)
            m0 = len(xr1) - m1
            M11 = 0
            M10 = 0
            M01 = 0
            
            wi = [0 for j in range(len(xi))]
            for j in range(len(xi)):
                if xi[j] == 1 and wi[j] == 1:
                    M11 += 1
                if xi[j] == 1 and wi[j] == 0:
                    M10 += 1
                if xi[j] == 0 and wi[j] == 1:
                    M01 += 1
                    
            if M11 + M10 + M01 == 0:
                minM = abs(1 -fi_diss)
            else:                    
                minM = abs(1 - M11 / (M11 + M10 + M01) - fi_diss)
            
            while True:
                found_better_neighbour = False
                neighbours = self.get_neighbours(xi)

                for wi_new in neighbours:
                    M11 = 0
                    M10 = 0
                    M01 = 0
                    for j in range(len(xi)):
                        if xi[j] == 1 and wi_new[j] == 1:
                            M11 += 1
                        if xi[j] == 1 and wi_new[j] == 0:
                            M10 += 1
                        if xi[j] == 0 and wi_new[j] == 1:
                            M01 += 1
                    if M11 + M10 + M01 == 0:
                        minM_new = abs(1 -fi_diss)
                    else:                    
                        minM_new = abs(1 - M11 / (M11 + M10 + M01) - fi_diss)
                    if M11 + M01 == m1 and M10 <= m0 and minM_new < minM:
                        wi = wi_new
                        minM = minM_new
                        found_better_neighbour = True
                        
                if not found_better_neighbour:
                    break
                       
                
                ################# INTEGER MODEL PROGRAMMING

#            
#             m1 = np.sum(xr1)
#             m0 = len(xr1) - m1
            
#             m = GEKKO()
#             m.options.SOLVER = 1 #integer solver
#             m.options.MAX_ITER = 500
            
#             M11 = m.Var(integer=True, name="M11")
#             M10 = m.Var(integer=True, name="M10")
#             M01 = m.Var(integer=True, name="M01")   
            
#             M11.value = m1
#             M10.value = 0
#             M01.value = m0
            
#             m.Equation(M11 + M01 == m1)
#             m.Equation(M10 <= m0)
#             m.Equation(M11 >= 0)
#             m.Equation(M10 >= 0)
#             m.Equation(M01 >= 0)
            
#             m.Minimize(abs(1 - M11 / (M11 + M10 + M01) - fi_diss))
            
#             m.solve(disp=False)
#
            
#             wi = [0 for j in range(len(xr1))]
#             wi = np.array(wi)
#             xr1 = np.array(xr1)
    
            
#             index_M11 = np.where(xr1 == 1)
#             np.add.at(wi, random.choices(index_M11[0], k=int(M11.value[0])-1), 1)
#             index_M10 = np.where(xr1 == 0)
#             np.add.at(wi, random.choices(index_M10[0], k=int(M10.value[0])-1), 1)
            

            # omegai = []
            # if np.random.rand() > 0.5:
            #     if Wi.f < Neighbor_first.f:
            #         omegai = copy.deepcopy(Wi)
            #     else:
            #         omegai = copy.deepcopy(Neighbor_first)
            # else:
            #     if np.random.rand() > 0.5:
            #         omegai = copy.deepcopy(Wi)
            #     else:
            #         omegai = copy.deepcopy(Neighbor_first)
                    
            # Omegai = SolutionABC(omegai, task, True, self.rng)
            # Omegai.evaluate(task, self.rng)
            
            Neighbor_first = SolutionABC(xr1, task, True, rng=self.rng)
            Neighbor_first.evaluate(task,self.rng)
            
            Wi = SolutionABC(wi, task, True, rng=self.rng)
            Wi.evaluate(task,self.rng)
            
            Omegai = copy.deepcopy(Wi)
                    
            #REKOMBINACIJA
            cr = 0.25
            ui = []
            for j in range(len(Omegai.x)):
                if np.random.rand() < cr:
                    ui.append(Omegai.x[j])
                else:
                    ui.append(xi[j])
                    
            Ui = SolutionABC(ui, task, True, rng=self.rng)
            Ui.evaluate(task,self.rng)
            
            if Ui.f < population[i].f:
                population[i] = copy.deepcopy(Ui)
                trials[i] = 0
            else:
                #population[i] = xi ostaje
                trials[i] += 1

                
        #print("Promatraci pcele")
        #Faza promatraca
        probabilities = self.calculate_probabilities(population)
        for i in range(self.food_number):
            #print("Iteracija: ", i)
            r = np.random.rand()
            index = 0
            while r > probabilities[index]:
                index += 1
            xi = population[index].x
            
            #************* FAZA ZAPOSLENIH PČELA OD 3 SUSJEDA
            i1 = np.random.randint(0,self.food_number)
            while i1 == index:
                i1 = np.random.randint(0,self.food_number)
            i2 = np.random.randint(0,self.food_number)
            while i2 == i1 or i2 == index:
                i2 = np.random.randint(0,self.food_number)
            i3 = np.random.randint(0,self.food_number)
            while i3 == i2 or i3 == i1 or i3 == index:
                i3 = np.random.randint(0,self.food_number)
            neighbor_first = population[i1]
            neighbor_second = population[i2]
            neighbor_third = population[i3]
            
            #print("Susjedi: ")
            #print("     ", neighbor_first, "    ", i1)
            #print("     ", neighbor_second, "    ", i2)
            #print("     ", neighbor_third, "    ", i3)
            
            xr1 = neighbor_first.x
            xr2 = neighbor_second.x
            xr3 = neighbor_third.x
            
            ##### INTEGER MODEL PROGRAMMING
            
            # diss = 1 - jaccard_score(xr2,xr3)
            # fi = 0.8 #za sada, inace u svakoj iteraciji izracunati
            # fi_diss = fi*diss
            
            # m1 = np.sum(xr1)
            # m0 = len(xr1) - m1
            
            # m = GEKKO()
            # m.options.SOLVER = 1 #integer solver
            # m.options.MAX_ITER = 500
            
            # M11 = m.Var(integer=True, name="M11")
            # M10 = m.Var(integer=True, name="M10")
            # M01 = m.Var(integer=True, name="M01")   
            
            # M11.value = m1
            # M10.value = 0
            # M01.value = m0
            
            # m.Equation(M11 + M01 == m1)
            # m.Equation(M10 <= m0)
            # m.Equation(M11 >= 0)
            # m.Equation(M10 >= 0)
            # m.Equation(M01 >= 0)
            
            # m.Minimize(abs(1 - M11 / (M11 + M10 + M01) - fi_diss))
            
            # m.solve(disp=False)
            
            # wi = [0 for j in range(len(xr1))]
            # wi = np.array(wi)
            # xr1 = np.array(xr1)
    
            # index_M11 = np.where(xr1 == 1)
            # np.add.at(wi, random.choices(index_M11[0], k=int(M11.value[0])-1), 1)
            # index_M10 = np.where(xr1 == 0)
            # np.add.at(wi, random.choices(index_M10[0], k=int(M10.value[0])-1), 1)
  
            # Wi = SolutionABC(wi, task, True, rng=self.rng)
            # Wi.evaluate(task,self.rng)
            
            
            # Neighbor_first = SolutionABC(xr1, task, True, rng=self.rng)
            # Neighbor_first.evaluate(task,self.rng)
            
            # omegai = []
            # if np.random.rand() > 0.5:
            #     if Wi.f < Neighbor_first.f:
            #         omegai = copy.deepcopy(Wi)
            #     else:
            #         omegai = copy.deepcopy(Neighbor_first)
            # else:
            #     if np.random.rand() > 0.5:
            #         omegai = copy.deepcopy(Wi)
            #     else:
            #         omegai = copy.deepcopy(Neighbor_first)
                    
            # Omegai = SolutionABC(omegai, task, True, self.rng)
            # Omegai.evaluate(task, self.rng)
            
            ############# HILL CLIMBING
            
            diss = 1 - jaccard_score(xr2,xr3)
            fi = 0.8
            fi_diss = fi*diss
            
            m1 = np.sum(xr1)
            m0 = len(xr1) - m1
            M11 = 0
            M10 = 0
            M01 = 0
            
            wi = [0 for j in range(len(xi))]
            for j in range(len(xi)):
                if xi[j] == 1 and wi[j] == 1:
                    M11 += 1
                if xi[j] == 1 and wi[j] == 0:
                    M10 += 1
                if xi[j] == 0 and wi[j] == 1:
                    M01 += 1
                    
            if M11 + M10 + M01 == 0:
                minM = abs(1 -fi_diss)
            else:                    
                minM = abs(1 - M11 / (M11 + M10 + M01) - fi_diss)
            
            while True:
                found_better_neighbour = False
                neighbours = self.get_neighbours(xi)

                for wi_new in neighbours:
                    M11 = 0
                    M10 = 0
                    M01 = 0
                    for j in range(len(xi)):
                        if xi[j] == 1 and wi_new[j] == 1:
                            M11 += 1
                        if xi[j] == 1 and wi_new[j] == 0:
                            M10 += 1
                        if xi[j] == 0 and wi_new[j] == 1:
                            M01 += 1
                    if M11 + M10 + M01 == 0:
                        minM_new = abs(1 -fi_diss)
                    else:                    
                        minM_new = abs(1 - M11 / (M11 + M10 + M01) - fi_diss)
                    if M11 + M01 == m1 and M10 <= m0 and minM_new < minM:
                        wi = wi_new
                        minM = minM_new
                        found_better_neighbour = True
                        
                if not found_better_neighbour:
                    break
            
            Omegai = copy.deepcopy(Wi)
                    
            #REKOMBINACIJA
            cr = 0.25
            ui = []
            for j in range(len(Omegai.x)):
                if np.random.rand() < cr:
                    ui.append(Omegai.x[j])
                else:
                    ui.append(xi[j])
                    
            Ui = SolutionABC(ui, task, True, rng=self.rng)
            Ui.evaluate(task,self.rng)
            
            if Ui.f < population[i].f:
                population[i] = copy.deepcopy(Ui)
                trials[i] = 0
            else:
                #population[i] = xi ostaje
                trials[i] += 1
        
                        
        for i in range(self.food_number):
            if trials[i] >= self.limit:
                population[i] = SolutionABC(task=task, rng=self.rng)
                trials[i] = 0
                if population[i].f < best_f:
                    Gbest, best_f = population[i].x[:], population[i].f
            else:
                if population[i].f <= best_f:
                    Gbest, best_f = population[i].x[:], population[i].f
                
                
        return population, np.asarray([f.f for f in population]), Gbest, best_f, {'trials': trials}
    
    
    def get_neighbours(self, current_solution):

        neighbours = []

        for i in range(len(current_solution)):
            
            neighbour = current_solution[:]

            if neighbour[i] == 0:
                neighbour[i] = 1
            else:
                neighbour[i] = 0

            neighbours.append(neighbour)

        return neighbours
