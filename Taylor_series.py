import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Callable, Iterator



'''
Functions currently added
'''
FunctionMap = {
    'exp': np.exp,
    'sin': math.sin
}

'''
Taylor series functions
'''
class TaylorFunctions:
    def sin(x, n):
        sum = 0
        for i in range(0, n+1):
            sum += ((-1)**i) * (x**(2*i + 1)) / math.factorial(2*i+1)
        return sum

    def exp(x, n):
        sum = 0
        for i in range(0, n+1):
            sum += (x**i) / math.factorial(i)
        return sum


'''
Error functions
'''
def error_percent(estimate, actual):
    return (estimate - actual) / actual * 100

def error_absolute(estimate, actual):
    return abs(estimate - actual)



class TaylorSeries:
    def __init__(self, function: Callable, error_function: Callable=None) -> None:
        # Save name of the function            
        self.function_name = function.__name__
        self.function = function
        self.error_function = error_function

    def calculateResults(self, x: float, n: int) -> None:
        # Calculate results
        self.results = pd.DataFrame(
            {self.function_name: self.solution_array_generator(self.function, x, n)}
            )

        # Calculate error if function is provided
        if self.error_function is not None:
            actual_value = FunctionMap[self.function_name](x)
            self.results['error'] = self.results[self.function_name].apply(
                lambda i: self.error_function(i, actual_value)
                )

    def solution_array_generator(self, func: Callable, x: float, depth: int) -> Iterator[float]:
        '''
        Series generation functions
        '''
        for iteration in range(depth):
            yield func(x, iteration)

    def displayResults(self) -> None:
        print(self.results)
    
    def plotResults(self) -> None:
        # Create subplot for taylor series
        ax1 = plt.subplot()
        # Set the x axis ticks to match the n integers
        ax1.set_xticks(self.results.index)
        # Plot the Taylor series (column 1)
        s1 = self.results.iloc[:,0]
        plt.plot(self.results.index, s1,'b-')
        plt.xlabel('n')
        plt.ylabel(self.function_name, color='b')

        if 'error' in self.results:
            # Create a copy of the first plot object
            ax2 = ax1.twinx()
            # Plot the errors (column 2)
            s2 = self.results.iloc[:,1]
            ax2.plot(self.results.index, s2, 'r-')
            plt.ylabel('error', color='r')

        # Display the plot
        plt.show()
    
    def plotSeries(self, x_min: float, x_max: float, x_step: float, n: int) -> None:
        '''
        Plot the Taylor series approximation to n iterations over the range [x_min, x_max] with a step size of x_step
        '''
        x_axis = np.linspace(x_min, x_max, math.ceil((x_max-x_min)/x_step))

        y = np.array([self.function(x, n) for x in x_axis])

        plt.plot(x_axis, y)
        plt.title(f'{self.function_name} with n = {n}')
        plt.show()
            


if __name__ == "__main__":

    sin_taylor = TaylorSeries(TaylorFunctions.sin, error_function=error_absolute)
    sin_taylor.calculateResults(np.pi, 10)
    sin_taylor.plotSeries(-40, 40, 0.1, 10)
    sin_taylor.plotSeries(-40, 40, 0.1, 84)

    # sin_taylor.plotResults()

    # e_taylor = TaylorSeries(TaylorFunctions.exp, error_function=error_absolute)
    # e_taylor.calculateResults(-3, 20)
    # e_taylor.plotSeries()

