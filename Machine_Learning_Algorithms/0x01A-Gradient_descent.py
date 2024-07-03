kclass Solution:
    def get_minimizer(self, iterations: int, learning_rate: float, init: int) -> float:
        """
        Perform Gradient Descent to minimize the function f(x) = x^2.

        Gradient Descent is an optimization technique widely used in machine learning
        for minimizing the cost or loss function and finding optimal parameters of a model.
        For the function f(x) = x^2, the global minimizer is at x = 0. This function 
        implements an iterative approximation algorithm using Gradient Descent.

        Args:
        - iterations (int): Number of iterations to perform Gradient Descent. Must be >= 0.
        - learning_rate (float): Learning rate for Gradient Descent. Must satisfy 0 < learning_rate < 1.
        - init (int): Initial guess for the minimizer. Should not be 0.

        Returns:
        float: The value of x that globally minimizes the function f(x) = x^2, rounded to 5 decimal places.

        Example:
        >>> solution = Solution()
        >>> solution.get_minimizer(0, 0.01, 5)
        5
        >>> solution.get_minimizer(10, 0.01, 5)
        4.08536
        """
        value = init

        for _ in range(iterations):
            derivative = 2 * value
            value = value - learning_rate * derivative
        
        return round(value, 5)

