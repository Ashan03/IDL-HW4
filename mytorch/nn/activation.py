import numpy as np


class Softmax:
    """
    A generic Softmax activation function that can be used for any dimension.
    """
    def __init__(self, dim=-1):
        """
        :param dim: Dimension along which to compute softmax (default: -1, last dimension)
        DO NOT MODIFY
        """
        self.dim = dim

    def forward(self, Z):
        """
        :param Z: Data Z (*) to apply activation function to input Z.
        :return: Output returns the computed output A (*).
        """
        if self.dim > len(Z.shape) or self.dim < -len(Z.shape):
            raise ValueError("Dimension to apply softmax to is greater than the number of dimensions in Z")
        
        # TODO: Implement forward pass
        # Compute the softmax in a numerically stable way
        # Apply it to the dimension specified by the `dim` parameter
        Z_max = np.max(Z, axis=self.dim, keepdims = True)
        Z_shifted = Z - Z_max

        numerator = np.exp(Z_shifted)
        denominator = np.sum(numerator, axis=self.dim, keepdims=True)

        self.A = numerator / denominator
        return self.A
    

    def backward(self, dLdA):
        """
        :param dLdA: Gradient of loss wrt output
        :return: Gradient of loss with respect to activation input
        """
        # TODO: Implement backward pass
        
        # Get the shape of the input
        shape = self.A.shape
        
        # Find the dimension along which softmax was applied
        C = shape[self.dim]
           
        # Reshape input to 2D
        if len(shape) > 2:
            self.A = np.moveaxis(self.A, self.dim, -1)
            dLdA_transposed = np.moveaxis(dLdA, self.dim, -1)

            transposed_shape = self.A.shape
            self.A = self.A.reshape(-1, transposed_shape[-1])
            dLdA = dLdA_transposed.reshape(-1, transposed_shape[-1])

        term1 = dLdA * self.A
        term2 = self.A * np.sum(term1, axis=1, keepdims=True)

        dLdZ = term1 - term2

        if len(shape) > 2:
            dLdZ = dLdZ.reshape(transposed_shape)
            
            dLdZ = np.moveaxis(dLdZ, -1, self.dim)
            
            self.A = self.A.reshape(transposed_shape)
            self.A = np.moveaxis(self.A, -1, self.dim)

        return dLdZ       
 

    