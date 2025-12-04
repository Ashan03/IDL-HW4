import numpy as np
from .activation import Softmax

class ScaledDotProductAttention:
    """
    Scaled Dot Product Attention
    """ 
    def __init__(self):
        '''
        Initialize the ScaledDotProductAttention class.
        '''
        # Initialize your softmax layer
        # What dimension should you pass to the softmax constructor?
        self.eps = 1e10 # DO NOT MODIFY
        self.softmax = Softmax(dim=-1)
    
    def forward(self, Q, K, V, mask=None):
        """
        :param Q: Query matrix of shape (N, ..., H, L, E) where L is target sequence length
        :param K: Key matrix of shape (N, ..., H, S, E) where S is source sequence length
        :param V: Value matrix of shape (N, ..., H, S, Ev) where Ev is value dimension
        :param mask: Boolean mask matrix of shape (N, ..., H, L, S) or broadcastable shape where 1/True indicates a position to ignore
        :return: Output matrix of shape (N, ..., H, L, Ev)
        """
        # TODO: Implement forward pass
        self.Q = Q.astype(float)
        self.K = K.astype(float)
        self.V = V.astype(float)
        self.mask = mask

        d_k = Q.shape[-1]
        K_T = np.swapaxes(K, -1, -2)

        Z = np.matmul(self.Q, K_T) / np.sqrt(d_k)
        # Calculate attention scores: (N, ..., H, L, S)
        # (N, ..., H, L, E) @ (N, ..., H, E, S) -> (N, ..., H, L, S)
        
        # Apply mask before softmax if provided
        # If mask is not None, add -self.eps to the attention scores for positions to ignore
        if self.mask is not None:
            Z = np.where(self.mask, Z - self.eps, Z)
        self.Z = Z

        # Compute attention scores: Apply softmax along S dimension (N, ..., H, L, S)
        self.attention_scores = self.softmax.forward(Z)

        # Calculate output: (N, ..., H, L, Ev)
        # (N, ..., H, L, S) @ (N, ..., H, S, Ev) -> (N, ..., H, L, Ev) 
        output = np.matmul(self.attention_scores, V)

        return output
        # raise NotImplementedError
    
    def backward(self, d_output):
        """
        :param d_output: Gradient of loss wrt output of shape (N, ..., H, L, Ev)
        :return: Gradient of loss wrt input Q, K, V
        """
        # TODO: Implement backward pass
        attention_scores_T = np.swapaxes(self.attention_scores, -1, -2)
        # Calculate gradients for V: (N, ..., H, S, Ev)
        # (N, ..., H, L, S) @ (N, ..., H, S, Ev) -> (N, ..., H, L, Ev) 
        # Use the transpose of stored softmax output to swap last two dimensions   
        d_V = np.matmul(np.swapaxes(self.attention_scores, -1, -2), d_output)

        # Calculate gradients for attention scores
        # (N, ..., H, L, Ev) @ (N, ..., H, Ev, S) -> (N, ..., H, L, S)
        V_T = np.swapaxes(self.V, -1, -2)

        d_A = np.matmul(d_output, V_T)
        d_Z = self.softmax.backward(d_A)
        if self.mask is not None:
            d_Z = np.where(self.mask, 0.0, d_Z)
        
        
        # Scale gradients by sqrt(d_k)
        d_k = self.Q.shape[-1]
        # Calculate gradients for Q and K
        # (N, ..., H, L, S) @ (N, ..., H, S, E) -> (N, ..., H, L, E)   
        d_Q = np.matmul(d_Z, self.K) / np.sqrt(d_k)
        # (N, ..., H, L, S) @ (N, ..., H, L, E) -> (N, ..., H, S, E)
        d_K = np.matmul(np.swapaxes(d_Z, -1, -2), self.Q) / np.sqrt(d_k)

        
        return d_Q, d_K, d_V
        # raise NotImplementedError

