# Scaling Dual-based Protocol

## Background

Neural networks are composed with layers. Common layers are either linear (convolution, fully-connected) or non-linear functions (ReLU, max pooling).

For a linear function $f(x)$, we have:
1. $f(m x) = m f(x)$ for any $m$.
2. $f(x + r) = f(x) + f(r)$ for any $r$.

Non-linear function $f(x)$ in neural networks, such as ReLU and max, are also scalable for positive scaling factors. It means that:
1. $f(m x) = m f(x)$ for any positive $m$.

## Idea

Key idea: we can use a random scaling tensor $m$ over the input data tensor $x$ to compute the neural network layer by layer. And we can scale the result back to the correct data.

Invarient : each layer $i$ inputs $x_i \odot m_{i-1}$ and outputs $x_{i+1} \odot m_i$.

## Protocol

### Protocol for layer $i$ (exampled with a linear layer of weight $W_i$):

- Online phase (compute $W_i x_i \odot m_i$):

    1. Client sends $x_i \odot m_{i-1} - r_i$ to server.

    2. Server rescales it and gets $x_i - r'_i = x_i - r_i \oslash m_{i-1}$.

    3. Server performs computation on $x_i - r'_i$ and gets $W_i (x_i - r'_i)$.

    4. Server scales it with a positive $m_i$ and sends results $W_i (x_i - r'_i) \odot m_i $ back to client.
    
    5. Client removes $W_i r'_i \odot m_i$ using a precomputed data from the result and gets $x_{i+1} \odot m_i = W_i x_i \odot m_i$.

- Offline phase (compute $W_i r'_i \odot m_i$):

    Similar to the online phase, but the client sends $r_i$ to server instead of $x_i - r_i$.

### ReLU layer

The client run ReLU locally on $x_i \odot m_i$, because ReLU is a scalable function for positive scaling factors.

### Max-pooling layer

When generating the tensor $m_i$, the server makes sure that entries in the same pooling region are the same.

It requires the pooling stride to be no smaller than the pooling size. This is satisfied by most CNNs, because it is a common practice in neural networks.

## Security

This protocol guarantes the security of both the client's data and the server's network weights.
