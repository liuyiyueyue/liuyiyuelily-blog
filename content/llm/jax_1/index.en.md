---
title: "JAX (Part 1): Thinking in Pure Functions"
date: 2026-03-13
tags: ["llm", "jax", "training", "programming languages", "compiler"]
---


### Pure functions

**Functional Programming in JAX**

JAX requires to express all computations as **pure functions**, where any state change is represented explicitly through function arguments.

Not allowed

  * Modifying or relying on global variables
  * In-place mutation of arrays
  * Hidden or implicit state
  * Generating random numbers without explicitly passing a PRNG key

Allowed

  * Functions that depend only on their input arguments
  * Outputs that are fully determined by inputs
  * No side effects that modify the external world

**Why?**

In large-scale ML systems, hidden mutable state often becomes a liability.
Object-oriented abstractions tend to hide parameters and state inside objects, which makes program transformations such as JIT compilation, autodiff, parallelization, and vectorization harder for the compiler to reason about.

By contrast, functional programs make all dependencies explicit through function arguments, allowing systems like JAX to apply aggressive optimizations and transformations in a principled way.

Example 1:
```python
a = 1

def stateful(x):     # If 'a' changes, the result also changes
    return x + a

def stateless(a, x): # 'a' is now passed explicitly
    return a + x

result1 = stateful(x)
result2 = stateless(a, x)
```

Example 2:
```python
x[0] = 10             # Not allowed in JAX: NumPy-style in-place mutation
x = x.at[0].set(10)   # Allowed in JAX: Functional update returns a new array without mutating the old one
```

References: https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html#pure-functions


### From PyTorch to JAX

The key challenge when moving from PyTorch to JAX is learning how to **convert stateful code into stateless functions**.

In PyTorch, state is often hidden inside objects (e.g., model parameters stored in `nn.Module`). In JAX, however, all state must be passed explicitly as function inputs. This shift requires restructuring code so that computations are expressed as pure functions.

A practical way to approach this transition:

* Identify the inputs and outputs.
  Analyze what data goes into the computation and what results it produces.

* Understand what changes during execution.
  Determine which values are being updated or generated as the function runs.

* Track the state involved in each function call.
  Make sure you understand which pieces of state the computation depends on.

* Extract hidden state from objects.
  Take the state that PyTorch stores inside objects (such as model parameters or buffers) and turn them into explicit arguments to a pure function.

```python
PyTorch                                      JAX
(Stateful: hidden state)                     (Functional: explicit state)
─────────────────────────────────────        ────────────────────────────────────

# Create model object                        # Create model object
# Parameters live inside the model           # Parameters are created explicitly
model = MyMod(arg_model)                     model = Model(arg_model)
                                             params = model.init(key)

# Create optimizer                           # Create optimizer
# Optimizer reads params from model          # Optimizer state is explicit
opt = MyOpt(model.params, arg_opt)           opt = MyOpt(arg_opt, params)
                                             opt_state = opt.init(params)

# Training loop                              # Training loop
for x, target in data:                       for x, target in data:

    # Forward pass                           # Define pure loss function
    # Model implicitly uses internal params  # Parameters are passed explicitly
    y = model(x)                             def loss_func(params, x, target):
                                                 y = model.apply(params, x)

    # Compute loss                           # Compute loss
    loss = loss_f(y, target)                     loss = loss_f(y, target)
                                                 return loss

    # Backpropagation                        # Backpropagation
    # Autograd accumulates gradients         # Transform pure function to get grads
    loss.backward()                          loss, grads = jax.value_and_grad(loss_func)(
                                                 params, x, target
                                             )

    # Optimizer step                         # Optimizer step
    # Mutates parameters in-place            # Returns new params and optimizer state
    opt.step()                               opt_state, params = opt.step(
                                                 grads, opt_state, params
                                             )
```

