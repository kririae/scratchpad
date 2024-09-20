#import "@preview/physica:0.9.3": *

#set page(width: 148mm, height: auto)
#set par(justify: true)

= Introduction to Simulation Optimization

I'd like to start with differentiable simulation in the context of Computer
Graphics because it allows me to dive into the topic.

The issue is, to differentiate what?

Differentiable fluid simulation:

- #link("https://github.com/tum-pbs/PhiFlow")[PhiFlow], combination and JIT. A framework.
- #link("https://github.com/lettucecfd/lettuce")[Lettuce], LBM with TensorFlow.

== Adjoint Method

As for a physical process, we want to compute
$
  dv(J, bold(u)) = bold(g)^top B + pdv(J, bold(u)),
$
where $bold(g)$ is can be analytically computed, while $B$ is a matrix, such that
$
  B_(i j) = pdv(q_i, u_j),
$
which, is the Jacobian of the evolution process. We know that $B$ can be hard to compute because it is a $n$ by $n$ matrix.

One great observation about this process, is that $bold(g)^top B$ can be simplified. Suppose the evolution process follows $E(bold(q), bold(u)) = 0$, then $A B = C$.

We have instead solve for $A^top bold(s) = bold(g)$ and replace $bold(g)^top B$ by $bold(s)^top C$.

$
  bold(s)^top C = bold(s)^top A B = g^top B,
$
which recovers the original equation.


So, the original:
1. Derive $A$
2. Derive $C$
3. Derive $g$
4. Solve for $A B = C$ (this is the hard part)
5. Multiply $g^top C$

Here's
1. Dervie $A$
2. Derive $C$
3. Derive $g$
4. Solve for $A^top bold(s) = bold(g)$ (now this becomes easy)
5. Multiply $s^top C$

So... let's consider a simple example.
```cpp
float f(float x, float y) {
  x = y * y;
  return x;
}
```

All steps can be viewed as a operation on the whole states.
We define the loss function on only the on the final state, i.e., the result of this function, so
$
  u = (x, y), "in the parameter"
$
$
  q = (x, y), "in the return type"
$
pj/pq = [1, 0], pj/pu = 0

so $bold(g) = [1, 0]$, i.e., final states only depends on the result of $x$.
$E([x, y], [x', y']) = x' - y y = 0$.
Then $
  A = [1, 0]
$
and
$
  C = [0, 2y]
$
we solves for

== 对偶方法

对偶方法起源于这么一个非常简单的技巧。对于一个优化任务
$
  bold(u) = limits(arg min)_(bold(u))[J(bold(q), bold(u))], "w.r.t." E(bold(q), bold(u)) = 0,
$
其中 $bold(u)$ 为我们的可调参数，而 $bold(p)$ 为系统状态。对于这个优化任务，我们希望能计算其目标函数对于可调参数的导数，即
$
  dv(J, bold(u)) = bold(g)^top B + pdv(J, bold(u)),
$
