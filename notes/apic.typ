#import "@preview/physica:0.9.3": *

#set page(width: 148mm, height: auto)

#let foreground = rgb("1F2430")
#let background = rgb("FFFFFF")
#let dark_mode = false
#if dark_mode {
  foreground = rgb("FDFDFD")
  background = rgb("1F2430")
}

#set page(fill: background)
#set text(foreground)
#set par(justify: true)
#show link: set text(fill: rgb("34b8ff"))

= Introduction to Material Point Method

== Why PIC does not work

#line(length: 100%, stroke: foreground)

*Particles* have a total angular momentum:
$
  bold(L)_"tot"^p = sum_p bold(x)_p times (m_p bold(v)_p),
$
while the total angular momentum on the grid is
$
  bold(L)_"tot"^g = sum_i bold(x)_i times (m_i bold(v)_i).
$
We have the following transformations defined between particles and grid:
$
  cases(
    m_i = sum_p m_p N_i (bold(x)_p),
    m_i bold(v)_i = sum_p m_p bold(v)_p N_i (bold(x)_p),
    bold(v)_p = sum_i bold(v)_i N_i (bold(x)_p),
  ).
$
As for particle to grid, we have
$
  bold(L)_"tot"^g
  & = sum_i bold(x)_i times sum_p m_p bold(v)_p N_i (bold(x)_p) \
  & = sum_p (sum_i bold(x)_i N_i (bold(x)_p)) times m_p bold(v)_p \
  & = sum_p bold(x)_p times m_p bold(v)_p = bold(L)_"tot"^p.
$
But in the inverse, using the fact that $N_p (bold(x)_i) = N_i (bold(x)_p)$,
$
  bold(L)_"tot"^p
  & = sum_p bold(x)_p times m_p sum_i bold(v)_i N_i (bold(x)_p) \
  & = sum_i sum_p [bold(x)_p N_p (bold(x)_i) m_p] times bold(v)_i,
$
so we are expecting an approximation that
$
  m_i bold(x)_i approx sum_p bold(x)_p N_p (bold(x)_i) m_p.
$

This holds when $bold(x)_p = bold(x)_i$ is a constant. When particles lie in a small kernel, this can be regarded as an approximation.

== From PIC to RPIC

#line(length: 100%, stroke: foreground)

We can calculate the total angular momentum lost upon performing grid to particle transformation.
Through a trivial translation, compensating the $bold(x)_p$ to $bold(x)_i$ in each particle, we can obtain the following momentum loss for a specific particle $p$:
$ Delta bold(L)_p = sum_i (bold(x)_i - bold(x)_p) times m_p N_i (bold(x)_p) bold(v)_i, $
while
$
  bold(L)_"tot"^p + sum_p Delta bold(L)_p
  & = sum_i [sum_p bold(x)_i bold(v)_i N_i (bold(x)_p)] times m_p \
  & = sum_i bold(x)_i times m_i bold(v)_i = bold(L)_"tot"^g.
$
So we are expecting that, for a specific particle $p$, an additional term $Delta bold(L)_p$ can be added to the total angular momentum.

But how is this achievable? Although $Delta bold(L)_p$ is directly computable, it cannot be directly cast onto $Delta bold(v)$, so we need some heuristics to approximate it.
The physical intuition behind $Delta L_p$ is that a system's total angular momentum can be decomposed into _centroid_ and _relative_ angular momentum, where the relative angular momentum is always lost.
Then... why not add back the angular momentum by adding an additional term to the particle? Just like ```cpp
struct Particle {
  Vec2 x, v;  // Centroid position and velocity
  Real m, w;  // Mass and local angular momentum
}; /* with SoA */
```

Although $omega$ can be trivially transferred onto the grid to reflect the local transformation on the particle, it is quite hard to project the grid velocities back to the particle.
I should *emphasize* how to build the mental process here. Consider only $1$ particle in a 2D scenario, and the grid is a $3 times 3$ grid.
Upon the particle to grid process, the velocity is transferred with
$
  bold(v)_i = bold(v)_p + bold(C) (bold(x)_i - bold(x)_p),
$
where $bold(C)$ is a #link("https://en.wikipedia.org/wiki/Skew-symmetric_matrix")[skew matrix] constructed from angular velocity. This is a trivial process, as described by the rigid rotation.

The inverse process involves a projection, from higher degree of freedom on the grid, to a lower DoF on the particle. The easiest approach is to calculate the weighted average of the velocities on the grid, which is what the PIC does.
But this is not enough to compensate for the loss of local rigid angular momentum.
To account for the local angular velocity, we calculate
$
  bold(L) = sum_i (bold(x)_i - bold(x)_p) times m_i bold(v)'_i,
$
multiplied by the inverse rigid's inertia tensor, giving the updated local angular velocity $omega'_p$ from $bold(v)'_i$.

== From RPIC to APIC

#line(length: 100%, stroke: foreground)

We are now aware that the local DoF, i.e., the local velocity and angular velocity, is projected from a higher DoF presented on the grid.
This is the key to the APIC method.
*How about adding more DoF to the particle, such that this projection can be more representative?*

This is a generalization to the previously mentioned $bold(C)$ matrix, which is the matrix form of a cross product.
Instead of performing solely a rotation, we can perform a general affine transformation, say, $bold(A) in RR^(2 times 2)$ in $d = 2$.

Consider the two directions, from particle to grid and from grid to particle.
The former one is quite trivial, just replacing $bold(C)$ with a more generalized matrix $A$.
In a stricter sense, $bold(A)$ can be decomposed into $bold(A) = bold(B) bold(K)^(-1)$, where $bold(K)$ is the generalized inertia tensor (defined on the particle)
$
  bold(K)_p = sum_i N_i (bold(x)_p) (bold(x)_i - bold(x)_p) (bold(x)_i - bold(x)_p)^top,
$
which has a surprisingly simple form with $N_i$ as quadratic B-spline kernel, $bold(K)_p = 0.25 bold(I)$.
Meanwhile
$
  bold(B)_p = sum_i N_i bold(v)'_i (bold(x)_i - bold(x)_p)^top.
$

How are these matrices derived? There's no explannation in the original paper, but the derivation is indeed quite simple from a *regression* perspective.
Define the following regression matrix:
$
  bold(C)_p & = op("argmin", limits: #true)_bold(C)_p sum_i N_i (bold(x)_i - bold(x)_p) (
    bold(C)_p (bold(x)_i - bold(x)_p) - bold(v)_p
  )^2 \
  & arrow.double pdv(, bold(C)_p) sum_i N_i (bold(x)_i - bold(x)_p) (
    bold(C)_p (bold(x)_i - bold(x)_p) - bold(v)_p
  )^2 = 0 \
  & arrow.double pdv(, bold(C)_p) sum_i N_i (bold(C)_p bold(d)_i - bold(v)_p)^2 = 0 \
  & arrow.double pdv(, bold(C)_p) sum_i N_i (
    bold(d)_i^top bold(C)_p^top bold(C)_p bold(d)_i - 2 bold(d)_i^top bold(C)_p^top bold(v)_i
  ) = 0 \
  & arrow.double sum_i N_i (
    2 bold(C)_p bold(d)_i bold(d)_i^top
    - 2 bold(v)_i bold(d)_i^top
  ) = 0 \
  & arrow.double bold(C)_p sum_i N_i bold(d)_i bold(d)_i^top = sum_i N_i bold(v)_i bold(d)_i^top \
  & "where" bold(C)_p = bold(B)_p bold(K)_p^(-1).
$
For the formula used, refer to #link("https://www.math.uwaterloo.ca/~hwolkowi/matrixcookbook.pdf")[The Matrix Cookbook].
This do indicate a important factor, the $bold(C)_p$ is a least-square regression of the local velocities on the grid, projected onto the particle.

== Push-forward and Pull-back

#line(length: 100%, stroke: foreground)

It can be quite hard to understand the APIC method without deeply understanding the push-forward and pull-back operations.
The operations are simply defined as a transformation of a function from one space to another.

For example, we have a function $f: Omega_0 arrow RR$, where $Omega_0$ is the initial space, i.e., material space. You can imagine this as the initial position of particles, or a initial set of the object's coverage.
Performing a _push-forward_ on this function is to transfer $f$ to $g: Omega_t arrow RR$, where $Omega_t$ is the world space, i.e., the deformed material space at time $t$.
Vise versa, _pull-back_ is the inverse operation from $g$ to $f$.

The hard-to-understand part is, although we do have a quite general definition of the push-forward and pull-back, they only work in the context of integration.

== Nodal Forces

#line(length: 100%, stroke: foreground)

Nodal forces are quite non-trivial to calculate, since the forces are applied on the grid, not on the particles.

In a general FEM context, the nodal forces are calculated by
$
  bold(f)_p = -pdv(U, bold(x)_p),
$
where $U$ is the system potential; $bold(x)_i$ is the corresponding position of the node $i$. Note that $bold(f)_p$ here is not force density.

System potential is defined as
$
  U = integral_(Omega_0) Psi (bold(F)) dd(V),
$
where $Omega_0$ is the initial material space, $Psi$ is the strain energy density, and $bold(F)$ is the deformation gradient tensor. Without further discretization, this casts to
$
  -pdv(U, bold(x)_p) = - integral_(Omega_0) (pdv(Psi, bold(F)) : pdv(bold(F), bold(x)_p)) dd(V).
$

But what really is $partial bold(F) \/ partial bold(x)_p$? Let's delve into this.

A common discretization is Barycentric coordinates:
$
  bold(x) = bold(x)_0 + sum_i bold(alpha)_i (bold(x)_i - bold(x)_0),
$
where $bold(alpha)_i$ is component of the barycentric coordinate; $bold(x)_i$ are the vertices of the element in either world space or material space.
This is to say, world space and material space share the same barycentric coordinates.

In calculating deformation gradient tensor, this becomes
$
  bold(F)_(i j)
  & = pdv(bold(x)_i, bold(X)_j) = pdv((bold(x)_0 + bold(D) bold(alpha))_i, bold(X)_j) \
  & = pdv((bold(D) bold(D)_m^(-1) (bold(X) - bold(X)_0))_i, bold(X)_j) \
  & = bold(D)_(i k) (bold(D)_m^(-1))_(k j)
$
where $bold(D)$ and $bold(D)_m$ are $RR^(d times d)$ matrices representing the current and initial configurations of the elements respectively.
They are constructed from the positions like this (for $d = 2$):
$
  bold(D) = mat(bold(x)_1 - bold(x)_0 | bold(x)_2 - bold(x)_0), bold(D)_m = mat(bold(X)_1 - bold(X)_0 | bold(X)_2 - bold(X)_0).
$

Based on this, energy potential is discretized to
$
  -pdv(U, bold(x)_(p i))
  & = - sum_e V_e^((0)) (bold(P)_e : pdv(bold(F)_e, bold(x)_(p i))) \
$
where $bold(x)_p$ lies exactly on the vertices of the elements.
Computing $partial bold(F)_e \/ partial bold(x)_(p i)$ is highly redundant, refer to the detailed computation as presented in #link("https://graphics.pixar.com/library/DynamicDeformablesSiggraph2020/paper.pdf")[Dynamic Deformables].

Luckily, in the original MPM, we have a much simpler way to calculate the nodal forces (although tricky), MLS-MPM will make it even simpler.
One thing to notice, in MPM's context, the node positions $bold(x)_i$ are samples in $Omega_t$, which is related as
$
  bold(x)_p = sum_i bold(x)_i N_i (bold(x)_p),
$
where $bold(x)_p$ are the particles' positions, and $N_i$ are the kernel functions.

Here comes the tricky part.
To actually calculate the gradient of $bold(F)$, we recall the update rule of $bold(F)$:
$
  pdv(bold(F), t) = (nabla bold(v)) bold(F),
$
which thus becomes (a implicit push-forward is performed)
$
  bold(F)_p^(n+1) = (bold(I) + Delta t sum_i bold(v)_i nabla_bold(x)^top N_i (bold(x)_p)) bold(F)_p^n,
$
where $nabla_bold(x)^top N_i (bold(x)_p)$ is the transpose of the tensor $nabla_bold(x) N_i (bold(x)_p)$.
We can replace $Delta t bold(v)_i$ with $bold(x)^(n+1)_i - bold(x)^(n)_i$ , resulting in
$
  bold(F)_p^(n+1) = (bold(I) + sum_i (bold(x)^(n+1)_i - bold(x)_i^n) nabla_bold(x)^top N_i (bold(x)_p)) bold(F)_p^n.
$
A great observation is that, these two equations holds for arbitrary $bold(v)_i$ on grid, not limited to the current configuration. If we happend to *construct* a set of $bold(v)_i$ such that the particle, thus the $bold(F)_p$ land on a specific location $bold(x)$, we can obtain $bold(F)$ at that location. The approximation
$
  bold(F)_p (bold(x)) = (bold(I) + sum_i (bold(x)_i - bold(x)_i^n) nabla_bold(x)^top N_i (bold(x)_p)) bold(F)_p^n
$
is quite accurate near $bold(x)_p$, where $bold(x)_i$ can be chosen under the constraint that
$
  bold(x) = bold(x)_p + Delta t sum_i bold(v)_i N_i (bold(x)_p).
$
Taking the derivative near the node $i$, we have
$
  pdv(bold(F)_p, bold(x)_i) = pdv((sum_j (bold(x)_j - bold(x)_j^n) nabla_bold(x)^top N_j (bold(x))), bold(x)_i) bold(F)_p^n,
$
where we choose only the $i$-th node to be adjusted, lefting $bold(x)_j$ where $i = j$ as the only variable, the equation becomes
$
  pdv(bold(F)_p, bold(x)_i) = nabla_bold(x)^top N_i (bold(x)) bold(F)_p^n,
$
This finaaaaaaaly gives us the force, as
$
  bold(f)_i
  & = -pdv(U, bold(x)_i) = - sum_p V_p Psi(bold(F)_p) \
  & = - sum_p V_p (pdv(Psi, bold(F)_p) : pdv(bold(F)_p, bold(x)_i)) \
  & = - sum_p V_p bold(P) bold(F)_p^(n top) nabla_bold(x) N_i (bold(x)_p),
$
where $bold(P)$ is the first Piola-Kirchhoff stress tensor.

This loooong approximation works no better than this direct loosy approximation
$
  bold(F)(bold(x)) = bold(F)_p N_p (bold(x)), "where" |bold(x) - bold(x)_p| "is small enough",
$
thus
$
  pdv(bold(F), bold(x)_i) = nabla_bold(x)^top N_p (bold(x)_i) bold(F)_p = nabla_bold(x)^top N_i (bold(x)_p) bold(F)_p.
$

#block(fill: rgb("#34b8ff66"), inset: 1em, radius: 3pt)[
  Force calculation is always tricky, both in traditional FEM discretization and MPM,
  while the hard part is to calculate $nabla_bold(x) bold(F)$.
  Spatial gradient calculation is *always* related to discretization method, in MPM,
  the trick is to form a local approximation of $bold(F)$ near particles, which is done by the deformation gradient update rule. MLS-MPM's contribution is to further simplify this process.
]

The turnaround is in with, we use APIC in graphics, where $bold(C)_p$ has extra physical meaning.
But that's a story for later.

== MLS-MPM and its Simplification

#line(length: 100%, stroke: foreground)

Remember the original Galerkin method, where test functions are used for
+ Constructing the weak form, where the test functions are used to multiply the PDEs.
+ Accepting the derivative through _integral by parts_.
+ Compose the solution by linear combination of the test functions.

Consider the MPM's governing equation after time discretization,
$
  1 / (Delta t) integral_Omega_t^n bold(q)_alpha rho (bold(v)_alpha^(n+1) - bold(v)_alpha^(n)) dd(bold(x)) \
  = integral_(partial Omega_t^n) bold(q)_alpha bold(t)_alpha dd(bold(s)) - integral_Omega_t^n bold(q)_(alpha, beta) bold(sigma)_(alpha beta) dd(bold(x)),
$
where $alpha$ is the component index, $bold(t)_alpha$ is the traction field on the surface, $bold(q)_(alpha, beta)$ is the gradient of the test function. All variables are implicitly functions on the world space. Gradient of the stress tensor is already transferred onto the test function.

Traditional MPM express the solution field as a linear combination of the B-spline basis functions, say, quadratic B-spline kernel $N_i$.

MLS-MPM choose to use the MLS basis functions, which are quite similar to _Kernel method_. Express arbitrary field $f$ with
$
  f(bold(z)) = bold(P)^top (bold(z) - bold(x)) bold(c)(bold(x)),
$
where $bold(z)$ are the position to evaluate the field, $bold(x)$ is a nearby point (not necessarily reside on the samples), $bold(P)$ is a polynomial basis, $bold(c)$ is a coefficient vector.
The coefficient vector is further obtained by minimizing
$
  J(bold(c)) = sum_i xi_i (bold(x)) (bold(P)^top (bold(x)_i - bold(x)) bold(c)(bold(x)) - f_i)^2,
$
where $xi_i$ are weight functions, $f_i$ are samples of the field $f$.
Again, let us paraphrase the variables involved. $bold(z)$ is the current evaluation point, $bold(x)$ is a point nearby selected intentionally, $bold(c)$ is a coefficient vector on $bold(x)$, $bold(x)_i$ are the sample points, following the order
$
  f_i "on" bold(x)_i arrow.double bold(c)(bold(x)) arrow.double f(bold(z)) "composed".
$

This gives a quite familar approximation,
$
  f(bold(z)) = sum_i xi_i (bold(x)) bold(P)^top (bold(z) - bold(x)) bold(M)^(-1)(bold(x)) bold(P)(
    bold(x)_i - bold(x)
  ) f_i,
$
where $bold(M)(x) = sum_i xi_i (bold(x)) bold(P) (bold(x)_i - bold(x)) bold(P)^top (bold(x)_i - bold(x))$.
Let $bold(z) = bold(x)_i$ and replace the index $i$ with $p$, we have
$
  f(bold(x)_i) = sum_p xi_p (bold(x)) bold(P)^top (bold(x)_i - bold(x)) bold(M)^(-1)(bold(x)) bold(P)(
    bold(x)_p - bold(x)
  ) f_p,
$
which, is further expressed with
$
  f(bold(x)_i) = sum_p Phi_p (bold(x)_i) f_p,
$
where basis function $Phi_p$ is defined as
$
  Phi_p (bold(z)) = xi_p (bold(x)) bold(P)^top (bold(z) - bold(x)) bold(M)^(-1)(bold(x)) bold(P)(bold(x)_p - bold(x)).
$
Note that although we use $p$ as the index, this basis function can also be defined on grid nodes, as they don't differ in essence.
We do still have the freedom to choose a $bold(x)$ near $bold(x)_p$.

Note that $f(bold(x))$ can also be expressed as $f(bold(x)) = bold(M)^(-1)(bold(x)) bold(b)(bold(x))$, where
$
  bold(b)(bold(x)) = sum_i xi_i bold(P)(bold(x)_i - bold(x)) f_i.
$

In case of $bold(P)(bold(x)) = [1, bold(x)^top]$, this casts to
$
  mat(delim: "[", bold(v)_p; ?) = sum_i N_i (bold(x)_p) bold(v)_i mat(delim: "[", 1; (bold(x)_i - bold(x)_p)^top).
$

Wait, what is the "?", did we just derive the $bold(B)_p$ in APIC? Remember that $bold(C)_p$ is $bold(B)_p (bold(M)_p)^(-1)$ while $bold(M)_p$ is a diagonal matrix "by chance", kind of.

This do also alleviate the problem of calculating the gradient of test function, consider
$
  grad bold(v)(bold(x))
  & = sum_i bold(v)_i grad Phi_i (bold(x)) \
  & = sum_i bold(v)_i N_i (bold(x)_p) grad bold(P)^top (bold(x) - bold(x)_p) bold(M)^(-1)(bold(x)_p) bold(P)(
    bold(x)_i - bold(x)_p
  ) \
  & = sum_i bold(v)_i N_i (
    bold(x)_p
  ) mat(delim: "[", 0; 1; 1; 1) mat(delim: "[", 1, 0, 0, 0; 0, 4, 0, 0; 0, 0, 4, 0; 0, 0, 0, 4) mat(delim: "[", 1; (bold(x)_i - bold(x)_p)^top) \
  & = 4 sum_i bold(v)_i N_i (bold(x)_p) (bold(x)_i - bold(x)_p)^top = bold(C)_p.
$

#block(fill: rgb("#34b8ff66"), inset: 1em, radius: 3pt)[
  Through this chapter, you might have noticed that, MPM comes with a lot of tricks to simplify the calculation.

  1. Mass lumping, where non-diagonal terms are sumed up, to take away the non-locality.
  2. Grid-based structure with B-spline kernel where the generalized inertia tensor is magically diagonal.
  3. MLS basis function's polynomial, refer to "A new implementation of the element free Galerkin method".

  All of these magic builds the MPM into a quite simple and efficient method.
]

== Derivations of Weakly Compressible MPM

#line(length: 100%, stroke: foreground)

When simulating fluid with MPM, one problem occurs.
As we have seen already, forces should be applied on the grid instead of particles.

We adopt this process into the APIC particle to grid transfer, i.e.,
$
  m bold(v)_i = sum_p m_p bold(v)_p N_i (bold(x)_p) + m_p N_i (bold(x)_p) bold(C)_p (bold(x)_i - bold(x)_p),
$
where $bold(C)_p$ is the affine transformation matrix in APIC's context.

In addition to this, nodal force, i.e., force on grid nodes are calculated by
$
  bold(f)_i = -(4) / (Delta x^2) sum_p V_p N_i (bold(x)_p) bold(sigma)_p (bold(x)_i - bold(x)_p).
$

Note that $bold(sigma)_p$ is the Cauchy stress tensor defined at the position $bold(x)_p$, which satisfies $bold(sigma)_p = -p bold(I)$, where $p$ is the pressure here.
Relating two EoS $p V = n R T$ (Ideal Gas Law) and $p = K (1 - J)$ (Murnaghan Equation of State), where $K = -V partial p \/ partial V$, we can derive the following relation:
$
  bold(sigma)_p
  & = V_p (partial (n_0 R T dot V_p^(-1))) / (partial V_p) (1 - J) bold(I) \
  & = -C V_0 V_p^(-1) (1 - J) bold(I),
$
where $C$ is a constant, $V_0$ is the initial volume, because $V_0 prop n_0$. I'm not sure if this derivation is correct
Bringing this into the nodal force equation, we have
$
  bold(f)_i
  & = -(4) / (Delta x^2) sum_p V_p N_i (bold(x)_p) bold(sigma)_p (bold(x)_i - bold(x)_p) \
  & = (4) / (Delta x^2) sum_p C V_0 (1 - J) N_i (bold(x)_p) (bold(x)_i - bold(x)_p).
$

Applying this to particle to grid process, we have
$
  m bold(v)_i = Delta t bold(f)_i + sum_p m_p bold(v)_p N_i (bold(x)_p) + N_i (bold(x)_p) m_p bold(C)_p (
    bold(x)_i - bold(x)_p
  ),
$
which is implemented by rewriting $m_p bold(C)_p$ to
$
  bold(A)_p = m_p bold(C)_p + (4 Delta t) / (Delta x^2) C V_0(1 - J),
$
where the final equation is
$
  m bold(v)_i = sum_p m_p bold(v)_p N_i (bold(x)_p) + N_i (bold(x)_p) bold(A)_p (bold(x)_i - bold(x)_p).
$

#block(fill: rgb("#34b8ff66"), inset: 1em, radius: 3pt)[
  The derivation might not depend on IGL, but only the Murnaghan EoS with $K$ as a constant. This results in only
  $ bold(sigma)_p = - K (1 - J) bold(I). $

  Bringing this into the nodal force equation, we have
  $
    bold(f)_i
    & = (4) / (Delta x^2) sum_p K V_0 J (1 - J) N_i (bold(x)_p) (bold(x)_i - bold(x)_p),
  $
  which corresponds to (only if this particle is fluid)
  $
    bold(A)_p = m_p bold(C)_p + (4 Delta t) / (Delta x^2) K V_0 J (1 - J).
  $

  This do also work.
]

$J$, i.e., the Jacobian of $bold(F)$, is maintained not by calculating $det(bold(F))$ on-the-fly, but by iterative update. Consider
$
  bold(F)_(n+1) = (bold(I) + Delta t bold(C)_p) bold(F)_n,
$
taking the determinant on both sides,
$
  J_(n+1)
  & = det(bold(I) + Delta t bold(C)_p) J_n \
  & = (1 + Delta t tr(bold(C)_p)) J_n.
$

This ensures the numerical stability and gets rid of catastrophic cancellation in calculating the determinant.

Above is the derivation of weakly compressible MPM, which completely gets rid of the Poisson pressure projection step in APIC free-surface fluid, making the simulation blazingly fast and simple to implement, while achieving a similar level of incompressibility.
