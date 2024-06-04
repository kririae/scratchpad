#import "@preview/physica:0.9.3": *

#set page(width: 148mm, height: auto)

#let foreground = rgb("1F2430")
#let background = rgb("FFFFFF")
#let dark_mode = true
#if dark_mode {
  foreground = rgb("FDFDFD")
  background = rgb("1F2430")
}

#set page(fill: background)
#set text(foreground)
#set par(justify: true)

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

$bold(K)_p$ can be understood not via its inverse.
Let $bold(d)_i = bold(x)_i - bold(x)_p$, inertia tensor casts $bold(d)$ into angular momentum. Consider $d = 3$, where $bold(d) = (d_1, d_2, d_3)$, multiplied by $bold(omega) in RR^3$ gives the angular momentum, $bold(L) = bold(K) bold(omega)$. Then $bold(K)^(-1)$ can be understood as the transformation from angular momentum to angular velocity.

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

== Derivations of Weakly Compressible MPM

#line(length: 100%, stroke: foreground)

When simulating fluid with MPM, one problem occurs.
As we have seen already, forces should be applied on the grid instead of particles.

We adopt this process into the APIC particle to grid transfer, i.e.,
$
  m bold(v)_i = sum_p m_p bold(v)_p N_i (bold(x)_p) + m_p bold(C)_p (bold(x)_i - bold(x)_p),
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
