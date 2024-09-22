#import "@preview/physica:0.9.3": *
#import "@preview/cetz:0.2.2": canvas, draw, vector, matrix
#import "@preview/codly:1.0.0": *
#import "@preview/gentle-clues:1.0.0": *

#let foreground = rgb("1F2430")
#let background = rgb("FFFFFF")
#let blue = rgb(blue)
#let stylize = gradient.linear(blue, blue)

#show: codly-init.with()
#show raw.where(block: false): set raw(lang: "typc")
#show raw.where(block: false): box.with(
  fill: luma(240),
  inset: (x: 3pt, y: 0pt),
  outset: (y: 3pt),
  radius: 2pt,
)

// Add a little circle to each of the link.
#show link: it => {
  it
  if type(it.dest) != label {
    sym.wj
    h(1.6pt)
    sym.wj
    super(box(height: 3.8pt, circle(radius: 1.2pt, stroke: 0.8pt + blue)))
  }
}

#show heading: it => {
  box[
    #if (it.depth == 1) {
      place(
        dx: -1em * 80%,
        dy: 0em,
        text(sym.hash, stylize),
      )
    }
    #it
  ]
}


#set page(width: 160mm, height: auto, fill: background)
#set text(foreground)
#set par(justify: true)
#set math.equation(numbering: "(1)")

#let hline = line(length: 100%, stroke: (paint: stylize, thickness: 0.8pt))
#let bold(symb) = math.bold(math.upright(symb))
#let ub(symb) = math.upright(math.bold(symb))

#align(
  center,
  [
    #text(20pt)[
      Intuitions and Derivations in Geometry
    ] \
    Zike Xu
  ],
)
#v(1em)

This is to record my notes on learning both the #link("https://github.com/alecjacobson/geometry-processing")[Alec Jacobson's Geometry Processing] and #link("https://web.archive.org/web/20240624130500/https://brickisland.net/DDGSpring2024/")[Keenan Crane's Discrete Differential Geometry].
The former one is more practical and is based on #link("https://libigl.github.io/")[`libigl`], the latter one might serve only as a mathematical reference in this note.

= Discrete Laplacian

#hline

When it comes to differential geometry in Computer Graphics, we cannot avoid
discussing _Laplacian_. More specifically, the _Laplacian Matrix_ $L$, which
might seem to come out by magic.

It turned out that $L$'s derivation is indeed quite concrete, that roots in the
Finite Element Method. As a recap, to work on mesh structure, we must remember
all the time the connection between the continuous definition and its
discretization on barycentric coordinate.

The geometry that we'll encounter in graphics applications can be described as a
map $f: M -> RR^3$, where $M$ is a region in the Euclidean plane in $RR^2$.
Think about it as a _texture mapping_ for now. Given an arbitrary point in $M$,
$f(bold(u))$ gives a unique point on the geometry surface, no need to further
illustrate it by figure. One'd better think of this mapping as a _parametric
geometry_ in high-school math instead of a $RR^2 -> R$ function, i.e., to avoid
placing the geometry over a 2D plane, but to snap the 2D plane towards the
geometry. This is really the correct mental model.

As for mesh, what is the analogy? $f$ is solely a utility for determining points
on the geometry and vice versa. Given the triangle id and its barycentric
coordinate $bold(b)$, we can also obtain the point. We are interpolating data on
the mesh, by this formula:
$
  g(bold(b)) = b_x g_i + b_y g_j + (1 - b_x - b_y) g_k,
$
where $g$ can be an arbitrary quantity to be interpolated on the mesh, $g_i$ and
the similar terms are the corresponding quantities stored on the vertices $i$,
$j$, $k$, respectively. To make this complete, we define a similar $g$ on the parameterization $M$,
$
  g(bold(u)) = B_i (bold(u)) g_i + B_j (bold(u)) g_j + B_k (bold(u)) g_k,
$
where $B_i (bold(u))$ gives a parameterization from $bold(u) in RR^2$ to the
barycentric coordinate $b in RR^2$. Though it makes sense in math, what do we
really mean by this parameterization? This is very interesting, as $B_i$(s) are the basis functions as in FEM that pull up *little tents* (@tents) around the vertex in $RR^2$. In this view, the actual 3D shape of the mesh is not important, the _vertex positions_ themselves are only the values, say, $RR^3$ field on a 2D plane.

#figure(
  image("figure/tent.svg"),
  caption: [The basis functions in FEM that appears as tents around vertices, figure from Keenan Crane's _Discrete Differential Geometry_ notes.],
) <tents>

#warning[
  In general, a mesh in graphics programs can be described as $M = (V, F)$, as
  `libigl` has done. But where's the room for the little tents? We don't need the
  extra parameterization vector $U$ to perform smoothing. We cannot even implicitly define the $B_i (bold(u))$.

  Actually, we must assume the existence of a conformal mapping on a plannar
  surface to this mesh structure by
  #link("https://en.wikipedia.org/wiki/Uniformization_theorem")[uniformization
  theorem]. We don't actually use this mapping in practice, as the dependency on
  which will finally be eliminated to angular mapping. You'll see the practice
  on this later.
]

What about the derivatives, e.g., $nabla g(bold(u))$? It really appears
everywhere. For the quantity to be differentiated, we should again figure out
what variables it is being differentiated against. We start from the continuous
view. Smoothing on geometry is described with
$
  pdv(q, t) = lambda Delta q,
$
where $lambda$ is a positive smoothing coefficient; $Delta q = nabla dot nabla
q$. The derivatives here are definitely w.r.t. $bold(u)$ in $M$ instead of $RR^3$, the
latter one is hard to define, as what's the meaning of a quantity defined on,
say, a plane but we are taking its derivative against the orthogonal direction.

Finite Element Method generally works on a certain equation or integration, e.g., Poisson Equation, but the idea of discretization holds still. Say we want to approximate $Delta q_i$, which can be approximated by taking the average of which on nearby area. Consider this integration,
$
  integral_(A_i) nabla dot nabla q (bold(u)) dd(A) = integral_(partial A_i) nabla q(bold(u)) dot bold(n)(bold(u)) dd(s),
$
with divergence theorem applied, we boost the area $A_i$ to the enclosing
area(it will be modified later ;) formed by its 1-ring neighbor, RHS can thus be
simplified to
$
  integral_(partial A_i) nabla q(bold(u)) dot bold(n)(bold(u)) dd(s)
  & = sum_(e in E(i)) nabla q(bold(u)) dot (bold(e)_0 - bold(e)_1)^bot,
$
where $bold(e)_0$ and $bold(e)_1$ are the vertices on the edge, being rotated
outwards. $nabla q(bold(u))$ can be quite hard to define on the boundary, so we
shrink $A_i$ a bit to ($bold(e)_0$ and $bold(e)_1$ to the corresponding edge
center)
$
  integral_(partial A_i) nabla q(bold(u)) dot bold(n)(bold(u)) dd(s)
  & = 1 / 2 sum_(e' in E(i)) nabla q(bold(u)) dot (bold(e)_0 - bold(e)_1)^bot,
$
where $e$ can be seen as shifted correspondingly to the inside of the triangle. Meanwhile, if we alias $bold(e)_0 - bold(e)_1$ to its opposite vertex's id $i$ as $e_i$
$
  1 / 2 nabla q(bold(u)) dot e_i^bot
  & = 1 / 2(nabla B_i (bold(u)) q_i + nabla B_j (bold(u)) q_j + nabla B_k (bold(u)) q_k) dot e_i^bot \
  & = (e_j^bot / (4 A_T) (q_j - q_i) + e_k^bot / (4 A_T) (q_k - q_i)) dot e_i^bot. \
$
Note that the simplification comes from $nabla B_i + nabla B_j + nabla B_k = 0$.
Hold on, let's simplify the $(e_alpha^bot dot e_beta ^bot) \/ (4 A_T)$ to aid
derivation:
$
  (e_alpha^bot dot e_beta^bot ) / (4 A_T) = (||e_alpha|| ||e_beta|| cos theta ) / (2 ||e_alpha|| ||e_beta|| sin theta ) = 1 / 2 cot theta.
$
Let $theta_k$ (the complement index) be the inner angle between the edges $e_j$
and $e_i$ (on $M$ for now). Here's the conclusion, we sum back per-edge value (which can also be
per-vertex)
$
  integral_(A_i) nabla dot nabla q (bold(u)) dd(A)
  & = 1 / 2 sum_(f in F(i)) [(q_j - q_i) cot theta_k + (q_k - q_i) cot theta_j] \
  ("rearrange") & -> 1 / 2 sum_(j in N(i)) [(q_j - q_i) (cot theta_k + cot theta_k')],
$
where $k$.. well, is hard to describe in a word, consider the edge from $i$ to
$j$ is shared by two triangles, $k$ and $k'$ are the opposite vertex of the edge
in the two triangles.
The average then becomes
$
  Delta q_i
  & = 1 / (2 A_i) sum_(j in N(i)) [(q_j - q_i) (cot theta_k + cot theta_k')] \
  & = 1 / (2 A_i) [sum_(j in N(i)) q_j (cot theta_k + cot theta_k') - sum_(j in N(i)) q_i (cot theta_k + cot theta_k')].
$

This way, we can define the Laplacian Matrix $L$, that $L q = Delta q$, following
$
  L_(i j) = cases(
    0.5 (cot theta_k + cot theta_k') & "if" i != j,
    -sum_(j != i) L_(i j) & "if" i = j
  )
$

Hold on a second! $theta_k$ is the angle defined on $M$ instead of on the $RR^3$
mesh. But actually, through conformal mapping, the angles are preserved. This is
why the 3D triangle's inner angles are used in $L$ instead of requiring to
compute a actual conformal parameterization. This answers the question
above---why only $M = (V, F)$ is provided with the absense of a parameterization
vector $U$.

Sometimes $L$ is multiplied by $M^(-1)$ to be $M^(-1) L$ when $A_i$ is taken into consideration and $M_(i i) = A_i$. We call this matrix *mass matrix*.

#info[
  But, hey, this is only the first approach to understand $L$. Solely understand
  the mathematical derivation gives almost nothing to work on. We'll cover some
  of the alternative approaches later, which might give you a surprise.
]

= Understanding Curvature

#hline

Although Alec Jacobson left *curvature* to the end of the assignments, perhaps because curvature has a lot to do with real differential geometry, I think it's worth talking about here.

For curvature to be defined on geometry, let's recap its 2D definition. Consider $gamma: RR -> RR^3$ a parametric curve, written as $gamma(t)$, $t in [a, b]$. The length of the curve is defined as
$
  L(a, b) = integral_a^b ||gamma'(t)|| dd(t).
$
As we know, the tangent vector encodes the _metric_ of the curve. We alternatively define $gamma(s)$ such that $s in [0, L(a, b)]$, which trivially gives $||gamma'(s)|| = 1$. This is called *arc-length parametrization*.

With arc-length parameterization defined,
$
  kappa(s) = lim_(t -> s) lr(||(gamma'(t) - gamma'(s)) / (t - s)||) = lr(||gamma''(s)||),
$
which, is _the magnitude of change in direction_. This is how we define _normal_
$bold(n)$ in parametric curve. With $bold(n)$ and $bold(t)$ orthogonal, we can
define _binormal_ $bold(b) = bold(t) times bold(n)$ in $RR^3$.

Now that we have curvature $kappa(s)$ defined on arbitrary parametric curve
$gamma$, we can define it for a surface. Let me take out the classical figure illustrated by Keenan Crane, see @curvature.

#figure(
  image("figure/curvature.svg"),
  caption: "Curvature of a surface, taken from Keenan Crane's notes",
) <curvature>

The geometry can be sliced by arbitrary number of times to obtain different parametric curves, each define its curvature at $bold(p)$. We can encode this approach to a function $kappa_"n" (bold(p), phi)$, which is called the _normal curvature_ at this point. Few quantities can be defined by taking operations on these normal curvatures:
$
  cases(
    H(bold(p)) = display(1/(2 pi) integral_0^(2 pi)) kappa_"n" (bold(p), phi) dd(phi) & -> "mean curvature",
    // principal curvatures
    k_1 (bold(p)) = display(max_phi) kappa_"n" (bold(p), phi) & -> "principal curvature",
    k_2 (bold(p)) = display(min_phi) kappa_"n" (bold(p), phi) & -> "principal curvature",
  )
$

#link("https://en.wikipedia.org/wiki/Euler's_theorem_(differential_geometry)")[Euler's theorem] states that,
$
  cases(
    kappa_"n" (bold(p), phi) = k_1 (bold(p)) cos^2(phi) + k_2 (bold(p)) sin^2(phi) & -> "normal curvature",
    H(bold(p)) = display(1/(2)) (k_1 (bold(p)) + k_2 (bold(p))) & -> "mean curvature",
    K(bold(p)) = k_1 (bold(p)) k_2 (bold(p)) & -> "gaussian curvature",
  )
$

All quantities are related by principal curvatures.

Working out their definitions, what's more important is their geometry
interpretation. Indeed, this is a really complex topic that connects a lot of
topics together.

== Mean Curvature Normal

I'll skip the derivation of this part because the original one is quite.. how to say, too relaxed. Meanwhile this requires a lot of knowledge of differential geometry and functional analysis.

Remember the conclusion. Mean curvature normal is the directional to maximize
the surface area.

== Shape Operator

The shape operator is a linear operator that maps a tangent vector to a vector,
defined with a $S_bold(p) : T_p M -> T_p M$. We use the symbol $S_bold(p)$ to denote the
shape operator at point $bold(p)$. It is defined as
$
  S_bold(p) (bold(v)) = - nabla_bold(v) bold(n),
$
where $bold(v)$ is a vector in the tangent space. The operator "measures how the
surface
bends in different directions".
(Differentiated against a vector in the tangent space is indeed quite strange)

Its quite surprising that $S_bold(p) (bold(v))$ satisfies that
$
  cases(
    S_bold(p) (bold(v)_1) = kappa_1 bold(v)_1,
    S_bold(p) (bold(v)_2) = kappa_2 bold(v)_2,
  )
$
that is to say, principal directions $bold(v)_1, bold(v)_2 in T_p M$ are
eigenvectors of $S_bold(p)$ while principal curvatures are eigenvalues. This
definition is not suitable for numerical computation, as we definitely cannot
form a matrix of $S_bold(p) : T_p M -> T_p M$ to perform eigenvalue
decomposition.

Luckily, $T_p M$ is a 2D plane. We can possibly define a matrix $S in RR^(2
times 2)$ based on arbitrary basis $bold(e)_1$ and $bold(e)_2$,
$
  S_bold(p) (v_1 bold(e)_1 + v_2 bold(e)_2)
  & = v_1 kappa_1 bold(r)_1 + v_2 kappa_2 bold(r)_2 \
  & = mat(kappa_1 bold(r)_1, kappa_2 bold(r)_2) mat(v_1, v_2)^top \
  & = mat(bold(r)_1, bold(r)_2) mat(kappa_1, 0; 0, kappa_2) mat(v_1, v_2)^top \
  => mat(bold(r)_1, bold(r)_2) mat(v'_1, v'_2)^top & = mat(bold(r)_1, bold(r)_2) Lambda mat(v_1, v_2)^top \
  => bold(v)' & = Lambda bold(v).
$
When the coordinate is $E$ instead of $R$, i.e., $E U = R$, where $U$ is a
rotation matrix. They satisfy
$
  cases(
    E = display(mat(bold(e)_1, bold(e)_2)),
    R = display(mat(bold(r)_1, bold(r)_2)),
  ) in RR^(3 times 2),
$
World vector $bold(x) = R bold(v)$ should be expressed with
$bold(x) = E bold(v)^star$. It indicates that $bold(v)^star$, the newly defined
vector in another parameterization, is related to $bold(v)$ with $bold(v) =
U^(-1) bold(v)^star$, giving
$
  U^(-1) bold(v)^star = Lambda U^(-1) bold(v)^star
$
then
$
  bold(v)^star = U Lambda U^(-1) bold(v)^star.
$
This really aids the computation of the shape operator, because when we obtain
$U$, a direct matrix multiplication gives $R$, which consist of the principal
directions. As always, the concrete example can be readily explored with #link("https://doc.sagemath.org/html/en/reference/riemannian_geometry/sage/geometry/riemannian_manifolds/parametrized_surface3d.html")[sage]:
```python
u, v = var('u, v', domain='real')
paraboloid = ParametrizedSurface3D([u, v, u^2+v^2], [u, v], 'paraboloid')
S = paraboloid.shape_operator()
print(S)
print(S.eigenvalues())
```

== Discrete Shape Operator

Obviously, as a linear discretization, the shape operator cannot be trivially
defined on the mesh. One approach is to define the shape operator on a quadratic
surface, as stated by _Estimating Differential Quantities Using Polynomial Fitting of Osculating Jets_.

The idea involves the following steps:
1. Find the 2-ring neighborhood of the vertex.
2. Do PCA on the 2-ring neighborhood to obtain the principal directions.
3. Project vertices onto the plane spanned by the principal directions, then create the height field.
4. Fit a quadratic surface to the height field, then compute the shape operator.
5. Re-project the shape operator back to the original space.

Suppose that the height field is $h(u, v) = a_1 u + a_2 v + a_3 u^2 + a_4 u v + a_5 v^2$,
then the shape operator can be computed as
```python
u, v = var('u, v', domain='real')
a_1, a_2, a_3, a_4, a_5 = var('a_1, a_2, a_3, a_4, a_5', domain='real')
surface = ParametrizedSurface3D([u, v, a_1*u + a_2*v + a_3*u^2 + a_4*u*v + a_5*v^2], [u, v], 'surface')
S = surface.shape_operator()
```

= Mesh Parameterization

#hline

One tricky thing to remember about parameterization is that, before actually
performing the parameterization, we do generally assume that there exists a
conformal mapping $f: M -> RR^3$ to perform derivative on similar to above. See @parameterization.

#figure(
  image("figure/parameterization.svg", width: 80%),
  caption: [Parameterization, figure borrowed from Keenan Crane's _Discrete Differential Geometry_ notes],
) <parameterization>

== Tutte Mapping

There's just one thing about Tutte's mapping, which is also called barycentric mapping.

#info[
  In 1963, Tutte showed that if the boundary of a disk topology mesh is fixed to a convex polygon (and all spring stiffnesses are positive), then minimising the energy above will result in injective (i.e. fold-free) flattening.
]

Tutte's mapping is obtained by minimising the following,
$
  E(q) = sum_({i, j} in E) w_(i j) ||q_i - q_j||^2,
$
where $w_(i j) > 0$ is an arbitrary weight that can be chosen manually. We reformulate the summation at the vertex,
$
  E(q) = sum_i sum_(j in N(i)) w_(i j) ||q_i - q_j||^2.
$

Before we go any further, we will introduce some mathematical tricks to deal
with this kind of with this kind of summation. If you see $sum_i$ somewhere,
don't panic. Because matrix product encapsulates an element-wise product and
sum, which can also be to a dot product. So our urgent task is
$
  a_i
  & = sum_(j in N(i)) w_(i j) ||q_i - q_j||^2 \
  & = sum_(j in N(i)) w_(i j) [tr(q_i q_i^top) - 2 tr(q_i q_j^top) + tr(q_j q_j^top)].
$
One should be very sensitive to symmetry. The above equation also appears in the
opposite side sum of $i$ from $j$. So it simplifies to
$
  a_i
  & = 2 sum_(j in N(i)) w_(i j) [tr(q_i q_i^top) - tr(q_i q_j^top)] \
  & = 2 tr(q_i sum_(j in N(i)) w_(i j) (q_i - q_j)^top).
$

This is an outer product between $q_i$ and the last one. So we put $q_i$ in the
columns of the first matrix, the later ones in the rows of the second matrix,
resulting in
$
  E(q) = sum_i a_i = 2 tr(q^top L q).
$

This already seems to be in the standard form accepted by `libigl`, where
`igl::min_quad_with_fixed` necessarily optimises the equation in the following
form,
$
  E(Z) = tr(1/2 Z^top A Z + Z^top B + C), "where" Z("known", :) = Y.
$

== Least Sqaure Conformal Mapping

Tutte's mapping do suffer from a series of problems,
1. It is, obviously not conformal.
2. One need to come up with the weight $w_(i j)$ manually. Anyway, the only
  weight provided is the geometry's edge lengths.

Can we do better? We assume the existence of a hypothetical domain $M$ in $RR^2$
on which we can perform a derivative. The geometry is necessarily a mapping $f:
M -> RR^3$, while the parameterization is $g: M -> RR^2$. In this setting we can
think about the derivative and the so-called conformal mapping, since the
original domain should also be $RR^2$.

Consider the definition of the Jacobian of a 2D to 2D mapping, $J in RR^(2 times
2)$ can help to map any vector in the domain to its image. Let us name the
axises in the domain to $x, y$ and the one in the image $u, v$. The Jacobian is
defined as
$
  J = mat(display(pdv(u, x)), display(pdv(u, y)); display(pdv(v, x)), display(pdv(v, y))),
$
while
$
  cases(
    display(mat(1, 0)^top) & -> J x = mat(display(pdv(u, x)), display(pdv(u, y)))^top,
    display(mat(0, 1)^top) & -> J y = mat(display(pdv(v, x)), display(pdv(v, y)))^top,
  ),
$
so the corresponding vectors in the image are $nabla u$ and $nabla v$. We
expect $nabla u$ and $nabla v$ to be as orthogonal as possible. Then comes the
question, how can we measure their ortogonality? I personally comes up with two
methods:
1. The absolute of the dot product between them.
2. The 2-norm between $nabla u$ rotated by 90 degrees and $nabla v$.

I don't exactly know why the former one is not used in practice (maybe to avoid
taking the absolute), but the second one definitely works. We define the
rotation matrix as
$
  x^bot = R x, "where" R = mat(0, -1; 1, 0),
$
then the energy can be formulated as
$
  E(q) = integral_Omega ||nabla v - (nabla u)^bot||^2 dd(A),
$
where $q = (u, v)$.

Massage the equation a little bit, we get
$
  E(q) = integral_Omega ||nabla v||^2 + ||nabla u||^2 - 2 nabla v dot (nabla u)^bot dd(A),
$
which can be decomposed into two parts:
$
  E(q) = E_"dirichlet" (q) - 2 E_"angle" (q),
$
with
$
  E_"angle" (q) = integral_Omega nabla v dot (nabla u)^bot dd(A).
$

Before we continue, note that the integration domain is $Omega$, which is still
inside the domain $M$, with the dummy variables $x, y$. We'll take a shift to $u, v$
later.

Let's examine the equation for $E_"angle" (q)$:
$
  E_"angle" (q)
  & = integral_Omega nabla v dot R (nabla u) dd(A) \
  & = integral_Omega (pdv(u, x) pdv(v, y) - pdv(u, y) pdv(v, x)) dd(A) \
  & = integral_Omega det(J) dd(A) -> integral_(S in RR^2) 1 dd(A(q)).
$

This indicates that $E_"angle" (q)$ is the area of the image inside $RR^2$. This
can also be understood by solely considering $nabla v dot (nabla u)^bot$, which
is simply $||nabla v|| ||nabla u|| sin theta$, the area in the image.

Momentarily forget the derivation we just made. *The area of a 2D shape can be
calculated by integrating over its boundary.* The matrix $A$ depends only on the
topology of the shape,
$
  "Area" = integral_(S in RR^2) 1 dd(A(q)) = 1 / 2 integral_(partial S in RR^2) q dot bold(n) dd(s),
$
which is a common, though not immediately obvious, trick. Here, $q$ is the
coordinate itself, satisfying
$
  nabla dot q = pdv(u, u) + pdv(v, v) = 2.
$

In discretization, the integral can be approximated with
$
  "Area" approx 1 / 2 sum_({i, j} in partial S)^n integral_"line" q dot bold(n) dd(s),
$
which, the astute readers will notice that the is just the area enclosed by $q_i$, $q_j$ and the origin.
Finally,
$
  "Area" approx 1 / 2 sum_({i, j} in partial S)^n det(q_i, q_j).
$

This can readily be encoded into a quadratic form:
$
  "Area" = Q^top A Q,
$
where $Q in RR^(2n times 1)$ and $A in RR^(2n times 2n)$, with $n$ the number of vertices. For convenience, $Q$ is arranged as the concatentation of all vertex coordinates:
$
  Q = (q_(1, x), q_(2, x), ..., q_(n, x), q_(1, y), q_(2, y), ..., q_(n, y))^top.
$
This makes it convenient to leverage the `igl::repdiag` function to construct later functions. For $A$ to appears symmetry, we further let $A <- 0.5(A + A^top)$, which does not change the result of the quadratic form.

As for the $E_"direchlet"$ portion, dirichlet energy is related to $L$ directly by (checkout the FEM view of the Laplacian Matrix)
$
  E_"dirichlet"(q) = Q^top mat(L, 0; 0, L) Q.
$

Those 2 terms can be combined into a single quadratic form to be optimized:
$
  E(q) = Q^top underbracket([mat(L, 0; 0, L) - 2 A], S) Q.
$

While in tutte's mapping, we simply use the `igl::map_vertices_to_circle` to
explicitly enforce the boundary condition, although this do work, we employ a
more general approach without explicit boundary coordinates, but instead
imposing some constrains, as proposed by
#link("http://www.geometry.caltech.edu/pubs/MTAD08.pdf")[Spectral Conformal
Parameterization]. The paper demonstrates the _implementation_ of the algorithm
through `igl::eigs` with generalized eigenvalue problem.

In addition to the minimization target, an extra constrain is imposed:
$
  integral_M ||u||^2 dd(A) = 1 => Q^top B Q = 1,
$
where
$
  B = mat(M, 0; 0, M),
$
and $M$ is the mass matrix, as previously defined in the first section. This makes sense because diagonal matrix in quadratic form sums up the squared $i$-th element of vector $Q$ with weight $M_(i i)$.

Although, let's recap the problem statement, is different from the previous statements:
$
  min_Q E(Q) = Q^top S Q, "w.r.t." Q^top B Q = 1.
$
With _Lagrange multiplier_, this writes to
$
  cal(L)(Q, lambda) = Q^top S Q + lambda (1 - Q^top B Q),
$
where
$
  cases(
    display(pdv(cal(L)(Q, lambda), Q)) = 0 & => S Q = lambda B Q,
    display(pdv(cal(L)(Q, lambda), lambda)) = 0 & => Q^top B Q = 1
  ),
$
which is equivalent to the generalized eigenvalue problem:
$
  A x = lambda B x
$
given $A$ and $B$ as $(S, B)$ (sorry for the different notations used here) respectively. This is implemented with a few lines
of code:
```cpp
Eigen::VectorXd sS;
igl::eigs(S, B, 3, igl::EIGS_TYPE_SM, Q, sS);

// skip the trivial solutions
U = Q.col(2).reshaped(Q.rows() / 2, 2);
```
