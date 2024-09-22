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
      Notes on Physically-based Rendering
    ] \
    Zike Xu
  ],
)

#v(1em)

This is yet another learning notes on
#link("https://shuangz.com/courses/pbdr-course-sg20/downloads/pbdr-course-sg20-notes.pdf")[Physics-Based Differentiable Rendering].
While the course itself is quite comprehensive and detailed, I still find it
necessary to write my own notes on the subject.

= Camera and Image: The Common Question

#hline

// address the differences between image-space filtering and camera's
// importance function.

One really needs to understand that $W_e$ and $L_e$ are symmetric. Let's start
from the pixel measurement equation. In obtaining $L_i (bold(p), omega)$, the
question becomes how _radiance_ can be measured in an image. The answer is
$
  I_k = integral_A integral_(S^2) W_e^((k))(bold(p)_f, omega) L_i (bold(p)_f, omega) dd(sigma(omega)) dd(A(bold(p)_f)),
$ <measurement-integral>
where $A$ is _the camera plane_; $W_e^((k))$ is _the importance function_ at
this pixel $k$; $omega$ is the direction of the incoming radiance. This is a
per-pixel measurement that should be performed on all pixels to get the whole
image.

A few things to note:
1. What do we mean by the camera plane? Is it behind the aperture or in front of it?
2. What is the concrete form of the importance function? What's its relationship to, say, the image-space filtering function?

The answers are, respectively:
1. The camera plane is in front of the aperture by a distance of $1$ (for only pin-hole camera to simplify the statement).
2. The importance function is constructed from the image-space filtering function.

Let's address the second question in more detail. Importance functions are
defined per-pixel. The $W_e^((k))$ should be non-zero within the pixel's range
on this camera plane. See @image-plane.

#figure(
  image("figure/camera-plane.svg", width: 50%),
  caption: "The camera plane and the camera lens.",
) <image-plane>

Before discretizing @measurement-integral, we need to reflect on the
relationship between $omega$ and $bold(p)_f$. Can $omega$ determine $bold(p)_f$,
or vice versa? This really makes a difference because when performing Monte
Carlo integration, we must choose one to be the primary variable; certainly,
they are not independent, i.e., the evaluation of $p_(S^2 times A)(omega,
bold(p)_f)$ is difficult to perform.

To further simplify the problem, we can assume that there's only one pixel, and
rewrite the integral in surface form:
$
  I
  & = integral_A integral_cal(M) W_e (bold(p)_f -> bold(p)_s) L_i (bold(p)_f <- bold(p)_s) G(
    bold(p)_f, bold(p)_s
  ) dd(A(bold(p)_s)) dd(A(bold(p)_f)),
$
where $bold(p)_f$ is still the point on the camera plane, and $bold(p)_s$ is the
point on the scene; $G$ is the geometry term; $cal(M)$ is the scene manifold.

Consider the question: can we directly sample $bold(p)_s$ and get its PDF? The
answer is definitely no. Instead, we generally sample the camera lens, which is
a disk with radius $r$. To rewrite the integral onto the camera lens, we have
$
  dd(A(bold(p)_s)) = dd(A(bold(p)_l)) cos theta_l (d_s^2) / (d_l^2 cos theta_s),
$
where $bold(p)_l$ is the point on the lens; $theta_l$ is the angle between the
lens' normal and the ray and $theta_s$ is the angle between the scene geoemtry's
normal and the ray; $d_l = ||bold(p)_f - bold(p)_l||_2^2$ and $d_s = ||bold(p)_s
- bold(p)_l||_2^2$. Bringing this back to the integral, we have
$
  I = integral_A integral_cal(A)_l W_e (bold(p)_l -> bold(p)_f) L_i (bold(p)_l <- bold(p)_f) (cos^2 theta_l) / (d_l^2)
  dd(A(bold(p)_l)) dd(A(bold(p)_f)),
$
where we again get rid of the $bold(p)_s$ defined on the hypothetical scene.

Writing its Monte Carlo form, we have
$
  lr(angle.l I angle.r)
  & = W_e (bold(p)_l -> bold(p)_f) L_i (
    bold(p)_f <- bold(p)_l
  ) d_l^(-2) cos^2 theta_l dot underbracket(
    A pi r^2, "inverse pdf"
  ) \
  & = W_e L dot A pi r^2 dot cos^4 theta.
$
The symbols are stand-alone, so I'll skip the parameter and the subscript. For
the $W_e$ to be _invisible_, i.e., estimating $I$ is simply taking the average
of the samples of $L$, we have
$
  W_e =
  cases(display(1 / (A pi r^2 cos^4 theta)) &"  if within the frustrum",
        display(0) &"  otherwise")
$

Taking a step further, to compute the measurement of a certain pixel, $A$ has to
be confined to $A_k$, the pixel's area. Meanwhile, we need to confine $W_e's$
range to the pixel's range by introducing the image-space filtering function
$f_k$:
$
  W_e^((k)) (bold(p)_f) = (f_k (bold(p)_f)) / (A_k pi r^2 cos^4 theta).
$
While $angle.l I_k angle.r$ is made complex, we can use a biased form:
$
  angle.l I_k angle.r = (sum_j f_k (bold(p)_(f, j)) L_(i, j)) / (sum f_k (bold(p)_(f, j))),
$
where $bold(p)_(f, j)$ is the $j$-th sample on pixel $k$; $L_(i, j)$ is the
radiance of the $j$-th sample. This definitely introduces bias but is a correct
approximation since as $f_k$ approaches a delta function, the bias will be zero.
This is similar to the APIC's filtering function that also introduces numerical
diffusion. Nevertheless, the bias is not a problem in practice.

Anyway, per-pixel importance function is just a trick to efficiently estimate
the $I_k$. In characterizing the importance function, we still use the global
form.

#warning[
  I skipped the derivation of pin-hole camera's importance function, which I
  don't know how to get correct.
]

For brevity, Veach defines the _ray space inner product_ on two functions of
both space and direction as
$
  I = lr(angle.l W_e, L angle.r) = integral_cal(M) integral_(S^2) W_e (bold(p), omega) L (
    bold(p), omega
  ) dd(sigma(omega)) dd(A(bold(p))).
$
This formulation will demonstrate its power in the future. Good formulation
really simplifies the derivation. Let's try examing it for the BDPT's case.

Similar to #link("https://graphics.stanford.edu/papers/veach_thesis/")[Robust Monte Carlo], we use $ub(K)$ (kernel) to denote _local scattering_, $ub(G)$ (geometry) to denote propagation, where
$
  (ub(K) f) (bold(p), omega) & = integral_(S^2) f (bold(p), omega) K (bold(p), omega, omega') dd(sigma(omega')) \
  (ub(G) f) (bold(p), omega) & = cases(f(upright(trace)(bold(p), omega), -omega) "if reachable", 0)
$
It is easy to prove that they are linear operators.
Since $L_o = L_e + ub(K) L_i$, while $L_i = ub(G) L_o$, we have
$
  L_o & = L_e + ub(K) ub(G) L_o \
  & = (1 - ub(K) ub(G))^(-1) L_e \
  & = ub(I) L_e + ub(K) ub(G) L_e + ub(K) ub(G) ub(K) ub(G) L_e + dots.
$

Let $ub(S) = (1 - ub(K) ub(G))^(-1)$ (solution), we have $L_o = ub(S) L_e$. $I$ can be written as
$
  I = lr(angle.l W_e, L_i angle.r) = lr(angle.l W_e, ub(G) ub(S) L_e angle.r).
$
Through self-adjoint property, we have
$
  I = lr(angle.l ub(G) ub(S) W_e, L_e angle.r).
$

It looks like that we've just done something quite non-trivial? $ub(G) ub(S)$
has transferred onto $W_e$ instead of $L_e$, which reveals quite a unique
computational pattern:
$
  I
  & = integral_cal(M) integral_(S^2) (ub(G) ub(S) W_e) (bold(p), omega) L_e (
    bold(p), omega
  ) dd(sigma(omega)) dd(A(bold(p))) \
  & = integral_cal(M) integral_cal(M) (ub(G) ub(S) W_e) (bold(p)_0 -> bold(p)_1) L_e (bold(p)_0 -> bold(p)_1) G(
    bold(p)_0 <-> bold(p)_1
  )
  dd(A(bold(p)_1)) dd(A(bold(p)_0)).
$
Let me turn to my favorite notations of $L_e [0, 1] = L_e (bold(p)_0 ->
bold(p)_1)$, $W_e [1, 0] = W_e (bold(p)_1 -> bold(p)_0)$, and $G [0; 1] = G
(bold(p)_0 <-> bold(p)_1)$, let $ub(T) = ub(K) ub(G)$,
$
  I
  & = integral_(cal(M)^2) (ub(G) ub(S) W_e) [0, 1] L_e [0, 1] G [0; 1] dd(A[0, 1]) \
  & = integral_(cal(M)^2) (ub(G) W_e + ub(G) ub(T) W_e + dots) [0, 1] L_e [0, 1] G [0; 1] dd(A[0, 1]) \
  & = integral_(cal(M)^2) L_e [0, 1] G [0; 1] W_e [1, 0] dd(A[0, 1]) \
  & + integral_(cal(M)^3) L_e [0, 1] G [0; 1] f[2, 1, 0] G[1; 2] W_e [2, 1] dd(A[0, 1, 2]) \
  & + dots,
$
which, is almost the same as the NEE formulation! Think about it, in evaluating $I_k$, we enable the corresponding portion on the camera plane to illuminate, sample it correspondingly and evaluate the contribution on the light plane. Here no constrain is imposed, as long as $bold(p)_0$ is within the pixel's frustrum, Then we repeat this process for each pixel.

#info[
  Although in the implementation of BDPT we generally split the path integral into
  two parts parts so that the two multiply to the original path estimate and
  `connect` as if they were the same path--the mental model is no different from
  the NEE, this formulation is still an idea to be explored. In
  #link("https://rgl.s3.eu-central-1.amazonaws.com/media/papers/NimierDavid2020Radiative_1.pdf")[Radiative Backpropagation], it will completely unleash its power, where obtaining the
  adjoint form, i.e., $angle.l ub(G) ub(S) W_e, L_e angle.r$ wipes away almost all
  computational burden.
]

= Camera and Image: Towards Photon Mapping

#hline

It might be surprising, but photon mapping can also be encapsulated in this
framework. I found it disappointing that almost no resource covers this topic.
#link("https://pbr-book.org/3ed-2018/Light_Transport_III_Bidirectional_Methods/Stochastic_Progressive_Photon_Mapping")[PBRT] mixes the usage of $N$ to describe both the number of photons and the
number of paths, which is quite confusing. Let's try to clarify this.

= Differentiable Rendering 1: An Intuition

#hline

I'll skip the concept of differentiable rendering, as it is already well
explained in the course notes (and in many of the readers' mind).
We are taking derivative with respect to a parameter set $bold(pi)$, which might
contain vertex positions $bold(pi)_v$, color attributes $bold(pi)_c$ and more.

The task might be trivial when the geometry is static, e.g., we are only
mutating $bold(pi)_c$. That way we just simply differentiating the path's
measurement, which is just a sum on the products.

= Differentiable Rendering 2: To Make It Concrete

#hline

Remember in back-propagation, say the BP of $u(x, y), v(x, y)$, we should first
define a loss function $L(u, v)$, with the original Jacobi matrix
$
  J(x, y) = mat(pdv(u, x), pdv(u, y); pdv(v, x), pdv(v, y)).
$
Backpropagation is just to give $pdv(L, x)$ and $pdv(L, y)$ given $pdv(L, u)$
and $pdv(L, v)$. Formally speaking, this process can be described as
$
  nabla_(u, v) L = [J(x, y)^top] nabla_(x, y) L.
$
The takeaway is, to perform a BP, we simply transpose the Jacobi matrix and
multiply it with the gradient of the loss function.

This trivially leads to a problem---when the number of parameters is large, the
size of Jacobi matrix grows quadratically. Not to say that the Jacobi matrix is
not sparse, while chain-of-operations requires holding _at least_ all
intermediate parameters along the way. As for physically-based rendering, this
is definitely impossible.

Here's where the previously mentioned #link("https://rgl.s3.eu-central-1.amazonaws.com/media/papers/NimierDavid2020Radiative_1.pdf")[Radiative Backpropagation] comes into play.
It achieves a seemingly impossible and super cool task, instead of actually
computing the derivative, it propagates the _differential radiance_ just as the
radiance itself. This is yet another kind of symmetricy where instead of
emitting $L_e$, we can emit $W_e$. Now that we can emit $partial_bold(pi) L in
RR^(n times 1)$, $n$ is the number of parameters to be optimized considering one
component of RGB radiance only (because the assumption of independence).

We perform a simple differentiation on $L_o$:
$
  partial_bold(pi) L_o (bold(p), omega_o)
  & = underbracket(partial_bold(pi) L_e (bold(p), omega_o), ub(Q)) \
  & + integral_(S^2) partial_bold(pi) L_i (bold(p), omega_i)f(bold(p), omega_o, omega_i) dd(sigma(omega_i)) \
  & + underbracket(integral_(S^2) L_i (bold(p), omega_i) partial_bold(pi) f(bold(p), omega_o, omega_i) dd(sigma(omega_i)), ub(Q)).
$
Where $ub(Q) in RR^(n times 1)$ is
$
  ub(Q) =
  partial_bold(pi) L_e (bold(p), omega_o) +
  integral_(S^2) L_i (bold(p), omega_i) partial_bold(pi) f(bold(p), omega_o, omega_i) dd(omega_i).
$
Leading to
$
  partial_bold(pi) L_o = ub(Q) + ub(T) partial_bold(pi) L_o
$ <differential-solution>
following the conventions above. This is the _differential rendering equation_.
No magic till now.

Remember our target in backward propagation is to compute $partial_bold(pi) J$ given $partial_I J$, where $I$ is per-pixel measurement, say there are $m$ pixels. This is mathematically described as
$
  underbracket(partial_bold(pi) J, RR^(n times 1))
  & = underbracket([J^top], RR^(n times m)) times underbracket(partial_I J, RR^(m times 1)) \
  => [partial_bold(pi) J]_i & = sum_j [J^top]_(i, j) [partial_I J]_j \
  & = sum_j pdv(I_j, bold(pi)_i) [partial_I J]_j,
$
which involves the dot product between the gradient of the per-pixel measurement and the Jacobian matrix. Recall that
$
  I_j = integral_A integral_(S^2) W_e^((j)) (bold(p)_f, omega) L_i (
    bold(p)_f, omega
  ) dd(sigma(omega)) dd(A(bold(p)_f)) = lr(angle.l W_e^((j)), L_i angle.r).
$
turning the estimation into
$
  partial_bold(pi) J
  & = sum_j (
    underbracket(lr(angle.l partial_bold(pi)_i W_e^((j)), L_i angle.r), 0) + lr(angle.l W_e^((j)),  partial_bold(pi) L_i angle.r)
  ) [partial_I J]_j \
  & = lr(angle.l sum_j W_e^((j)) [partial_I J]_j,  partial_bold(pi) L_i angle.r) \
  & = lr(angle.l A_e, partial_bold(pi) L_i angle.r),
$
where $A_e = sum_j W_e^((j)) [partial_I J]_j in RR$. What a linearity! We further expand @differential-solution to
$
  partial_bold(pi) L_o = (ub(I) - ub(T))^(-1) ub(Q) = ub(S) ub(Q),
$
turning the computation to
$
  partial_bold(pi) J = lr(angle.l A_e, ub(G) ub(S) ub(Q) angle.r) = lr(angle.l ub(G) ub(S) A_e, ub(Q) angle.r).
$

This is indeed much more powerful and non-trivial than the traditional use of
the self-adjoint operator! Applying the operator to LHS and RHS gives a
completely different computational pattern. $A_e in RR$ is a very simple scalar
and can easily be carried over the whole scene, but $ub(Q)$ is not. $ub(Q)$
depends on the size of the differential parameter set, which is generally large.
Performing NEE on $ub(Q)$ is definitely not efficient enough (equivalent to
performing spectral rendering with a large number of spectral samples).

We elaborate on the computation of $partial_bold(pi) J$ here (you might ask why we're using again $lr(angle.l A_e, ub(G) ub(S) ub(Q) angle.r)$, because it still reveals the computational pattern in integral form, just as the BDPT):,
$
  lr(angle.l A_e, ub(G) ub(S) ub(Q) angle.r)
  & = integral_A integral_(S^2) A_e(bold(p), omega) [ub(G) ub(S) ub(Q)] (
    bold(p), omega
  ) dd(sigma(omega)) dd(A(bold(p))) \
  & = integral_(A^2) {A_e [0, 1] G[0; 1]} ub(Q)[1, 0] dd(A[0, 1]) \
  & + integral_(A^3) {A_e [0, 1] G[0; 1] G[1; 2] f[2, 1, 0]} ub(Q) [2, 1] dd(A[0, 1, 2]) \
  & dots
$
Since we trace from $bold(p)_0$ to $bold(p)_1$, where $bold(p)_0$ sits on the
camera plane. The computational pattern is exactly the same as the NEE, illustrated by the following pseudo code:

#show "c_omega": "omega"
#show "c_omegap": "omega'"
#show "c_p": math.bold("p")
#show "c_dy": $partial_bold(y) J$
#show "c_pp": math.bold("p'")
```python
def rbp(p, omega, adjoint_radiance):
    # 1. Find intersection with the closest surface
    c_pp = ray_intersect(p, omega)
    # 2. Accumulate scene parameter gradients
    grad += Q(p, -omega) * adjoint_radiance
    # 3. Sample the surface's BSDF
    c_omegap, weight = sample_bsdf(c_pp, -omega)
    # 4. Recurse
    rbp(c_pp, c_omegap, adjoint_radiance * weight)
```

It is almost no different from the NEE, except for the atomic accumulation.

#warning[
  This above method does not account for change of the scene geometry, so does
  the original paper. They "believe that the framework can be extended to handle
  the change of the scene geometry", but I'm not sure how to do it for now.
]

= Differentiable Rendering 3: The Path-Space Differentiable Rendering

#hline

So the question remains: how to efficiently deal with the discontinuity in the
geometry without introducing bias or explicitly sampling the sillohutte?

Before continuing, let's first address the discontinuity in the integrator.
Transferring the differential operator into the integral requires the integrand
to be differentiable:
$
  pdv(, pi) integral_0^1 f(x; pi) dd(x) != integral_0^1 pdv(f(x; pi), pi) dd(x).
$
Consider when $f(x; pi) = 1_(x > pi)$, the right-hand side always integrates to
zero. The trick is to separate the integral into parts at each discontinuity:
$
  pdv(, pi) integral_0^1 f(x; pi) dd(x)
  = sum_i pdv(, pi) integral_(a_i (pi))^(b_i (pi)) f(x; pi) dd(x) \
  = sum_i [
    integral_(a_i (pi))^(b_i (pi)) pdv(f(x; pi), pi) dd(x) + pdv(b_i (pi), pi) f(b_i (pi)^-; pi) - pdv(a_i (pi), pi) f(
      a_i (pi)^+; pi
    )
  ] \
  = underbracket(integral_(0)^(1) pdv(f(x; pi), pi) dd(x), "interior")
  + underbracket(sum_i [ pdv(b_i (pi), pi) f(b_i (pi)^-; pi) - pdv(a_i (pi), pi) f(
    a_i (pi)^+; pi
  )], "boundary").
$

The equation seems scary, but it is actually quite simple. Extending onto spherical integration,
$
  pdv(, pi) (integral_Omega f dd(Omega))
  = underbracket(integral_Omega pdv(f, pi) dd(Omega), "interior")
  + underbracket(integral_Gamma lr(angle.l bold(n), pdv(bold(x), pi) angle.r) (f^- - f^+) dd(Gamma), "boundary").
$ <discontinuity>

You'll easily find that $bold(n)$'s direction does not matter. The components in
the integral is non-trivial. The result is a $n$ by $1$ matrix, where $n$ is the
number of parameters. The boundary term requires more attention, as the inner
product involves a operation on $RR^(3 times 1)$ with $RR^(3 times n)$,
correspondingly to $n$ inner product of $RR^(3 times 1)$ vector. The later
integral can be easily written to integrate over the boundary of the scene
geometry instead of the local sphere (thus solved with Monte Carlo integration).

#info[
  The method from #link("https://people.csail.mit.edu/tzumao/diffrt/")[Differentiable Monte Carlo Ray Tracing through Edge Sampling] is very complex. That assumes
  1. Manifold everywhere.
  2. Detection of Silhouette edges.

  Meanwhile, although the inner product is "not that hard" to compute, $f^- -
  f^+$ is much more challenging. Because the discontinuity might occur not only at a single geometry but also at the visibility function. See @visibility.

  #figure(
    image("figure/discontinuity.png", width: 50%),
    caption: "The discontinuity in the integrator.",
  ) <visibility>
]

A little bit tooo complex, I give up here.

= Differentiable Rendering 4: Warped-Area Sampling

#hline

This approach is rather easy to understand. Let's go directly into derivation,
that can readily reveals almost everything.

As the @discontinuity suggests, two integrals are to be computed: the interior
and the boundary. While the interior is easy to compute, the boundary is not,
that requires hard approaches like the sillohutte sampling or bi-directional
tracing. The idea of the warped-area sampling is similar to the inverse of
FVM---instead of turning the interior integral into a sum on the boundary, we do
the opposite.

Consider
$
  II_B = integral_(partial D) lr(angle.l bold(n), pdv(bold(x), pi) angle.r) f(bold(x)) dd(partial D(bold(x))),
$ <discontinuity-boundary>
where the previous boundary integral is collapsed back to piece-wise integration
on the area, where $bold(n)$ pointing outwards. The reason to do this is, in
previous derivations, we want to take care of the two adjacent triangles
simultaneously in a single boundary sample. But now we sample the area instead
of the boundary, so the trick is removed.

By _divergence theorem_, @discontinuity-boundary casts to
$
  II_B = integral_D nabla_D dot (f(bold(x)) cal(V)_pi (bold(x))) dd(D(bold(x))).
$
Note the "boost of DoF" here. We don't need $partial bold(x) \/ partial pi$ to
be well-defined on the interior, but only on the boundary. The two equations are
the same, as long as
1. $cal(V)_pi (bold(x))$ is continuous.
2. $cal(V)_pi (bold(x))$ converges to the $partial bold(x) \/ partial pi$ near the boundary.
This is different from how FVM works in a sense that the original function
covers the boundary and the interior, while this method only requires the
function to be well-defined on the boundary, giving us much more flexibility to
choose it our own.

Instead of using the area integral (where the divergence is harder to define),
we again use the solid-angle integral, which analogous to perform a importance
sampling from the point's view. This gives us
$
  II_B in RR^(n times 1)
  & = integral_(S^2) nabla_omega dot underbracket([f(omega) cal(V)_pi (omega)], RR^(n times 1)) dd(sigma(omega)) \
  & = integral_(S^2) [nabla_omega f] (omega)dot cal(V)_pi (omega) dd(sigma(omega)) + integral_(S^2) f(omega) (
    nabla_omega dot cal(V)_pi
  ) (omega) dd(sigma(omega)).
$
Note that we consider the $1$ channel at a time here.

#info[
  Change of variables is indeed necessary but got canceled out by the end.
]

I'll give up this yet. #link("https://shuangz.com/projects/psdr-was-sa23/")[Warped-Area Reparameterization of Differential Path Integrals] does provide a good and brief explannation.