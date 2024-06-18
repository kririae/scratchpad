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

= Introduction to Incremental Potential Contact

#line(length: 100%, stroke: foreground)

Implicit time integration is often expressed as
$
  bold(x)^(t+1) & = bold(x)^(t) + h bold(v)^(t+1), \
  bold(v)^(t+1) & = bold(v)^(t) + h bold(M)^(-1) (f_"int" (bold(x)^(t+1)) + f_"ext"),
$
which expresses a linear system to be solved.
