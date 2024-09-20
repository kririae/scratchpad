#import "@preview/physica:0.9.3": *

#let foreground = rgb("1F2430")
#let background = rgb("FFFFFF")
#let dark_mode = true
#if dark_mode {
  foreground = rgb("FDFDFD")
  background = rgb("1F2430")
}

#set page(fill: background, width: 148mm, height: auto)
#set text(foreground)
#set par(justify: true)

= Implicit Methods
