import settings;
settings.outformat = "svg";
settings.prc       = false;
settings.render    = 4;

import three;

currentprojection = orthographic(4, 3, 5, up=Y, center=true);
size(300);
pen our_blue = rgb("#D1D2E8");
pen our_darkblue = rgb("#B2B3C9");
defaultpen(linewidth(0.6pt));

real gridSize = 5;
real spacing = 1;
for (real x = -gridSize/2; x <= gridSize/2; x += spacing) {
    draw((x, -gridSize/2, 0)--(x, gridSize/2, 0), black);
    draw((-gridSize/2, x, 0)--(gridSize/2, x, 0), black);
}

for (int x = -2; x <= 2; x += 2) {
    for (int y = -2; y <= 2; y += 2) {
        draw(surface((x-spacing/2, y-spacing/2, 0)--(x+spacing/2, y-spacing/2, 0)--(x+spacing/2, y+spacing/2, 0)--(x-spacing/2, y+spacing/2, 0)--cycle), our_darkblue, nolight);
    }
}

for (int x = -1; x <= 2; x += 2) {
    for (int y = -1; y <= 2; y += 2) {
        draw(surface((x-spacing/2, y-spacing/2, 0)--(x+spacing/2, y-spacing/2, 0)--(x+spacing/2, y+spacing/2, 0)--(x-spacing/2, y+spacing/2, 0)--cycle), our_darkblue, nolight);
    }
}

for (int x = -1; x <= 2; x += 2) {
    for (int y = -2; y <= 2; y += 2) {
        draw(surface((x-spacing/2, y-spacing/2, 0)--(x+spacing/2, y-spacing/2, 0)--(x+spacing/2, y+spacing/2, 0)--(x-spacing/2, y+spacing/2, 0)--cycle), our_blue, nolight);
    }
}

for (int x = -2; x <= 2; x += 2) {
    for (int y = -1; y <= 2; y += 2) {
        draw(surface((x-spacing/2, y-spacing/2, 0)--(x+spacing/2, y-spacing/2, 0)--(x+spacing/2, y+spacing/2, 0)--(x-spacing/2, y+spacing/2, 0)--cycle), our_blue, nolight);
    }
}

dot((0, 0, 0), black);
label("$\mathbf{p}_f$", (0, 0, 0), N+W, black);

dot((0, 0, 5), black+2pt);
path3 c = circle((0, 0, 5), 0.4);
draw(c, our_blue);
draw(surface(c), our_blue+opacity(0.4), nolight);

dot((0.1, 0.1, 5), black+2pt);
label("$\mathbf{p}_c$", (0.1, 0.1, 5), S+W, black);
draw((0.1, 0.1, 5)--(-0.5, -0.5, 0), dashed);
draw((0.1, 0.1, 5)--( 0.5, -0.5, 0), dashed);
draw((0.1, 0.1, 5)--(-0.5,  0.5, 0), dashed);
draw((0.1, 0.1, 5)--( 0.5,  0.5, 0), dashed);
