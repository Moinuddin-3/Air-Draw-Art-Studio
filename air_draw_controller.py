#!/usr/bin/env python3
"""
Air Draw Controller  (FINAL - single file)
==========================================
Draw, paint and create shapes in the air with your webcam. No mouse, no keyboard.

    LEFT  hand  ->  SELECT & CONTROL + ERASER (pinch on the panel to click;
                    point on the canvas to erase)
    RIGHT hand  ->  DRAW & CREATE     (point with index finger; fold the middle finger)
    BOTH  hands ->  ADVANCED          (two-hand drawing, two-hand shape sizing)

Gestures
--------
  Point (index up, middle folded) ..... draw / drag a shape / move pointer
  Pinch (thumb + index together) ...... click a button (when over the panel)
  Shape mode, one hand ................ point and drag, then lower the finger to place
  Shape mode, both index fingers up ... the two fingertips are opposite corners of the
                                        shape (distance = size); PINCH with the right
                                        hand to place it

Shape library: 2,000 shapes
---------------------------
  1,000 2D / 3D-style shapes (original) +
  1,000 NEW: Solid 3D (360 real 3D solids), Wire 3D (180 wireframes), Model 3D (72 objects:
  table, chair, castle, rocket, robot ...), Curves (120), Pattern (100), Fractal (40),
  Badge (100), Voxel 3D (28)

Fill modes: Outline | Fill | Outline+Fill | Wireframe
Art styles: Clean, Neon, Sketch, Comic, Shadow, Pastel, Blueprint, Double

Keyboard
--------
  U undo (or Ctrl+Z) | R redo (or Ctrl+Y) | C clear | S save | P paper mode | [ ] size
  F cycle fill mode | W wireframe on/off | A next art style
  , . previous/next shape | ; ' jump -25/+25 shapes | K next shape category
  x y z rotate 3D +15 deg | X Y Z rotate -15 deg | V reset 3D rotation
  T next tool | 1-9,0 colours | L skeleton | M toggle draw/shape mode | G shape palette
  Q / Esc quit

Install:  pip install opencv-python mediapipe numpy      (works with mediapipe 0.10.x and 1.x)
Run:      python AirDraw.py
"""

import itertools
import math
import os
import sys
import time
from collections import namedtuple

import cv2
import numpy as np

WIN = "Air Draw Controller"
FONT = cv2.FONT_HERSHEY_SIMPLEX
MAX_UNDO = 25
SAVE_DIR = "airdraw_saves"

COLORS = [
    ("Red", (40, 40, 230)), ("Orange", (0, 140, 255)), ("Yellow", (0, 230, 255)),
    ("Green", (60, 200, 60)), ("Cyan", (230, 220, 0)), ("Blue", (240, 90, 30)),
    ("Purple", (180, 50, 140)), ("Magenta", (200, 50, 255)), ("White", (255, 255, 255)),
    ("Black", (0, 0, 0)),
]
TOOLS = [("pencil", "Pencil"), ("pen", "Pen"), ("brush", "Brush"),
         ("marker", "Marker"), ("highlighter", "Highlight"), ("eraser", "Eraser")]
FILL_NAMES = ["Outline", "Fill", "Outline+Fill", "Wireframe"]


# =====================================================================================
#  SHAPE LIBRARY  (every shape is built in unit space [0,1]x[0,1] and stretched to a box)
# =====================================================================================
Ctx = namedtuple("Ctx", "w h th")  # pixel width, height and line thickness of the box
SHAPES = []


def shape(name, cat="2D", square=False, directional=False):
    def deco(fn):
        SHAPES.append(dict(name=name, fn=fn, square=square, directional=directional, cat=cat))
        return fn
    return deco


def _prim(pts, closed, shade, stroke, solid):
    return dict(pts=np.array(pts, float), closed=closed, shade=shade, stroke=stroke, solid=solid)


def P(pts, shade=1.0):            # closed, filled + outlined
    return _prim(pts, True, shade, True, False)


def L(pts):                       # open polyline (always stroked)
    return _prim(pts, False, None, True, False)


def S(pts):                       # closed outline only
    return _prim(pts, True, None, True, False)


def F(pts, shade=1.0):            # fill only
    return _prim(pts, True, shade, False, False)


def DOT(pts, shade=1.0):          # always-filled detail (eyes, dots)
    return _prim(pts, True, shade, False, True)


def arc(cx, cy, rx, ry, a0, a1, n=40):
    return [(cx + rx * math.cos(math.radians(a0 + (a1 - a0) * i / n)),
             cy + ry * math.sin(math.radians(a0 + (a1 - a0) * i / n))) for i in range(n + 1)]


def ell(cx=.5, cy=.5, rx=.5, ry=.5, n=64):
    return arc(cx, cy, rx, ry, 0, 360, n)[:-1]


def ngon(n, cx=.5, cy=.5, rx=.5, ry=.5, rot=-90):
    return [(cx + rx * math.cos(math.radians(rot + 360 * i / n)),
             cy + ry * math.sin(math.radians(rot + 360 * i / n))) for i in range(n)]


def star(n, inner, rot=-90):
    pts = []
    for i in range(2 * n):
        r = .5 if i % 2 == 0 else .5 * inner
        a = math.radians(rot + 180 * i / n)
        pts.append((.5 + r * math.cos(a), .5 + r * math.sin(a)))
    return pts


def bez(p0, p1, p2, p3, n=24):
    out = []
    for i in range(n + 1):
        t = i / n
        u = 1 - t
        out.append((u**3 * p0[0] + 3 * u * u * t * p1[0] + 3 * u * t * t * p2[0] + t**3 * p3[0],
                    u**3 * p0[1] + 3 * u * u * t * p1[1] + 3 * u * t * t * p2[1] + t**3 * p3[1]))
    return out


def ring_fill(outer, inner):
    """Single polygon with a zero-width bridge so fillPoly leaves a hole."""
    return list(outer) + [outer[0], inner[0]] + list(reversed(inner))


def _heart_pts():
    ts = np.linspace(0, 2 * math.pi, 80, endpoint=False)
    x = 16 * np.sin(ts) ** 3
    y = -(13 * np.cos(ts) - 5 * np.cos(2 * ts) - 2 * np.cos(3 * ts) - np.cos(4 * ts))
    x = (x - x.min()) / (x.max() - x.min())
    y = (y - y.min()) / (y.max() - y.min())
    return list(zip(x, y))


HEART = _heart_pts()
ARROW_R = [(0, .3), (.6, .3), (.6, 0), (1, .5), (.6, 1), (.6, .7), (0, .7)]
PLUS = [(.35, 0), (.65, 0), (.65, .35), (1, .35), (1, .65), (.65, .65),
        (.65, 1), (.35, 1), (.35, .65), (0, .65), (0, .35), (.35, .35)]


def rrect_pts(c):
    r = min(c.w, c.h) * .22
    rx, ry = r / c.w, r / c.h
    return (arc(rx, ry, rx, ry, 180, 270, 8) + arc(1 - rx, ry, rx, ry, 270, 360, 8) +
            arc(1 - rx, 1 - ry, rx, ry, 0, 90, 8) + arc(rx, 1 - ry, rx, ry, 90, 180, 8))


# ------------------------------------------------------------------ 2D: lines
@shape("Line", directional=True)
def _(c): return [L([(0, 0), (1, 1)])]


@shape("Curve", directional=True)
def _(c): return [L(bez((0, 1), (.3, -.15), (.7, 1.15), (1, 0)))]


@shape("Dotted line", directional=True)
def _(c):
    length = math.hypot(c.w, c.h)
    n = max(2, int(length / max(10, c.th * 3.2)) + 1)
    r = max(2.0, c.th * .9)
    out = []
    for i in range(n):
        t = i / (n - 1)
        out.append(DOT(ell(t, t, r / c.w, r / c.h, 10)))
    return out


@shape("Dashed line", directional=True)
def _(c):
    length = math.hypot(c.w, c.h)
    dash = max(14, c.th * 4)
    step = dash * 1.6
    n = max(1, int(length / step))
    out = []
    for i in range(n):
        s0 = i * step / length
        s1 = min(s0 + dash / length, 1)
        out.append(L([(s0, s0), (s1, s1)]))
    return out


# ------------------------------------------------------------------ 2D: basic polygons
@shape("Rectangle")
def _(c): return [P([(0, 0), (1, 0), (1, 1), (0, 1)])]


@shape("Square", square=True)
def _(c): return [P([(0, 0), (1, 0), (1, 1), (0, 1)])]


@shape("Rounded rect")
def _(c): return [P(rrect_pts(c))]


@shape("Rounded square", square=True)
def _(c): return [P(rrect_pts(c))]


@shape("Circle", square=True)
def _(c): return [P(ell())]


@shape("Ellipse")
def _(c): return [P(ell())]


@shape("Triangle")
def _(c): return [P([(.5, 0), (1, 1), (0, 1)])]


@shape("Right triangle")
def _(c): return [P([(0, 0), (0, 1), (1, 1)])]


@shape("Diamond")
def _(c): return [P([(.5, 0), (1, .5), (.5, 1), (0, .5)])]


@shape("Trapezoid")
def _(c): return [P([(.25, 0), (.75, 0), (1, 1), (0, 1)])]


@shape("Inv. trapezoid")
def _(c): return [P([(0, 0), (1, 0), (.75, 1), (.25, 1)])]


@shape("Parallelogram")
def _(c): return [P([(.25, 0), (1, 0), (.75, 1), (0, 1)])]


@shape("Kite")
def _(c): return [P([(.5, 0), (1, .35), (.5, 1), (0, .35)])]


for _n, _name in [(5, "Pentagon"), (6, "Hexagon"), (7, "Heptagon"), (8, "Octagon"),
                  (9, "Nonagon"), (10, "Decagon"), (11, "Hendecagon"), (12, "Dodecagon")]:
    shape(_name)(lambda c, n=_n: [P(ngon(n))])

# ------------------------------------------------------------------ 2D: stars & symbols
shape("Star")(lambda c: [P(star(5, .382))])
shape("Star 4")(lambda c: [P(star(4, .38))])
shape("Star 6")(lambda c: [P(star(6, .55))])
shape("Star 8")(lambda c: [P(star(8, .6))])
shape("Starburst")(lambda c: [P(star(16, .72))])
shape("Sparkle")(lambda c: [P(star(4, .14))])


@shape("Arrow right")
def _(c): return [P(ARROW_R)]


@shape("Arrow left")
def _(c): return [P([(1 - x, y) for x, y in ARROW_R])]


@shape("Arrow up")
def _(c): return [P([(y, 1 - x) for x, y in ARROW_R])]


@shape("Arrow down")
def _(c): return [P([(y, x) for x, y in ARROW_R])]


@shape("Double arrow")
def _(c): return [P([(0, .5), (.25, 0), (.25, .3), (.75, .3), (.75, 0), (1, .5),
                     (.75, 1), (.75, .7), (.25, .7), (.25, 1)])]


@shape("Chevron")
def _(c): return [P([(0, 0), (.6, 0), (1, .5), (.6, 1), (0, 1), (.4, .5)])]


@shape("Pentagon arrow")
def _(c): return [P([(0, 0), (.7, 0), (1, .5), (.7, 1), (0, 1)])]


@shape("Spiral")
def _(c):
    pts = []
    for i in range(160):
        t = i / 159
        a = t * 3 * 2 * math.pi
        r = .5 * t
        pts.append((.5 + r * math.cos(a), .5 + r * math.sin(a)))
    return [L(pts)]


@shape("Wave")
def _(c): return [L([(i / 80, .5 - .4 * math.sin(2 * math.pi * 2 * i / 80)) for i in range(81)])]


@shape("Zigzag")
def _(c): return [L([(i / 6, 0 if i % 2 == 0 else 1) for i in range(7)])]


@shape("Heart")
def _(c): return [P(HEART)]


@shape("Cloud")
def _(c):
    pts = []
    for i in range(120):
        th = 2 * math.pi * i / 120
        r = .32 + .18 * abs(math.sin(2.5 * th))
        pts.append((.5 + r * math.cos(th), .5 + .8 * r * math.sin(th)))
    return [P(pts)]


@shape("Leaf")
def _(c):
    a = bez((0, 1), (0, .3), (.5, 0), (1, 0))
    b = bez((1, 0), (1, .5), (.5, 1), (0, 1))
    return [P(a + b), L([(0, 1), (.7, .3)])]


@shape("Flower")
def _(c):
    out = [P(ell(.5 + .28 * math.cos(math.radians(-90 + 72 * i)),
                 .5 + .28 * math.sin(math.radians(-90 + 72 * i)), .21, .21, 32), .9) for i in range(5)]
    out.append(P(ell(.5, .5, .13, .13, 32), 1.3))
    return out


@shape("Lightning")
def _(c): return [P([(.6, 0), (.2, .55), (.45, .55), (.35, 1), (.8, .4), (.55, .4)])]


@shape("Banner")
def _(c): return [P([(0, .2), (1, .2), (.9, .5), (1, .8), (0, .8), (.1, .5)])]


@shape("Ribbon")
def _(c): return [P([(0, .3), (.2, .3), (.2, .9), (.1, .8), (0, .9)], .75),
                  P([(1, .3), (.8, .3), (.8, .9), (.9, .8), (1, .9)], .75),
                  P([(.1, .1), (.9, .1), (.9, .6), (.1, .6)])]


@shape("Callout")
def _(c): return [P([(0, 0), (1, 0), (1, .7), (.4, .7), (.2, 1), (.25, .7), (0, .7)])]


@shape("Speech bubble")
def _(c): return [P(arc(.5, .42, .5, .4, 140, 470, 48) + [(.15, 1)])]


@shape("Check", directional=False)
def _(c): return [L([(0, .55), (.35, .9), (1, .1)])]


@shape("Cross X")
def _(c): return [L([(0, 0), (1, 1)]), L([(1, 0), (0, 1)])]


@shape("Plus")
def _(c): return [P(PLUS)]


@shape("Crescent")
def _(c):
    outer = arc(.5, .5, .5, .5, -60, -300, 48)
    inner = bez((.75, .933), (.38, .7), (.38, .3), (.75, .067))
    return [P(outer + inner)]


@shape("Semicircle")
def _(c): return [P(arc(.5, 1, .5, 1, 180, 360, 48))]


@shape("Quarter circle")
def _(c): return [P([(0, 1)] + arc(0, 1, 1, 1, -90, 0, 32))]


@shape("Ring")
def _(c):
    o, i = ell(.5, .5, .5, .5, 64), ell(.5, .5, .25, .25, 64)
    return [F(ring_fill(o, i)), S(o), S(i)]


@shape("Pie")
def _(c): return [P([(.5, .5)] + arc(.5, .5, .5, .5, 35, 325, 48))]


@shape("Arc")
def _(c): return [L(arc(.5, 1, .5, 1, 180, 360, 48))]


@shape("House")
def _(c): return [P([(.5, 0), (1, .4), (.85, .4), (.85, 1), (.15, 1), (.15, .4), (0, .4)]),
                  L([(.42, 1), (.42, .65), (.58, .65), (.58, 1)])]


@shape("Shield")
def _(c): return [P([(0, 0), (1, 0), (1, .45)] + bez((1, .45), (.95, .8), (.7, .92), (.5, 1)) +
                    bez((.5, 1), (.3, .92), (.05, .8), (0, .45)))]


@shape("Teardrop")
def _(c): return [P(arc(.5, .66, .4, .34, -35, 215, 40) + [(.5, 0)])]


@shape("Gear", square=True)
def _(c):
    pts = []
    for i in range(8):
        b = i * 45
        for a, r in ((b - 9, .5), (b + 9, .5), (b + 13, .39), (b + 32, .39)):
            pts.append((.5 + r * math.cos(math.radians(a)), .5 + r * math.sin(math.radians(a))))
    return [P(pts), S(ell(.5, .5, .15, .15, 32))]


@shape("Frame")
def _(c):
    o = [(0, 0), (1, 0), (1, 1), (0, 1)]
    i = [(.2, .2), (.8, .2), (.8, .8), (.2, .8)]
    return [F(ring_fill(o, i)), S(o), S(i)]


@shape("Brackets")
def _(c): return [L([(.3, 0), (0, 0), (0, 1), (.3, 1)]), L([(.7, 0), (1, 0), (1, 1), (.7, 1)])]


@shape("Smiley", square=True)
def _(c): return [P(ell()), DOT(ell(.35, .38, .06, .06, 16), .3), DOT(ell(.65, .38, .06, .06, 16), .3),
                  L(arc(.5, .5, .28, .28, 20, 160, 24))]


@shape("Sun", square=True)
def _(c):
    out = [P(ell(.5, .5, .22, .22, 48))]
    for i in range(8):
        a = math.radians(45 * i)
        out.append(L([(.5 + .32 * math.cos(a), .5 + .32 * math.sin(a)),
                      (.5 + .5 * math.cos(a), .5 + .5 * math.sin(a))]))
    return out



# =====================================================================================
#  EXTRA SHAPE PACKS: Nature, Objects, Symbols, Art  (+ more 2D)
# =====================================================================================
def rect(x1, y1, x2, y2): return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
def circ(cx, cy, r, n=40): return ell(cx, cy, r, r, n)


def rr(x1, y1, x2, y2, r, n=6):
    return (arc(x1 + r, y1 + r, r, r, 180, 270, n) + arc(x2 - r, y1 + r, r, r, 270, 360, n) +
            arc(x2 - r, y2 - r, r, r, 0, 90, n) + arc(x1 + r, y2 - r, r, r, 90, 180, n))


def rot(pts, deg, cx=.5, cy=.5):
    a = math.radians(deg)
    c, sn = math.cos(a), math.sin(a)
    return [(cx + (x - cx) * c - (y - cy) * sn, cy + (x - cx) * sn + (y - cy) * c) for x, y in pts]


def sc(pts, s, cx=.5, cy=.5):
    return [(cx + (x - .5) * s, cy + (y - .5) * s) for x, y in pts]


def fit(pts, pad=.02, keep=True):
    a = np.array(pts, float)
    mn, mx = a.min(0), a.max(0)
    span = mx - mn
    if keep:
        a = (a - (mn + mx) / 2) / max(span.max(), 1e-9) * (1 - 2 * pad) + .5
    else:
        a = (a - mn) / np.maximum(span, 1e-9) * (1 - 2 * pad) + pad
    return [tuple(p) for p in a]


def param(fn, n=400, t1=2 * math.pi):
    return fit([fn(t1 * i / n) for i in range(n + 1)])


def cloud_pts(cx=.5, cy=.5, rx=.5, ry=.4, n=120):
    out = []
    for i in range(n):
        th = 2 * math.pi * i / n
        r = .64 + .36 * abs(math.sin(2.5 * th))
        out.append((cx + rx * r * math.cos(th), cy + ry * r * math.sin(th)))
    return out


def noisy(cx, cy, base, terms, n=90):
    out = []
    for i in range(n):
        t = 2 * math.pi * i / n
        r = base + sum(a * math.sin(k * t + ph) for a, k, ph in terms)
        out.append((cx + r * math.cos(t), cy + r * math.sin(t)))
    return out


def star_poly(n, k):
    pts = ngon(n)
    return [pts[(i * k) % n] for i in range(n)]


# ------------------------------------------------------------------ more 2D
shape("Pentagram")(lambda c: [P(star_poly(5, 2))])
shape("Heptagram")(lambda c: [P(star_poly(7, 3))])
shape("Octagram")(lambda c: [P(star_poly(8, 3))])
shape("Triangle star")(lambda c: [P(star(3, .45))])
shape("Notched arrow")(lambda c: [P([(0, .3), (.5, .3), (.5, 0), (1, .5), (.5, 1), (.5, .7), (0, .7), (.15, .5)])])
shape("Striped arrow")(lambda c: [P(rect(0, .3, .1, .7)), P(rect(.15, .3, .25, .7)),
                                  P([(.3, .3), (.6, .3), (.6, 0), (1, .5), (.6, 1), (.6, .7), (.3, .7)])])
shape("Bent arrow")(lambda c: [P([(0, 1), (0, .45), (.55, .45), (.55, .2), (1, .55), (.55, .9), (.55, .65), (.25, .65), (.25, 1)])])
shape("Up-down arrow")(lambda c: [P([(.5, 0), (1, .3), (.7, .3), (.7, .7), (1, .7), (.5, 1), (0, .7), (.3, .7), (.3, .3), (0, .3)])])
shape("Four-way arrow")(lambda c: [P([(.5, 0), (.65, .2), (.55, .2), (.55, .45), (.8, .45), (.8, .35), (1, .5), (.8, .65),
                                      (.8, .55), (.55, .55), (.55, .8), (.65, .8), (.5, 1), (.35, .8), (.45, .8), (.45, .55),
                                      (.2, .55), (.2, .65), (0, .5), (.2, .35), (.2, .45), (.45, .45), (.45, .2), (.35, .2)])])
shape("Curved arrow")(lambda c: [L(bez((0, 1), (0, .3), (.4, 0), (.78, .1))), P([(.72, 0), (1, .1), (.74, .28)])])
shape("U-turn arrow")(lambda c: [L([(.1, 1), (.1, .35)] + arc(.4, .35, .3, .3, 180, 360, 16) + [(.7, .55)]),
                                 P([(.55, .5), (.85, .5), (.7, .78)])])


@shape("Circular arrow")
def _(c):
    end = math.radians(240)
    ex, ey = .5 + .4 * math.cos(end), .5 + .4 * math.sin(end)
    tx, ty = -math.sin(end), math.cos(end)
    px, py = -ty, tx
    head = [(ex + .17 * tx, ey + .17 * ty), (ex + .1 * px, ey + .1 * py), (ex - .1 * px, ey - .1 * py)]
    return [L(arc(.5, .5, .4, .4, -30, 240, 40)), P(head)]


@shape("Thought bubble")
def _(c): return [P(cloud_pts(.5, .36, .5, .34)), P(circ(.22, .8, .07, 24)), P(circ(.1, .94, .04, 20))]


@shape("Braces")
def _(c):
    left = bez((.4, 0), (.2, 0), (.3, .4), (.1, .5)) + bez((.1, .5), (.3, .6), (.2, 1), (.4, 1))
    return [L(left), L([(1 - x, y) for x, y in left])]


# ------------------------------------------------------------------ NATURE
@shape("Pine tree", "Nature")
def _(c): return [P(rect(.44, .72, .56, 1), .7), P([(.5, .25), (.9, .75), (.1, .75)]), P([(.5, 0), (.8, .4), (.2, .4)])]


@shape("Round tree", "Nature")
def _(c): return [P(rect(.44, .55, .56, 1), .7), P(circ(.5, .3, .3), 1.0), P(circ(.3, .45, .22), .95), P(circ(.7, .45, .22), .95)]


shape("Mountain", "Nature")(lambda c: [P([(0, 1), (.3, .35), (.45, .6), (.7, 0), (1, 1)]),
                                       L([(.6, .2), (.68, .28), (.72, .2), (.8, .27)])])
shape("Mountain range", "Nature")(lambda c: [P([(0, 1), (.25, .35), (.5, 1)], .85), P([(.3, 1), (.65, 0), (1, 1)])])


@shape("Snowflake", "Nature")
def _(c):
    out = []
    for k in range(6):
        a = math.radians(60 * k - 90)
        ca, sa = math.cos(a), math.sin(a)
        out.append(L([(.5, .5), (.5 + .5 * ca, .5 + .5 * sa)]))
        for r0 in (.28, .4):
            for da in (-35, 35):
                b = a + math.radians(da)
                out.append(L([(.5 + r0 * ca, .5 + r0 * sa),
                              (.5 + r0 * ca + .12 * math.cos(b), .5 + r0 * sa + .12 * math.sin(b))]))
    return out


shape("Full moon", "Nature", square=True)(lambda c: [P(circ(.5, .5, .5)), DOT(circ(.35, .35, .1, 20), .75),
                                                     DOT(circ(.65, .55, .13, 20), .75), DOT(circ(.4, .72, .07, 16), .75)])
shape("Rain cloud", "Nature")(lambda c: [P(cloud_pts(.5, .35, .5, .3)), L([(.25, .72), (.2, .95)]),
                                         L([(.5, .72), (.45, .95)]), L([(.75, .72), (.7, .95)])])


@shape("Sunrise", "Nature")
def _(c):
    out = [P(arc(.5, .75, .3, .3, 180, 360, 24)), L([(0, .75), (1, .75)])]
    for a in (200, 235, 270, 305, 340):
        r = math.radians(a)
        out.append(L([(.5 + .38 * math.cos(r), .75 + .38 * math.sin(r)), (.5 + .55 * math.cos(r), .75 + .55 * math.sin(r))]))
    return out


shape("Rainbow", "Nature")(lambda c: [L(arc(.5, 1, .5, 1, 180, 360, 40)), L(arc(.5, 1, .4, .8, 180, 360, 40)),
                                      L(arc(.5, 1, .3, .6, 180, 360, 40))])
shape("Sea waves", "Nature")(lambda c: [L([(i / 60, .32 - .15 * math.sin(2 * math.pi * 1.5 * i / 60)) for i in range(61)]),
                                        L([(i / 60, .68 - .15 * math.sin(2 * math.pi * 1.5 * i / 60 + 1)) for i in range(61)])])
shape("Fish", "Nature")(lambda c: [P(ell(.4, .5, .4, .3, 48)), P([(.72, .5), (1, .2), (1, .8)]),
                                   DOT(circ(.22, .42, .04, 12), .3), L(arc(.5, .5, .1, .2, -60, 60, 10))])
shape("Bird", "Nature")(lambda c: [L(bez((0, .4), (.2, 0), (.4, .1), (.5, .5))), L(bez((.5, .5), (.6, .1), (.8, 0), (1, .4)))])
shape("Butterfly", "Nature")(lambda c: [P(ell(.27, .28, .26, .25, 40)), P(ell(.73, .28, .26, .25, 40)),
                                        P(ell(.3, .72, .2, .22, 32)), P(ell(.7, .72, .2, .22, 32)),
                                        P(ell(.5, .5, .04, .35, 20), .5), L([(.5, .2), (.38, 0)]), L([(.5, .2), (.62, 0)])])
shape("Cat face", "Nature")(lambda c: [P([(.1, .4), (.1, .02), (.38, .22)]), P([(.9, .4), (.9, .02), (.62, .22)]),
                                       P(ell(.5, .58, .46, .38, 48)), DOT(circ(.33, .5, .05, 12), .3),
                                       DOT(circ(.67, .5, .05, 12), .3), DOT([(.46, .62), (.54, .62), (.5, .68)], .4),
                                       L([(.05, .6), (.3, .66)]), L([(.05, .75), (.3, .72)]),
                                       L([(.95, .6), (.7, .66)]), L([(.95, .75), (.7, .72)])])
shape("Paw print", "Nature")(lambda c: [P(ell(.5, .72, .3, .25, 32)), P(ell(.15, .42, .1, .13, 20)), P(ell(.38, .2, .1, .15, 20)),
                                        P(ell(.62, .2, .1, .15, 20)), P(ell(.85, .42, .1, .13, 20))])
shape("Mushroom", "Nature")(lambda c: [P(rr(.35, .5, .65, 1, .1), .9),
                                       P(arc(.5, .55, .5, .55, 180, 360, 32)),
                                       DOT(circ(.3, .3, .06, 14), 1.6), DOT(circ(.6, .22, .05, 14), 1.6), DOT(circ(.72, .4, .05, 14), 1.6)])
shape("Tulip", "Nature")(lambda c: [L([(.5, .45), (.5, 1)]), P([(.5, .85), (.9, .65), (.6, .95)], .85),
                                    P([(.2, .05), (.35, .25), (.5, 0), (.65, .25), (.8, .05)] + arc(.5, .4, .3, .25, 0, 180, 16))])
shape("Cactus", "Nature")(lambda c: [P(rr(.38, .05, .62, 1, .12)), P(rr(.08, .35, .22, .7, .07)), P(rr(.08, .6, .4, .72, .06)),
                                     P(rr(.78, .25, .92, .6, .07)), P(rr(.6, .5, .92, .62, .06))])
shape("Feather", "Nature")(lambda c: [P(bez((.1, 1), (0, .4), (.4, .05), (.9, 0)) + bez((.9, 0), (.95, .5), (.5, .9), (.1, 1))),
                                      L([(.1, 1), (.85, .1)])])
shape("Sea shell", "Nature")(lambda c: [P([(.5, 1)] + arc(.5, 1, .5, 1, 200, 340, 30))] +
                             [L([(.5, 1), (.5 + .5 * math.cos(math.radians(a)), 1 + math.sin(math.radians(a)))])
                              for a in (225, 250, 270, 290, 315)])
shape("Apple", "Nature")(lambda c: [P(ell(.5, .6, .46, .4, 48)), L([(.5, .25), (.55, .02)]), P([(.55, .1), (.85, 0), (.78, .22)], .9)])

# ------------------------------------------------------------------ OBJECTS
shape("Key", "Objects")(lambda c: [S(circ(.2, .5, .2)), S(circ(.2, .5, .07)), L([(.4, .5), (1, .5)]),
                                   L([(.85, .5), (.85, .72)]), L([(.95, .5), (.95, .66)])])
shape("Light bulb", "Objects")(lambda c: [P(ell(.5, .36, .33, .34, 40)), P(rr(.38, .68, .62, .88, .04)),
                                          P([(.42, .88), (.58, .88), (.5, 1)], .8), L([(.38, .78), (.62, .78)])] +
                               [L([(.5 + .4 * math.cos(math.radians(a)), .36 + .42 * math.sin(math.radians(a))),
                                   (.5 + .5 * math.cos(math.radians(a)), .36 + .52 * math.sin(math.radians(a)))])
                                for a in (-150, -120, -90, -60, -30)])
shape("Umbrella", "Objects")(lambda c: [P(arc(.5, .5, .5, .5, 180, 360, 32) + arc(.833, .5, .167, .12, 0, 180, 8) +
                                          arc(.5, .5, .167, .12, 0, 180, 8) + arc(.167, .5, .167, .12, 0, 180, 8)),
                                        L([(.5, .5), (.5, .9)]), L(arc(.4, .9, .1, .1, 0, 180, 10))])
shape("Mug", "Objects")(lambda c: [P(rr(.08, .25, .68, 1, .08)), L(arc(.68, .6, .22, .22, -90, 90, 16)),
                                   L([(.25, .02), (.3, .1), (.25, .18)]), L([(.45, .02), (.5, .1), (.45, .18)])])
shape("Envelope", "Objects")(lambda c: [P(rect(0, .15, 1, .9)), L([(0, .15), (.5, .6), (1, .15)])])
shape("Camera", "Objects")(lambda c: [P([(.3, .25), (.38, .08), (.62, .08), (.7, .25)], .85), P(rr(0, .25, 1, .92, .08)),
                                      S(circ(.5, .58, .22)), S(circ(.5, .58, .11))])
shape("Phone", "Objects")(lambda c: [P(rr(.25, 0, .75, 1, .08)), S(rect(.31, .1, .69, .8)), S(circ(.5, .91, .04, 16))])
shape("Open book", "Objects")(lambda c: [P([(0, .1), (.5, .2), (.5, .95), (0, .85)], .95), P([(1, .1), (.5, .2), (.5, .95), (1, .85)], .8),
                                         L([(.1, .3), (.4, .36)]), L([(.1, .45), (.4, .51)]), L([(.9, .3), (.6, .36)]), L([(.9, .45), (.6, .51)])])
shape("Bell", "Objects")(lambda c: [P(arc(.5, .4, .35, .4, 180, 360, 24) + [(.95, .78), (.05, .78)]),
                                    DOT(circ(.5, .9, .08, 16), 1.0), L([(.5, 0), (.5, .02)])])


@shape("Flag", "Objects")
def _(c):
    top = [(.15 + .8 * t / 24, .05 + .05 * math.sin(2 * math.pi * 1.5 * t / 24)) for t in range(25)]
    bot = [(.15 + .8 * t / 24, .5 + .05 * math.sin(2 * math.pi * 1.5 * t / 24)) for t in range(24, -1, -1)]
    return [L([(.15, 0), (.15, 1)]), P(top + bot)]


shape("Trophy", "Objects")(lambda c: [L(arc(.2, .2, .14, .14, 90, 270, 16)), L(arc(.8, .2, .14, .14, 270, 450, 16)),
                                      P([(.2, 0), (.8, 0), (.75, .35), (.6, .52), (.4, .52), (.25, .35)]),
                                      P(rect(.45, .52, .55, .78), .85), P(rr(.3, .78, .7, .95, .03), .8)])
shape("Gift box", "Objects")(lambda c: [P(rect(.05, .35, .95, 1), .9), P(rect(0, .2, 1, .38)), L([(.5, .2), (.5, 1)]),
                                        P(ell(.35, .1, .15, .1, 24)), P(ell(.65, .1, .15, .1, 24))])
shape("Clock", "Objects", square=True)(lambda c: [P(circ(.5, .5, .5, 48)), L([(.5, .5), (.5, .2)]), L([(.5, .5), (.7, .6)])] +
                                       [L([(.5 + .4 * math.cos(math.radians(30 * k)), .5 + .4 * math.sin(math.radians(30 * k))),
                                           (.5 + .46 * math.cos(math.radians(30 * k)), .5 + .46 * math.sin(math.radians(30 * k)))])
                                        for k in range(12)])
shape("Hourglass", "Objects")(lambda c: [P([(.15, 0), (.85, 0), (.55, .5), (.85, 1), (.15, 1), (.45, .5)]),
                                         DOT([(.33, .12), (.67, .12), (.5, .4)], .6), L([(.08, 0), (.92, 0)]), L([(.08, 1), (.92, 1)])])
shape("Pencil", "Objects")(lambda c: [P(rot(rect(.25, .35, .85, .65), -45)), P(rot([(.25, .35), (0, .5), (.25, .65)], -45), .9),
                                      DOT(rot([(0, .5), (.08, .46), (.08, .54)], -45), .3), P(rot(rect(.85, .35, 1, .65), -45), .75)])
shape("Paper plane", "Objects")(lambda c: [P([(0, .5), (1, 0), (.6, 1), (.45, .7)]), L([(1, 0), (.45, .7)])])
shape("Rocket", "Objects")(lambda c: [P([(.7, .55), (.95, .88), (.7, .8)], .8), P([(.3, .55), (.05, .88), (.3, .8)], .8),
                                      P([(.5, 0)] + bez((.5, 0), (.85, .25), (.8, .6), (.7, .78), 12) + [(.3, .78)] +
                                        bez((.3, .78), (.2, .6), (.15, .25), (.5, 0), 12)),
                                      S(circ(.5, .35, .1, 24)), P([(.4, .8), (.5, 1), (.6, .8)], 1.4)])
shape("Car", "Objects")(lambda c: [P([(0, .68), (.04, .52), (.25, .46), (.35, .2), (.7, .2), (.82, .46), (.97, .52), (1, .68)]),
                                   P(ell(.25, .72, .14, .16, 24), .5), P(ell(.75, .72, .14, .16, 24), .5)])
shape("Sailboat", "Objects")(lambda c: [P([(0, .8), (1, .8), (.85, 1), (.15, 1)], .8), L([(.5, 0), (.5, .8)]),
                                        P([(.5, .05), (.5, .7), (.15, .7)]), P([(.55, .15), (.55, .7), (.85, .7)], .9)])
shape("Crown", "Objects")(lambda c: [P([(0, 1), (0, .3), (.25, .6), (.5, .1), (.75, .6), (1, .3), (1, 1)]),
                                     DOT(circ(.0, .28, .05, 12), 1.5), DOT(circ(.5, .08, .05, 12), 1.5), DOT(circ(1, .28, .05, 12), 1.5)])
shape("Gem", "Objects")(lambda c: [P([(.2, 0), (.8, 0), (1, .35), (.5, 1), (0, .35)]), L([(0, .35), (1, .35)]),
                                   L([(.35, 0), (.25, .35), (.5, 1)]), L([(.65, 0), (.75, .35), (.5, 1)])])
shape("Magnifier", "Objects")(lambda c: [S(circ(.4, .4, .35)), P([(.62, .72), (.72, .62), (1, .9), (.9, 1)], .8)])
shape("Padlock", "Objects")(lambda c: [L(arc(.5, .45, .28, .4, 180, 360, 24)), P(rr(.1, .45, .9, 1, .08)),
                                       DOT(circ(.5, .68, .07, 16), .3), DOT(rect(.47, .7, .53, .85), .3)])
shape("Headphones", "Objects")(lambda c: [L(arc(.5, .55, .42, .5, 180, 360, 30)), P(rr(.03, .5, .2, .92, .05)), P(rr(.8, .5, .97, .92, .05))])
shape("Music note", "Objects")(lambda c: [P(ell(.3, .82, .2, .15, 24)), L([(.48, .8), (.48, 0)]), P([(.48, 0), (.92, .2), (.92, .36), (.48, .16)])])
shape("Balloon", "Objects")(lambda c: [P(ell(.5, .38, .38, .4, 40)), P([(.45, .78), (.55, .78), (.5, .86)], .8),
                                       L(bez((.5, .86), (.3, .92), (.7, .96), (.5, 1)))])
shape("Candle", "Objects")(lambda c: [P(rect(.35, .4, .65, 1), .9), P(arc(.5, .2, .1, .13, -35, 215, 20) + [(.5, 0)], 1.3), L([(.5, .4), (.5, .32)])])
shape("Tent", "Objects")(lambda c: [P([(.5, 0), (1, 1), (0, 1)]), P([(.5, .45), (.65, 1), (.35, 1)], .55), L([(.5, 0), (.5, .45)])])

# ------------------------------------------------------------------ SYMBOLS
shape("Prohibited", "Symbols", square=True)(lambda c: [S(circ(.5, .5, .5, 48)), L([(.15, .15), (.85, .85)])])
shape("Play", "Symbols")(lambda c: [P([(.15, 0), (1, .5), (.15, 1)])])
shape("Pause", "Symbols")(lambda c: [P(rect(.1, 0, .4, 1)), P(rect(.6, 0, .9, 1))])
shape("Record", "Symbols", square=True)(lambda c: [P(circ(.5, .5, .5, 48))])
shape("Fast forward", "Symbols")(lambda c: [P([(0, .1), (.5, .5), (0, .9)]), P([(.5, .1), (1, .5), (.5, .9)])])
shape("Rewind", "Symbols")(lambda c: [P([(1, .1), (.5, .5), (1, .9)]), P([(.5, .1), (0, .5), (.5, .9)])])
shape("Power", "Symbols", square=True)(lambda c: [L(arc(.5, .55, .42, .42, -60, 240, 40)), L([(.5, 0), (.5, .5)])])
shape("Wifi", "Symbols")(lambda c: [L(arc(.5, .9, r, r * 1.0, 225, 315, 16)) for r in (.2, .38, .56)] + [DOT(circ(.5, .9, .05, 12), 1.0)])


@shape("Peace", "Symbols", square=True)
def _(c):
    ll = (.5 + .5 * math.cos(math.radians(135)), .5 + .5 * math.sin(math.radians(135)))
    lr = (.5 + .5 * math.cos(math.radians(45)), .5 + .5 * math.sin(math.radians(45)))
    return [S(circ(.5, .5, .5, 48)), L([(.5, 0), (.5, 1)]), L([(.5, .5), ll]), L([(.5, .5), lr])]


shape("Yin yang", "Symbols", square=True)(lambda c: [
    P(circ(.5, .5, .5, 48)),
    P(arc(.5, .5, .5, .5, 90, 270, 24) + arc(.5, .25, .25, .25, 270, 450, 16) + arc(.5, .75, .25, .25, 270, 90, 16), .35),
    DOT(circ(.5, .25, .07, 12), 1.8), DOT(circ(.5, .75, .07, 12), .3)])
shape("Infinity", "Symbols")(lambda c: [L(fit([(math.cos(t) / (1 + math.sin(t) ** 2), math.sin(t) * math.cos(t) / (1 + math.sin(t) ** 2))
                                              for t in np.linspace(0, 2 * math.pi, 200)], keep=False))])
shape("Location pin", "Symbols")(lambda c: [P(arc(.5, .38, .38, .38, 150, 390, 40) + [(.5, 1)]), S(circ(.5, .38, .14, 24))])
shape("Bookmark", "Symbols")(lambda c: [P([(.2, 0), (.8, 0), (.8, 1), (.5, .72), (.2, 1)])])
shape("Check circle", "Symbols", square=True)(lambda c: [P(circ(.5, .5, .5, 48)), L([(.25, .52), (.43, .7), (.75, .32)])])
shape("Cross circle", "Symbols", square=True)(lambda c: [P(circ(.5, .5, .5, 48)), L([(.3, .3), (.7, .7)]), L([(.7, .3), (.3, .7)])])
shape("Plus circle", "Symbols", square=True)(lambda c: [P(circ(.5, .5, .5, 48)), L([(.5, .25), (.5, .75)]), L([(.25, .5), (.75, .5)])])
shape("Minus", "Symbols")(lambda c: [P(rr(0, .38, 1, .62, .1))])
shape("Equals", "Symbols")(lambda c: [P(rect(0, .2, 1, .4)), P(rect(0, .6, 1, .8))])
shape("Divide", "Symbols")(lambda c: [P(rect(0, .42, 1, .58)), DOT(circ(.5, .14, .1, 16), 1.0), DOT(circ(.5, .86, .1, 16), 1.0)])
shape("Target", "Symbols", square=True)(lambda c: [S(circ(.5, .5, .5, 48)), S(circ(.5, .5, .33, 40)), P(circ(.5, .5, .16, 32))])
shape("Eye", "Symbols")(lambda c: [P(bez((0, .5), (.3, 0), (.7, 0), (1, .5)) + bez((1, .5), (.7, 1), (.3, 1), (0, .5))),
                                   S(circ(.5, .5, .2, 28)), DOT(circ(.5, .5, .09, 16), .3)])
shape("Exclamation", "Symbols")(lambda c: [P(rr(.4, 0, .6, .65, .05)), DOT(circ(.5, .88, .09, 16), 1.0)])

# ------------------------------------------------------------------ ART (patterns, curves, paint)
for _k, _nm in ((3, "Rose 3"), (5, "Rose 5"), (4, "Rose 8"), (6, "Rose 12")):
    shape(_nm, "Art", square=True)(lambda c, k=_k: [P(param(lambda t: (math.cos(k * t) * math.cos(t), math.cos(k * t) * math.sin(t)), 360))])
shape("Lissajous", "Art")(lambda c: [L(param(lambda t: (math.sin(3 * t + math.pi / 2), math.sin(2 * t)), 400))])
shape("Trefoil knot", "Art", square=True)(lambda c: [L(param(lambda t: (math.sin(t) + 2 * math.sin(2 * t), math.cos(t) - 2 * math.cos(2 * t)), 400))])
shape("Spirograph A", "Art", square=True)(lambda c: [L(param(lambda t: (2 * math.cos(t) + 5 * math.cos(2 * t / 3),
                                                                      2 * math.sin(t) - 5 * math.sin(2 * t / 3)), 600, 6 * math.pi))])
shape("Spirograph B", "Art", square=True)(lambda c: [L(param(lambda t: (5 * math.cos(t) + 5 * math.cos(5 * t / 3),
                                                                      5 * math.sin(t) - 5 * math.sin(5 * t / 3)), 700, 6 * math.pi))])
shape("Cardioid", "Art")(lambda c: [P(param(lambda t: ((1 - math.cos(t)) * math.cos(t), (1 - math.cos(t)) * math.sin(t)), 200), 1.0)])
shape("Astroid", "Art", square=True)(lambda c: [P(param(lambda t: (math.cos(t) ** 3, math.sin(t) ** 3), 200))])
shape("Limacon", "Art")(lambda c: [P(param(lambda t: ((1 + 2 * math.cos(t)) * math.cos(t), (1 + 2 * math.cos(t)) * math.sin(t)), 300))])


@shape("Double spiral", "Art", square=True)
def _(c):
    a = [(.5 + .48 * (t / 120) * math.cos(t / 120 * 5 * math.pi), .5 + .48 * (t / 120) * math.sin(t / 120 * 5 * math.pi)) for t in range(121)]
    return [L(a), L([(1 - x, 1 - y) for x, y in a])]


@shape("Square spiral", "Art", square=True)
def _(c):
    pts, x, y = [(0, 0)], 0, 0
    dirs = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    for i in range(16):
        d = dirs[i % 4]
        ln = i // 2 + 1
        x, y = x + d[0] * ln, y + d[1] * ln
        pts.append((x, y))
    return [L(fit(pts))]


shape("Ripple", "Art", square=True)(lambda c: [S(circ(.5, .5, r, 48)) for r in (.5, .4, .3, .2, .1)])
shape("Nested squares", "Art", square=True)(lambda c: [S(rect(m, m, 1 - m, 1 - m)) for m in (0, .12, .24, .36)])
shape("Honeycomb", "Art", square=True)(lambda c: [P(ngon(6, cx, cy, .15, .15, rot=-90)) for cx, cy in
                                                  [(.5, .5)] + [(.5 + .26 * math.cos(math.radians(60 * k)), .5 + .26 * math.sin(math.radians(60 * k))) for k in range(6)]])


def _sier(a, b, c, d):
    if d == 0:
        return [[a, b, c]]
    m = lambda p, q: ((p[0] + q[0]) / 2, (p[1] + q[1]) / 2)
    ab, bc, ca = m(a, b), m(b, c), m(c, a)
    return _sier(a, ab, ca, d - 1) + _sier(ab, b, bc, d - 1) + _sier(ca, bc, c, d - 1)


shape("Sierpinski", "Art")(lambda c: [P(t) for t in _sier((.5, 0), (1, 1), (0, 1), 3)])


def _koch(p, q, d):
    if d == 0:
        return [p]
    a, b = p + (q - p) / 3, p + 2 * (q - p) / 3
    peak = a + (b - a) * complex(math.cos(math.pi / 3), math.sin(math.pi / 3))
    return _koch(p, a, d - 1) + _koch(a, peak, d - 1) + _koch(peak, b, d - 1) + _koch(b, q, d - 1)


@shape("Koch snowflake", "Art", square=True)
def _(c):
    v = [complex(0, .75), complex(1, .75), complex(.5, .75 - .866)]
    pts = []
    for i in range(3):
        pts += _koch(v[i], v[(i + 1) % 3], 3)
    return [P(fit([(p.real, p.imag) for p in pts]))]


shape("Pinwheel", "Art", square=True)(lambda c: [P(rot([(.5, .5), (.5, .02), (.92, .12)], 90 * k), .8 + .1 * k) for k in range(4)])
shape("Sunburst", "Art", square=True)(lambda c: [P([(.5, .5), (.5 + .5 * math.cos(math.radians(30 * k - 7)), .5 + .5 * math.sin(math.radians(30 * k - 7))),
                                                   (.5 + .5 * math.cos(math.radians(30 * k + 7)), .5 + .5 * math.sin(math.radians(30 * k + 7)))]) for k in range(12)])
shape("Checkerboard", "Art", square=True)(lambda c: [S(rect(0, 0, 1, 1))] + [P(rect(i / 4, j / 4, (i + 1) / 4, (j + 1) / 4), .35)
                                                                           for i in range(4) for j in range(4) if (i + j) % 2 == 0])
shape("Stripes", "Art")(lambda c: [P(rect(0, i * .2, 1, i * .2 + .12)) for i in range(5)])
shape("Polka dots", "Art", square=True)(lambda c: [P(circ(.17 + .33 * i, .17 + .33 * j, .13, 20)) for i in range(3) for j in range(3)])
shape("Chevrons", "Art")(lambda c: [L([(0, .2 + .3 * k), (.5, .45 + .3 * k - .25), (1, .2 + .3 * k)]) for k in range(3)])
shape("Scallops", "Art")(lambda c: [P([(0, 1)] + sum([arc((i + .5) / 5, .6, .1, .5, 180, 360, 10) for i in range(5)], []) + [(1, 1)])])


@shape("Wave ribbon", "Art")
def _(c):
    top = [(t / 40, .3 + .2 * math.sin(2 * math.pi * t / 40)) for t in range(41)]
    bot = [(t / 40, .55 + .2 * math.sin(2 * math.pi * t / 40)) for t in range(40, -1, -1)]
    return [P(top + bot)]


@shape("Brush stroke", "Art")
def _(c):
    top, bot = [], []
    for i in range(41):
        t = i / 40
        w = .26 * (math.sin(math.pi * t) ** .55) + .04
        y = .5 + .14 * math.sin(2 * math.pi * t)
        top.append((t, y - w / 2))
        bot.append((t, y + w / 2))
    return [P(top + bot[::-1])]


@shape("Splatter", "Art", square=True)
def _(c):
    out = [P(noisy(.5, .5, .2, [(.04, 5, 0), (.03, 9, 1)]))]
    for k in range(9):
        a = math.radians(k * 41 + 10)
        d = .3 + .14 * ((k * 7) % 3) / 2
        r = .025 + .03 * ((k * 5) % 4) / 3
        out.append(DOT(circ(.5 + d * math.cos(a), .5 + d * math.sin(a), r, 12), 1.0))
    return out


shape("Ink blot", "Art", square=True)(lambda c: [P(noisy(.5, .5, .3, [(.1, 3, 1), (.07, 5, 0), (.05, 8, 2)]))])
shape("Paint palette", "Art")(lambda c: [P(ell(.5, .5, .5, .42, 64)), S(ell(.3, .74, .07, .09, 16)),
                                         DOT(circ(.3, .3, .07, 16), 1.5), DOT(circ(.5, .2, .07, 16), .6), DOT(circ(.7, .3, .07, 16), 1.8),
                                         DOT(circ(.8, .52, .07, 16), .5)])
shape("Paintbrush", "Art")(lambda c: [P(rot(rect(.4, .4, 1, .6), -45)), P(rot(rect(.3, .38, .42, .62), -45), .7),
                                      P(rot([(.3, .38), (.3, .62)] + bez((.3, .62), (.1, .65), (.05, .55), (0, .5)) +
                                            bez((0, .5), (.1, .4), (.2, .36), (.3, .38)), -45), 1.3)])
shape("Landscape frame", "Art")(lambda c: [S(rect(0, 0, 1, 1)), S(rect(.1, .1, .9, .9)),
                                           L([(.1, .8), (.35, .45), (.5, .65), (.65, .4), (.9, .8)]), DOT(circ(.75, .25, .07, 16), 1.0)])


@shape("Mandala", "Art", square=True)
def _(c):
    out = [P(rot(ell(.5, .17, .07, .17, 24), 30 * k), .9) for k in range(12)]
    out.append(S(circ(.5, .5, .2, 48)))
    out += [P(rot(ell(.5, .33, .03, .07, 12), 30 * k + 15), 1.2) for k in range(12)]
    out.append(DOT(circ(.5, .5, .06, 20), 1.3))
    return out


@shape("Mandala 8", "Art", square=True)
def _(c):
    out = [P(rot([(.5, .02), (.6, .22), (.5, .4), (.4, .22)], 45 * k), .9) for k in range(8)]
    out += [P(rot([(.5, .42), (.56, .5), (.5, .58), (.44, .5)], 45 * k + 22.5), 1.2) for k in range(8)]
    out.append(S(circ(.5, .5, .5, 48)))
    return out


shape("Quatrefoil", "Art", square=True)(lambda c: [P(circ(.5, .27, .25, 32)), P(circ(.27, .5, .25, 32)), P(circ(.73, .5, .25, 32)), P(circ(.5, .73, .25, 32))])
shape("Scales", "Art")(lambda c: [L(arc(.125 + .25 * i + (.125 if r % 2 else 0), .35 + .2 * r, rad, rad, 180, 360, 12))
                                  for r in range(3) for i in range(4 - (r % 2)) for rad in (.12, .08, .04)])
shape("Halftone", "Art", square=True)(lambda c: [P(circ(.125 + .25 * i, .125 + .25 * j, .03 + .035 * (i + j) / 6 * 1.0, 16)) for i in range(4) for j in range(4)])
shape("Starfield", "Art")(lambda c: [P(sc(star(4, .25), s, cx, cy)) for cx, cy, s in
                                     ((.25, .3, .5), (.7, .25, .35), (.55, .65, .6), (.2, .78, .3), (.88, .8, .4))])


@shape("Confetti", "Art")
def _(c):
    items = [(.15, .2, .1, .05, 20), (.4, .12, .08, .05, 70), (.7, .2, .1, .05, 110), (.88, .45, .08, .05, 40), (.25, .5, .1, .05, 150),
             (.55, .45, .09, .05, 10), (.12, .8, .08, .05, 95), (.45, .78, .1, .05, 60), (.75, .82, .09, .05, 130)]
    return [P(rot(rect(x - w / 2, y - h / 2, x + w / 2, y + h / 2), a, x, y), .8 + .15 * (i % 3)) for i, (x, y, w, h, a) in enumerate(items)]


# =====================================================================================
#  3D / 3D-STYLE SHAPES (shaded faces so they look solid in fill mode)
# =====================================================================================
def extrude(front, ox=.3, oy=.28, side=+1):
    """Oblique extrusion of a polygon. Visible side faces are back-face culled and shaded."""
    f = np.array(front, float)
    if side > 0:
        f[:, 0] = f[:, 0] * (1 - ox)
        off = np.array([ox, -oy])
    else:
        f[:, 0] = ox + f[:, 0] * (1 - ox)
        off = np.array([-ox, -oy])
    f[:, 1] = oy + f[:, 1] * (1 - oy)
    n = len(f)
    area2 = sum(f[i][0] * f[(i + 1) % n][1] - f[(i + 1) % n][0] * f[i][1] for i in range(n))
    sides = []
    for i in range(n):
        a, b = f[i], f[(i + 1) % n]
        e = b - a
        nrm = np.array([e[1], -e[0]]) if area2 > 0 else np.array([-e[1], e[0]])
        ln = np.linalg.norm(nrm)
        if ln < 1e-9:
            continue
        nrm = nrm / ln
        if np.dot(nrm, off) > 1e-6:
            sides.append(P([a, b, b + off, a + off], 0.85 - .2 * nrm[0] - .3 * nrm[1]))
    return sides + [P(f)]


UNIT_SQ = [(0, 0), (1, 0), (1, 1), (0, 1)]
shape("Cube", "3D", square=True)(lambda c: extrude(UNIT_SQ, .3, .3))
shape("Cuboid", "3D")(lambda c: extrude(UNIT_SQ, .25, .35))
shape("Rect prism", "3D")(lambda c: extrude(UNIT_SQ, .3, .22, side=-1))
shape("Tri prism", "3D")(lambda c: extrude([(.5, 0), (1, 1), (0, 1)], .3, .2))
shape("Pent prism", "3D")(lambda c: extrude(ngon(5), .28, .22))
shape("Hex prism", "3D")(lambda c: extrude(ngon(6, rot=0), .28, .22))
shape("Oct prism", "3D")(lambda c: extrude(ngon(8, rot=22.5), .28, .22))
shape("3D arrow", "3D")(lambda c: extrude(ARROW_R, .22, .25))
shape("3D star", "3D")(lambda c: extrude(star(5, .4), .2, .2))
shape("3D plus", "3D")(lambda c: extrude(PLUS, .25, .25))
shape("Stairs", "3D")(lambda c: extrude([(0, 1), (0, .66), (.33, .66), (.33, .33), (.66, .33),
                                         (.66, 0), (1, 0), (1, 1)], .28, .28))
shape("L-block", "3D")(lambda c: extrude([(0, 0), (.4, 0), (.4, .6), (1, .6), (1, 1), (0, 1)], .28, .28))


@shape("Cylinder", "3D")
def _(c):
    body = [(0, .18)] + arc(.5, .82, .5, .18, 180, 0, 32) + [(1, .18)] + arc(.5, .18, .5, .18, 0, -180, 32)
    return [P(body, .85), P(ell(.5, .18, .5, .18, 48), 1.2)]


@shape("Hollow cylinder", "3D")
def _(c):
    body = [(0, .18)] + arc(.5, .82, .5, .18, 180, 0, 32) + [(1, .18)] + arc(.5, .18, .5, .18, 0, -180, 32)
    return [P(body, .85), P(ell(.5, .18, .5, .18, 48), 1.2), P(ell(.5, .18, .34, .11, 48), .4)]


@shape("Cone", "3D")
def _(c): return [P([(.5, 0), (1, .85)] + arc(.5, .85, .5, .15, 0, 180, 32) + [(0, .85)], .9)]


@shape("Sphere", "3D", square=True)
def _(c): return [P(ell(.5, .5, .5, .5), .95), L(arc(.5, .5, .5, .15, 0, 180, 32)),
                  DOT(ell(.34, .3, .12, .08, 24), 1.5)]


@shape("Hemisphere", "3D")
def _(c): return [P(arc(.5, .6, .5, .6, 180, 360, 40) + arc(.5, .6, .5, .22, 0, 180, 32), .95),
                  DOT(ell(.32, .25, .1, .07, 24), 1.5)]


@shape("Sq. pyramid", "3D")
def _(c): return [P([(.5, 0), (0, .7), (.5, 1)], 1.0), P([(.5, 0), (.5, 1), (1, .7)], .7)]


@shape("Tetrahedron", "3D")
def _(c): return [P([(.55, 0), (0, .8), (1, .65)], .85), P([(.55, 0), (0, .8), (.45, 1)], 1.0),
                  P([(.55, 0), (.45, 1), (1, .65)], .7)]


@shape("Octahedron", "3D")
def _(c): return [P([(.5, 0), (0, .5), (.5, .7)], 1.05), P([(.5, 0), (.5, .7), (1, .5)], .75),
                  P([(.5, 1), (0, .5), (.5, .7)], .8), P([(.5, 1), (.5, .7), (1, .5)], .55)]


@shape("Torus", "3D")
def _(c):
    o, i = ell(.5, .5, .5, .38, 64), ell(.5, .55, .2, .12, 48)
    return [F(ring_fill(o, i), .9), S(o), S(i), L(arc(.5, .5, .38, .25, 200, 340, 24))]


@shape("Capsule", "3D")
def _(c):
    body = arc(.5, .25, .5, .25, 180, 360, 24) + arc(.5, .75, .5, .25, 0, 180, 24)
    return [P(body, .9), L([(0, .5), (1, .5)]), DOT(ell(.3, .2, .07, .1, 16), 1.5)]



def _pyramid(n):
    base = ngon(n, .5, .8, .5, .2, rot=90)
    out = []
    for i in range(n):
        a, b = base[i], base[(i + 1) % n]
        mx = (a[0] + b[0]) / 2
        if (a[1] + b[1]) / 2 > .8:
            out.append(P([(.5, 0), a, b], .65 + .5 * (1 - mx)))
    return out


shape("Pent pyramid", "3D")(lambda c: _pyramid(5))
shape("Hex pyramid", "3D")(lambda c: _pyramid(6))
shape("3D heart", "3D")(lambda c: extrude(HEART, .2, .2))
shape("3D house", "3D")(lambda c: extrude([(.5, 0), (1, .4), (.85, .4), (.85, 1), (.15, 1), (.15, .4), (0, .4)], .25, .25))
shape("3D chevron", "3D")(lambda c: extrude([(0, 0), (.6, 0), (1, .5), (.6, 1), (0, 1), (.4, .5)], .22, .25))
shape("3D lightning", "3D")(lambda c: extrude([(.6, 0), (.2, .55), (.45, .55), (.35, 1), (.8, .4), (.55, .4)], .2, .2))
shape("3D crescent", "3D")(lambda c: extrude(arc(.5, .5, .5, .5, -60, -300, 40) + bez((.75, .933), (.38, .7), (.38, .3), (.75, .067)), .2, .2))


@shape("Cylinder flat", "3D")
def _(c):
    body = [(.15, 0), (.85, 0)] + arc(.85, .5, .15, .5, -90, 90, 20) + [(.15, 1)] + arc(.15, .5, .15, .5, 90, 270, 20)
    return [P(body, .85), P(ell(.15, .5, .15, .5, 40), 1.2)]



# =====================================================================================
#  GENERATED SHAPE EXPANSION PACK  (original shapes bring the library to exactly 1,000)
# =====================================================================================

def _add_generated(name, cat, fn, square=False, directional=False):
    SHAPES.append(dict(name=name, fn=fn, square=square,
                       directional=directional, cat=cat))

# 1) POLYGON VARIANTS: 144
_idx = 0
for _n in range(3, 21):
    for _ri, _rot in enumerate(range(0, 360, 45)):
        _name = f"Polygon {_n}-gon {_ri + 1}"
        _add_generated(_name, "Shape", lambda c, n=_n, r=_rot: [P(ngon(n, rot=r))], square=(_n in (4, 8, 12, 16, 20)))
        _idx += 1

# 2) STAR VARIANTS: 64
for _n in range(5, 13):
    for _ri, _inner in enumerate((.18, .24, .30, .36, .42, .48, .56, .64)):
        _name = f"Star {_n} point {_ri + 1}"
        _add_generated(_name, "Art", lambda c, n=_n, inner=_inner: [P(star(n, inner))], square=True)

# 3) FLOWER / ROSETTE VARIANTS: 72
for _petals in range(3, 13):
    for _style in range(6):
        def _flower(c, petals=_petals, style=_style):
            out = []
            for k in range(petals):
                a = 360 * k / petals - 90
                cx = .5 + .25 * math.cos(math.radians(a))
                cy = .5 + .25 * math.sin(math.radians(a))
                rx = .10 + .025 * (style % 3)
                ry = .20 + .025 * ((style // 2) % 3)
                pts = ell(cx, cy, rx, ry, 28)
                if style % 2:
                    pts = rot(pts, a + 90, cx, cy)
                out.append(P(pts, .72 + .06 * (style % 4)))
            out.append(P(ell(.5, .5, .10 + .015 * style, .10 + .015 * style, 28), 1.1))
            return out
        _add_generated(f"Flower {_petals} petals style {_style + 1}", "Nature", _flower, square=True)

# 4) GEAR VARIANTS: 20
for _teeth in range(6, 16):
    for _hole in (0.10, 0.16):
        def _gear(c, teeth=_teeth, hole=_hole):
            pts = []
            for k in range(teeth):
                a = 360 * k / teeth
                for da, r in ((-10, .50), (10, .50), (16, .38), (360 / teeth - 16, .38)):
                    aa = math.radians(a + da)
                    pts.append((.5 + r * math.cos(aa), .5 + r * math.sin(aa)))
            return [P(pts), S(ell(.5, .5, hole, hole, 32))]
        _add_generated(f"Gear { _teeth }T hole {int(_hole*100)}", "Technical", _gear, square=True)

# 5) WAVE / SIGNAL VARIANTS: 50
for _freq in range(1, 11):
    for _amp_i, _amp in enumerate((.12, .18, .24, .30, .36)):
        def _wave(c, freq=_freq, amp=_amp):
            pts = [(i / 160, .5 + amp * math.sin(2 * math.pi * freq * i / 160)) for i in range(161)]
            return [L(pts)]
        _add_generated(f"Wave frequency {_freq} amplitude {_amp_i + 1}", "Art", _wave)

# 6) SPIRAL VARIANTS: 50
for _turns in range(2, 12):
    for _style in range(5):
        def _spiral(c, turns=_turns, style=_style):
            pts = []
            count = 260
            for i in range(count):
                t = i / (count - 1)
                a = turns * 2 * math.pi * t + style * math.pi / 5
                r = .48 * t
                if style % 2:
                    r *= .78 + .22 * math.sin(math.pi * t)
                pts.append((.5 + r * math.cos(a), .5 + r * math.sin(a)))
            return [L(pts)]
        _add_generated(f"Spiral {_turns} turns style {_style + 1}", "Art", _spiral, square=True)

# 7) GEOMETRIC PATTERN VARIANTS: 80
for _rings in range(3, 11):
    for _rot in range(10):
        def _pattern(c, rings=_rings, rotdeg=_rot * 9):
            out = []
            for j in range(rings):
                s = 1 - j / rings * .86
                pts = ngon(4 + (j % 5), .5, .5, .5 * s, .5 * s, rot=rotdeg + j * 12)
                out.append(S(pts))
            return out
        _add_generated(f"Nested pattern {_rings} levels {_rot + 1}", "Art", _pattern, square=True)

# 8) PARAMETRIC ART CURVES: 100
_curves = [
    ("Lissajous", lambda t, a, b, p: (math.sin(a*t+p), math.sin(b*t))),
    ("Harmonic", lambda t, a, b, p: (math.sin(a*t) * math.cos(p*t), math.sin(b*t) * math.sin(p*t))),
    ("Hypotrochoid", lambda t, a, b, p: ((a-b)*math.cos(t)+p*math.cos((a-b)*t/b), (a-b)*math.sin(t)-p*math.sin((a-b)*t/b))),
    ("Superformula", lambda t, a, b, p: (math.copysign(abs(math.cos(a*t/4))**(2/b), math.cos(a*t/4)), math.copysign(abs(math.sin(a*t/4))**(2/b), math.sin(a*t/4)))),
    ("Rose curve", lambda t, a, b, p: (math.cos(a*t)*math.cos(t), math.cos(a*t)*math.sin(t))),
]
for _ci, (_cname, _cf) in enumerate(_curves):
    for _a in range(2, 12):
        for _b in (2, 3):
            def _curve(c, cf=_cf, a=_a, b=_b, ci=_ci):
                try:
                    pts = [cf(2*math.pi*i/500, a, b, ci+1) for i in range(501)]
                    return [L(fit(pts))]
                except Exception:
                    return [L(arc(.5, .5, .4, .4, 0, 360, 100))]
            _add_generated(f"{_cname} { _a }-{_b}", "Art", _curve, square=True)

# 9) NATURE PATTERNS: 70
for _rays in range(5, 15):
    for _variant in range(7):
        def _nature(c, rays=_rays, variant=_variant):
            out = [P(ell(.5, .5, .12 + .02*variant, .12 + .02*variant, 28), .8)]
            for k in range(rays):
                a = math.radians(360*k/rays - 90)
                length = .28 + .025*variant
                w = .055 + .008*(variant % 4)
                tip = (.5 + length*math.cos(a), .5 + length*math.sin(a))
                base = (.5 + .10*math.cos(a), .5 + .10*math.sin(a))
                left = (.5 + w*math.cos(a+math.pi/2), .5 + w*math.sin(a+math.pi/2))
                right = (.5 + w*math.cos(a-math.pi/2), .5 + w*math.sin(a-math.pi/2))
                out.append(P([left, tip, right, base], .65 + .04*(variant % 5)))
            return out
        _add_generated(f"Petal sun { _rays } rays v{_variant + 1}", "Nature", _nature, square=True)

# 10) SYMBOL VARIANTS: 60
_symbol_templates = [
    ("Plus", PLUS), ("Arrow", ARROW_R), ("Heart", HEART),
    ("Diamond", [(0,.5),(.5,0),(1,.5),(.5,1)]),
    ("Chevron", [(0,0),(.5,.5),(0,1),(.25,1),(.75,.5),(.25,0)]),
]
for _si, (_sname, _base) in enumerate(_symbol_templates):
    for _v in range(12):
        def _symbol(c, base=_base, v=_v):
            p = rot(sc(base, .72 + .02*(v%8)), v*15)
            return [P(p, .65 + .04*(v%8))]
        _add_generated(f"{_sname} symbol variant {_v + 1}", "Symbols", _symbol, square=True)

# 11) 3D / EXTRUDED VARIANTS: 79
_3d_bases = [
    ("Cube", UNIT_SQ), ("Triangle", [(.5,0),(1,1),(0,1)]),
    ("Star", star(5,.4)), ("Heart", HEART), ("Plus", PLUS),
    ("Arrow", ARROW_R), ("House", [(.5,0),(1,.4),(.85,.4),(.85,1),(.15,1),(.15,.4),(0,.4)]),
    ("Chevron", [(0,0),(.6,0),(1,.5),(.6,1),(0,1),(.4,.5)]),
    ("Pentagon", ngon(5)), ("Hexagon", ngon(6, rot=0)),
    ("Octagon", ngon(8, rot=22.5))
]
_3d_i = 0
for _bi, (_bn, _bp) in enumerate(_3d_bases):
    for _v in range(7):
        def _solid(c, base=_bp, v=_v):
            ox = .16 + .035*(v%6)
            oy = .14 + .035*((v+2)%6)
            side = 1 if v % 2 == 0 else -1
            return extrude(base, ox, oy, side=side)
        _add_generated(f"3D {_bn} variant {_v + 1}", "3D", _solid, square=(_bn not in ("House", "Arrow")))
        _3d_i += 1
_add_generated("3D diamond variant 1", "3D", lambda c: extrude([(0,.5),(.5,0),(1,.5),(.5,1)], .24, .18), square=True)
_add_generated("3D diamond variant 2", "3D", lambda c: extrude([(0,.5),(.5,0),(1,.5),(.5,1)], .32, .28, side=-1), square=True)

_add_generated("3D circle prism variant 1", "3D", lambda c: extrude(ell(.5,.5,.5,.5,48), .22, .20), square=True)
_add_generated("3D circle prism variant 2", "3D", lambda c: extrude(ell(.5,.5,.5,.5,48), .30, .26, side=-1), square=True)

assert len(SHAPES) == 1000, "Shape library must contain exactly 1000 shapes"


# =====================================================================================
#  NEW 1000 SHAPES - PART 1: REAL 3D ENGINE (solids, wireframes, models)
#  tiny software renderer: rotate -> cull back faces -> depth sort -> shaded polygons
# =====================================================================================
STYLES = ["Clean", "Neon", "Sketch", "Comic", "Shadow", "Pastel", "Blueprint", "Double"]
STATE = dict(style=0, rot=[0, 0, 0], prev_fill=0)
LIGHT = np.array([-.35, .55, .75])
LIGHT /= np.linalg.norm(LIGHT)
PHI = (1 + math.sqrt(5)) / 2


def _rotmat(rx, ry, rz):
    a, b, c = math.radians(rx), math.radians(ry), math.radians(rz)
    ca, sa, cb, sb, cc, sc_ = math.cos(a), math.sin(a), math.cos(b), math.sin(b), math.cos(c), math.sin(c)
    Rx = np.array([[1, 0, 0], [0, ca, -sa], [0, sa, ca]])
    Ry = np.array([[cb, 0, sb], [0, 1, 0], [-sb, 0, cb]])
    Rz = np.array([[cc, -sc_, 0], [sc_, cc, 0], [0, 0, 1]])
    return Rz @ Rx @ Ry


def _newell(pts):
    n = np.zeros(3)
    for i in range(len(pts)):
        p, q = pts[i], pts[(i + 1) % len(pts)]
        n += [(p[1] - q[1]) * (p[2] + q[2]), (p[2] - q[2]) * (p[0] + q[0]), (p[0] - q[0]) * (p[1] + q[1])]
    ln = np.linalg.norm(n)
    return n / ln if ln > 1e-12 else np.array([0., 0., 1.])


def hull_faces(V, eps=1e-6):
    """Faces (ordered vertex index lists) of the convex hull of a small point set."""
    V = np.asarray(V, float)
    faces = {}
    for i, j, k in itertools.combinations(range(len(V)), 3):
        nr = np.cross(V[j] - V[i], V[k] - V[i])
        ln = np.linalg.norm(nr)
        if ln < 1e-9:
            continue
        nr = nr / ln
        s = (V - V[i]) @ nr
        if s.max() > eps and s.min() < -eps:
            continue
        on = frozenset(np.where(np.abs(s) <= eps)[0].tolist())
        if on in faces:
            continue
        if s.max() > eps:
            nr = -nr
        pts = V[list(on)]
        c = pts.mean(0)
        u = pts[0] - c
        u = u / np.linalg.norm(u)
        w = np.cross(nr, u)
        faces[on] = sorted(on, key=lambda t: math.atan2(float((V[t] - c) @ w), float((V[t] - c) @ u)))
    return list(faces.values())


def build_mesh(parts):
    """parts: list of dict(V=..., F=[idx lists], optional N=[normals]). Returns render-ready mesh."""
    Vs, Fc, E, off = [], [], set(), 0
    for p in parts:
        V = np.asarray(p["V"], float)
        c = V.mean(0)
        for k, idx in enumerate(p["F"]):
            idx = list(idx)
            pts = V[idx]
            if "N" in p:
                n = np.asarray(p["N"][k], float)
            else:
                n = _newell(pts)
                if n @ (pts.mean(0) - c) < 0:
                    n = -n
            Fc.append((np.array(idx) + off, n))
            for a in range(len(idx)):
                u, v = idx[a] + off, idx[(a + 1) % len(idx)] + off
                E.add((min(u, v), max(u, v)))
        Vs.append(V)
        off += len(V)
    V = np.vstack(Vs)
    V = V - (V.min(0) + V.max(0)) / 2
    return dict(V=V, F=Fc, E=sorted(E))


def _project(mesh, view):
    ang = [view[i] + STATE["rot"][i] for i in range(3)]
    R = _rotmat(*ang)
    V = mesh["V"] @ R.T
    x, y = V[:, 0], -V[:, 1]
    cx, cy = (x.min() + x.max()) / 2, (y.min() + y.max()) / 2
    span = max(x.max() - x.min(), y.max() - y.min(), 1e-9)
    k = .94 / span
    return (x - cx) * k + .5, (y - cy) * k + .5, V[:, 2], R


def solid_prims(mesh, view):
    px, py, z, R = _project(mesh, view)
    items = []
    for idx, n in mesh["F"]:
        nr = R @ n
        if nr[2] <= 1e-4:
            continue
        sh = min(1.2, max(.42, .5 + .75 * max(0.0, float(nr @ LIGHT))))
        items.append((float(z[idx].mean()), [(float(px[i]), float(py[i])) for i in idx], sh))
    items.sort(key=lambda t: t[0])
    return [P(p, s) for _, p, s in items]


def wire_prims(mesh, view):
    px, py, _, _ = _project(mesh, view)
    return [L([(float(px[i]), float(py[i])), (float(px[j]), float(py[j]))]) for i, j in mesh["E"]]


# ------------------------------------------------------------------ mesh parts
def part_hull(V):
    V = np.array(V, float)
    return dict(V=V, F=hull_faces(V))


def part_box(cx, cy, cz, sx, sy, sz):
    hx, hy, hz = sx / 2, sy / 2, sz / 2
    V = [(cx - hx, cy - hy, cz - hz), (cx + hx, cy - hy, cz - hz), (cx + hx, cy + hy, cz - hz), (cx - hx, cy + hy, cz - hz),
         (cx - hx, cy - hy, cz + hz), (cx + hx, cy - hy, cz + hz), (cx + hx, cy + hy, cz + hz), (cx - hx, cy + hy, cz + hz)]
    Fc = [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4], [3, 2, 6, 7], [0, 3, 7, 4], [1, 2, 6, 5]]
    return dict(V=np.array(V, float), F=Fc)


def part_prism(n, h=1.6, r=1., tr=None, rot0=0.):
    tr = r if tr is None else tr
    ang = [math.radians(rot0 + 360 * i / n) for i in range(n)]
    V = [(r * math.cos(a), -h / 2, r * math.sin(a)) for a in ang]
    Fc = []
    if tr > 0:
        V += [(tr * math.cos(a), h / 2, tr * math.sin(a)) for a in ang]
        Fc += [[i, (i + 1) % n, n + (i + 1) % n, n + i] for i in range(n)]
        Fc.append(list(range(n, 2 * n)))
    else:
        V.append((0, h / 2, 0))
        Fc += [[i, (i + 1) % n, n] for i in range(n)]
    Fc.append(list(range(n)))
    return dict(V=np.array(V, float), F=Fc)


def part_antiprism(n, h=1.4, r=1.):
    V = [(r * math.cos(math.radians(360 * i / n)), -h / 2, r * math.sin(math.radians(360 * i / n))) for i in range(n)]
    V += [(r * math.cos(math.radians(360 * (i + .5) / n)), h / 2, r * math.sin(math.radians(360 * (i + .5) / n))) for i in range(n)]
    Fc = [list(range(n)), list(range(n, 2 * n))]
    for i in range(n):
        Fc.append([i, (i + 1) % n, n + i])
        Fc.append([(i + 1) % n, n + (i + 1) % n, n + i])
    return dict(V=np.array(V, float), F=Fc)


def part_bipyramid(n, h=1.3, r=1.):
    V = [(r * math.cos(math.radians(360 * i / n)), 0, r * math.sin(math.radians(360 * i / n))) for i in range(n)]
    V += [(0, h, 0), (0, -h, 0)]
    Fc = []
    for i in range(n):
        Fc.append([i, (i + 1) % n, n])
        Fc.append([i, (i + 1) % n, n + 1])
    return dict(V=np.array(V, float), F=Fc)


def part_sphere(lat=8, lon=14, r=1.):
    V = [(0, r, 0)]
    for k in range(1, lat):
        th = math.pi * k / lat
        for j in range(lon):
            ph = 2 * math.pi * j / lon
            V.append((r * math.sin(th) * math.cos(ph), r * math.cos(th), r * math.sin(th) * math.sin(ph)))
    V.append((0, -r, 0))
    ring = lambda k, j: 1 + (k - 1) * lon + (j % lon)
    Fc = [[0, ring(1, j), ring(1, j + 1)] for j in range(lon)]
    for k in range(1, lat - 1):
        for j in range(lon):
            Fc.append([ring(k, j), ring(k, j + 1), ring(k + 1, j + 1), ring(k + 1, j)])
    south = len(V) - 1
    Fc += [[south, ring(lat - 1, j + 1), ring(lat - 1, j)] for j in range(lon)]
    return dict(V=np.array(V, float), F=Fc)


def part_torus(R=1., r=.4, nu=16, nv=8):
    V = []
    for u in range(nu):
        for v in range(nv):
            a, b = 2 * math.pi * u / nu, 2 * math.pi * v / nv
            V.append(((R + r * math.cos(b)) * math.cos(a), r * math.sin(b), (R + r * math.cos(b)) * math.sin(a)))
    Fc, N = [], []
    for u in range(nu):
        for v in range(nv):
            u1, v1 = (u + 1) % nu, (v + 1) % nv
            Fc.append([u * nv + v, u1 * nv + v, u1 * nv + v1, u * nv + v1])
            um, vm = 2 * math.pi * (u + .5) / nu, 2 * math.pi * (v + .5) / nv
            N.append((math.cos(vm) * math.cos(um), math.sin(vm), math.cos(vm) * math.sin(um)))
    return dict(V=np.array(V, float), F=Fc, N=N)


def tf(p, s=(1, 1, 1), t=(0, 0, 0), ry=0, rz=0):
    V = np.asarray(p["V"], float) * np.array(s, float)
    if ry:
        a = math.radians(ry)
        c, sn = math.cos(a), math.sin(a)
        V = np.stack([V[:, 0] * c + V[:, 2] * sn, V[:, 1], -V[:, 0] * sn + V[:, 2] * c], 1)
    if rz:
        a = math.radians(rz)
        c, sn = math.cos(a), math.sin(a)
        V = np.stack([V[:, 0] * c - V[:, 1] * sn, V[:, 0] * sn + V[:, 1] * c, V[:, 2]], 1)
    return dict(V=V + np.array(t, float), F=p["F"])


def sym(base, parity=None):
    out = set()
    for p in set(itertools.permutations(base)):
        for sg in itertools.product((1, -1), repeat=3):
            if parity is not None and sg[0] * sg[1] * sg[2] != parity:
                continue
            out.add(tuple(round(a * b, 9) + 0.0 for a, b in zip(p, sg)))
    return sorted(out)


def _cyc(a, b):
    out = []
    for sa in (1, -1):
        for sb in (1, -1):
            x, y = sa * a, sb * b
            out += [(0, x, y), (x, y, 0), (y, 0, x)]
    return out


_TC = math.sqrt(2) - 1
BASE_SOLIDS = [
    ("Tetrahedron", lambda: [part_hull([(1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)])]),
    ("Cube", lambda: [part_hull(sym((1, 1, 1)))]),
    ("Octahedron", lambda: [part_hull(sym((1, 0, 0)))]),
    ("Dodecahedron", lambda: [part_hull(sym((1, 1, 1)) + _cyc(1 / PHI, PHI))]),
    ("Icosahedron", lambda: [part_hull(_cyc(1, PHI))]),
    ("Cuboctahedron", lambda: [part_hull(sym((1, 1, 0)))]),
    ("Truncated Octahedron", lambda: [part_hull(sym((0, 1, 2)))]),
    ("Rhombic Dodecahedron", lambda: [part_hull(sym((1, 1, 1)) + sym((2, 0, 0)))]),
    ("Truncated Tetrahedron", lambda: [part_hull(sym((3, 1, 1), parity=1))]),
    ("Truncated Cube", lambda: [part_hull(sym((_TC, 1, 1)))]),
    ("Square Pyramid", lambda: [part_prism(4, 1.4, 1, 0, 45)]),
    ("Pentagonal Pyramid", lambda: [part_prism(5, 1.4, 1, 0, -90)]),
    ("Hexagonal Pyramid", lambda: [part_prism(6, 1.4, 1, 0)]),
    ("Triangular Prism", lambda: [part_prism(3, 1.5, 1, None, -90)]),
    ("Pentagonal Prism", lambda: [part_prism(5, 1.5, 1, None, -90)]),
    ("Hexagonal Prism", lambda: [part_prism(6, 1.5, 1)]),
    ("Octagonal Prism", lambda: [part_prism(8, 1.5, 1, None, 22.5)]),
    ("Cylinder", lambda: [part_prism(16, 1.8, 1)]),
    ("Cone", lambda: [part_prism(16, 1.8, 1, 0)]),
    ("Square Frustum", lambda: [part_prism(4, 1.2, 1.2, .6, 45)]),
    ("Hexagonal Frustum", lambda: [part_prism(6, 1.2, 1.2, .55)]),
    ("Hexagonal Antiprism", lambda: [part_antiprism(6)]),
    ("Octagonal Antiprism", lambda: [part_antiprism(8, 1.2)]),
    ("Hexagonal Bipyramid", lambda: [part_bipyramid(6)]),
    ("Pentagonal Bipyramid", lambda: [part_bipyramid(5, 1.2)]),
    ("Globe Sphere", lambda: [part_sphere(8, 14)]),
    ("Donut Torus", lambda: [part_torus(1, .4, 16, 8)]),
    ("Long Cuboid", lambda: [part_box(0, 0, 0, 2.4, 1, 1)]),
    ("Flat Slab", lambda: [part_box(0, 0, 0, 2.2, .3, 2.2)]),
    ("Thin Disc", lambda: [part_prism(20, .3, 1.2)]),
]


# ------------------------------------------------------------------ composite 3D objects
def _table():
    return [part_box(0, 1, 0, 2.4, .2, 1.6)] + [part_box(sx * 1.05, .45, sz * .65, .2, .9, .2) for sx in (-1, 1) for sz in (-1, 1)]


def _chair():
    return [part_box(0, .9, 0, 1.2, .15, 1.2), part_box(0, 1.55, -.55, 1.2, 1.2, .15)] + \
           [part_box(sx * .5, .4, sz * .5, .15, .8, .15) for sx in (-1, 1) for sz in (-1, 1)]


def _bed():
    return [part_box(0, .3, 0, 1.6, .4, 2.4), part_box(0, .65, 0, 1.5, .3, 2.2), part_box(0, .9, -.9, .9, .2, .4)]


def _skyscraper():
    return [part_box(0, .8, 0, 1.4, 1.6, 1.4), part_box(0, 2, 0, 1, .8, 1), part_box(0, 2.7, 0, .6, .6, .6),
            tf(part_prism(6, 1.0, .05), t=(0, 3.5, 0))]


def _castle():
    out = [part_box(0, .7, 0, 1.6, 1.4, 1.6)]
    for sx in (-1, 1):
        for sz in (-1, 1):
            out.append(tf(part_prism(8, 2, .35), t=(sx * 1, 1, sz * 1)))
            out.append(tf(part_prism(8, .7, .45, 0), t=(sx * 1, 2.35, sz * 1)))
    return out


def _ziggurat():
    return [part_box(0, .15 + .3 * k, 0, 3 - .6 * k, .3, 3 - .6 * k) for k in range(5)]


def _snowman():
    return [tf(part_sphere(8, 12), t=(0, 0, 0)), tf(part_sphere(8, 12), s=(.7, .7, .7), t=(0, 1.55, 0)),
            tf(part_sphere(8, 12), s=(.5, .5, .5), t=(0, 2.6, 0))]


def _rocket():
    return [part_prism(12, 2.4, .5), tf(part_prism(12, 1, .5, 0), t=(0, 1.7, 0))] + \
           [tf(part_box(.6, -.8, 0, .6, .8, .1), ry=120 * k) for k in range(3)]


def _robot():
    return [part_box(0, 1, 0, 1.2, 1.4, .8), part_box(0, 2.05, 0, .8, .7, .7),
            part_box(-.9, 1.1, 0, .3, 1.1, .3), part_box(.9, 1.1, 0, .3, 1.1, .3),
            part_box(-.3, -.2, 0, .35, 1.0, .35), part_box(.3, -.2, 0, .35, 1.0, .35),
            tf(part_prism(6, .5, .04), t=(0, 2.65, 0))]


def _lamppost():
    return [part_prism(8, 3, .08), tf(part_sphere(6, 10), s=(.35, .35, .35), t=(0, 1.65, 0)),
            tf(part_prism(8, .2, .35), t=(0, -1.4, 0))]


def _bridge():
    return [part_box(0, 0, 0, 4, .2, .9), part_box(-1.6, .9, 0, .3, 2, .3), part_box(1.6, .9, 0, .3, 2, .3),
            part_box(0, 1.9, 0, 3.5, .15, .15)]


def _windmill():
    return [part_prism(8, 2, .7, .4), tf(part_prism(8, .6, .5, 0), t=(0, 1.3, 0))] + \
           [tf(part_box(0, .75, 0, .15, 1.5, .05), rz=90 * k, t=(0, .6, .55)) for k in range(4)]


def _house():
    return [part_box(0, .5, 0, 2, 1, 1.6), tf(part_prism(4, 1, 1.45, 0, 45), s=(1, 1, .8), t=(0, 1.5, 0)),
            part_box(0, .3, .82, .4, .6, .05)]


def _tree():
    return [part_prism(8, 1.2, .2), tf(part_prism(10, 1.0, .9, 0), t=(0, .9, 0)),
            tf(part_prism(10, 1.0, .7, 0), t=(0, 1.5, 0)), tf(part_prism(10, 1.0, .5, 0), t=(0, 2.1, 0))]


def _pagoda():
    return [tf(part_prism(4, .4, 1.5 - .3 * k, 1.1 - .3 * k, 45), t=(0, .45 * k, 0)) for k in range(4)] + \
           [tf(part_prism(4, .6, .3, 0, 45), t=(0, 1.85, 0))]


def _arch():
    return [part_box(-1, .8, 0, .4, 1.6, .5), part_box(1, .8, 0, .4, 1.6, .5), part_box(0, 1.75, 0, 2.8, .3, .5)]


def _stairs():
    return [part_box(0, .15 * (k + 1), -.4 * k, 1.6, .3 * (k + 1), .4) for k in range(6)]


def _cross():
    return [part_box(0, 1, 0, .4, 2.4, .4), part_box(0, 1.4, 0, 1.6, .4, .4)]


MODELS = [("Table", _table), ("Chair", _chair), ("Bed", _bed), ("Skyscraper", _skyscraper), ("Castle", _castle),
          ("Ziggurat", _ziggurat), ("Snowman", _snowman), ("Rocket Ship", _rocket), ("Robot", _robot),
          ("Lamp Post", _lamppost), ("Bridge", _bridge), ("Windmill", _windmill), ("Cottage", _house),
          ("Pine Tree", _tree), ("Pagoda", _pagoda), ("Stone Arch", _arch), ("Staircase", _stairs),
          ("Grave Cross", _cross)]

VIEWS = [("Iso", (25, 45, 0)), ("Iso-R", (25, -45, 0)), ("Front-up", (20, 0, 0)), ("Top-down", (60, 20, 0)),
         ("Low", (-20, 35, 0)), ("Side", (10, 90, 0)), ("Tilt-30", (30, 30, 15)), ("Tilt-60", (35, 60, -15)),
         ("Spin-120", (25, 120, 0)), ("Dutch", (25, 45, 20)), ("Bird", (70, 45, 0)), ("Worm", (-45, 30, 0))]

_MESH_CACHE = {}


def _get(key, builder):
    if key not in _MESH_CACHE:
        _MESH_CACHE[key] = build_mesh(builder())
    return _MESH_CACHE[key]


def _solid_fn(key, builder, view):
    return lambda c: solid_prims(_get(key, builder), view)


def _wire_fn(key, builder, view):
    return lambda c: wire_prims(_get(key, builder), view)


def reg(name, cat, fn, square=False, wire=None):
    spec = dict(name=name, fn=fn, square=square, directional=False, cat=cat)
    if wire is not None:
        spec["wire"] = wire
    SHAPES.append(spec)


_BASE_COUNT = len(SHAPES)

# ---- A) Solid 3D (30 x 12 = 360) and B) Wire 3D (30 x 6 = 180)
for _mn, _mb in BASE_SOLIDS:
    for _vn, _v in VIEWS:
        reg("Solid %s %s" % (_mn, _vn), "Solid 3D", _solid_fn(_mn, _mb, _v), True, wire=_wire_fn(_mn, _mb, _v))
    for _i in (0, 1, 3, 5, 6, 9):
        reg("Wire %s %s" % (_mn, VIEWS[_i][0]), "Wire 3D", _wire_fn(_mn, _mb, VIEWS[_i][1]), True)

# ---- C) Model 3D (18 x 4 = 72)
for _mn, _mb in MODELS:
    for _i in (0, 1, 8, 10):
        reg("Model %s %s" % (_mn, VIEWS[_i][0]), "Model 3D", _solid_fn("M" + _mn, _mb, VIEWS[_i][1]), True,
            wire=_wire_fn("M" + _mn, _mb, VIEWS[_i][1]))


# =====================================================================================
#  NEW 1000 SHAPES - PART 2: 2D FAMILIES (curves, patterns, fractals, badges) + voxels
# =====================================================================================
def _gcd(a, b):
    return math.gcd(int(a), int(b))


# ---- D) Curves (120)
for _R, _r in [(5, 3), (7, 3), (8, 3), (9, 4), (11, 4), (7, 2), (9, 2), (8, 5), (10, 3), (11, 6)]:
    for _k, _dk in enumerate((.6, 1.0, 1.5)):
        def _hypo(c, R=_R, r=_r, d=_dk * _r):
            k = (R - r) / r
            return [L(param(lambda t: ((R - r) * math.cos(t) + d * math.cos(k * t),
                                       (R - r) * math.sin(t) - d * math.sin(k * t)),
                            700, 2 * math.pi * r / _gcd(R, r)))]
        reg("Hypotrochoid %d/%d pen%d" % (_R, _r, _k + 1), "Curves", _hypo, True)

for _R, _r in [(5, 3), (7, 3), (8, 3), (9, 4), (11, 4), (7, 2), (9, 2), (8, 5), (10, 3), (11, 6)]:
    for _k, _dk in enumerate((.6, 1.0, 1.5)):
        def _epi(c, R=_R, r=_r, d=_dk * _r):
            k = (R + r) / r
            return [L(param(lambda t: ((R + r) * math.cos(t) - d * math.cos(k * t),
                                       (R + r) * math.sin(t) - d * math.sin(k * t)),
                            700, 2 * math.pi * r / _gcd(R, r)))]
        reg("Epitrochoid %d/%d pen%d" % (_R, _r, _k + 1), "Curves", _epi, True)

for _n, _d in [(2, 1), (3, 1), (4, 1), (5, 1), (6, 1), (7, 1), (8, 1), (3, 2), (5, 2), (7, 2), (5, 3), (7, 3), (8, 3), (9, 4)]:
    def _rose(c, n=_n, d=_d):
        k = n / d
        return [P(param(lambda t: (math.cos(k * t) * math.cos(t), math.cos(k * t) * math.sin(t)), 600, 2 * math.pi * d))]
    reg("Rhodonea rose %d/%d" % (_n, _d), "Curves", _rose, True)

for _a, _b in [(1, 2), (1, 3), (1, 4), (1, 5), (2, 3), (2, 5), (3, 4), (3, 5), (4, 5), (3, 7), (5, 6), (4, 7)]:
    for _di, _dl in enumerate((math.pi / 2, math.pi / 4)):
        def _lis(c, a=_a, b=_b, dl=_dl):
            return [L(param(lambda t: (math.sin(a * t + dl), math.sin(b * t)), 600))]
        reg("Lissajous knot %d:%d phase%d" % (_a, _b, _di + 1), "Curves", _lis, True)

for _n in (.3, .5, .8, 1.5, 2.5, 3, 4, 6, 8, 12):
    def _sup(c, n=_n):
        sg = lambda v: math.copysign(abs(v) ** (2 / n), v)
        return [P(param(lambda t: (sg(math.cos(t)), sg(math.sin(t))), 240))]
    reg("Superellipse n=%s" % _n, "Curves", _sup, False)

for _k in range(3, 15):
    def _epc(c, k=_k):
        return [P(param(lambda t: ((k + 1) * math.cos(t) - math.cos((k + 1) * t),
                                   (k + 1) * math.sin(t) - math.sin((k + 1) * t)), 500))]
    reg("Epicycloid %d cusps" % _k, "Curves", _epc, True)


# ---- E) Patterns (100)
def _norm_polys(polys, pad=.02):
    allp = np.vstack([np.array(p, float) for p in polys])
    mn, mx = allp.min(0), allp.max(0)
    s = (1 - 2 * pad) / max((mx - mn).max(), 1e-9)
    mid = (mn + mx) / 2
    return [[tuple((np.array(q) - mid) * s + .5) for q in p] for p in polys]


for _n in range(3, 13):
    for _lv in (4, 6, 8, 10, 12):
        def _tun(c, n=_n, lv=_lv):
            return [S(ngon(n, .5, .5, .5 * (1 - .9 * j / lv), .5 * (1 - .9 * j / lv),
                           rot=-90 + j * (180.0 / n) / (lv / 2))) for j in range(lv)]
        reg("Tunnel %d-gon x%d" % (_n, _lv), "Pattern", _tun, True)

for _m in (6, 8, 12, 16, 24):
    for _rg in range(4):
        def _spk(c, m=_m, rg=_rg):
            out = [S(circ(.5, .5, .5, 48))]
            out += [S(circ(.5, .5, .5 * k / (rg + 1), 40)) for k in range(1, rg + 1)]
            out += [L([(.5, .5), (.5 + .5 * math.cos(2 * math.pi * i / m), .5 + .5 * math.sin(2 * math.pi * i / m))]) for i in range(m)]
            return out
        reg("Spokes %d rings %d" % (_m, _rg), "Pattern", _spk, True)

for _i, _j in [(2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (8, 8), (3, 5), (4, 6), (2, 6), (5, 8)]:
    def _grid(c, i=_i, j=_j):
        return [L([(k / i, 0), (k / i, 1)]) for k in range(i + 1)] + [L([(0, k / j), (1, k / j)]) for k in range(j + 1)]
    reg("Grid %dx%d" % (_i, _j), "Pattern", _grid, False)

for _n in range(2, 7):
    def _tri(c, n=_n):
        out = []
        for r in range(n):
            for k in range(r + 1):
                out.append(S([(.5 + (k - r / 2) / n, r / n), (.5 + (k - r / 2 - .5) / n, (r + 1) / n),
                              (.5 + (k - r / 2 + .5) / n, (r + 1) / n)]))
        return out
    reg("Triangle grid %d" % _n, "Pattern", _tri, False)

for _i, _j in [(2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (3, 5), (4, 7), (6, 9)]:
    def _dots(c, i=_i, j=_j):
        return [P(circ((k + .5) / i, (l + .5) / j, .38 / max(i, j), 16)) for k in range(i) for l in range(j)]
    reg("Dot lattice %dx%d" % (_i, _j), "Pattern", _dots, False)

for _k in range(1, 6):
    def _hive(c, k=_k):
        polys = []
        for q in range(-k, k + 1):
            for r in range(-k, k + 1):
                if abs(q + r) > k:
                    continue
                cx, cy = math.sqrt(3) * (q + r / 2), 1.5 * r
                polys.append([(cx + math.cos(math.radians(30 + 60 * i)), cy + math.sin(math.radians(30 + 60 * i))) for i in range(6)])
        return [P(p) for p in _norm_polys(polys)]
    reg("Hive %d layers" % _k, "Pattern", _hive, True)


# ---- F) Fractals (40)
def _norm_segs(segs, pad=.02):
    arr = np.array(segs, float).reshape(-1, 2)
    mn, mx = arr.min(0), arr.max(0)
    s = (1 - 2 * pad) / max((mx - mn).max(), 1e-9)
    mid = (mn + mx) / 2
    return [L([tuple((np.array(p) - mid) * s + .5), tuple((np.array(q) - mid) * s + .5)]) for p, q in segs]


for _d in range(1, 6):
    reg("Sierpinski triangle depth %d" % _d, "Fractal", lambda c, d=_d: [P(t) for t in _sier((.5, 0), (1, 1), (0, 1), d)], True)


def _carpet(x, y, s, d):
    if d == 0:
        return [rect(x, y, x + s, y + s)]
    k, out = s / 3, []
    for i in range(3):
        for j in range(3):
            if (i, j) != (1, 1):
                out += _carpet(x + i * k, y + j * k, k, d - 1)
    return out


for _d in range(1, 4):
    reg("Sierpinski carpet depth %d" % _d, "Fractal", lambda c, d=_d: [P(r) for r in _carpet(0, 0, 1, d)], True)

for _d in range(1, 6):
    def _flake(c, d=_d):
        v = [complex(0, .75), complex(1, .75), complex(.5, .75 - .866)]
        pts = []
        for i in range(3):
            pts += _koch(v[i], v[(i + 1) % 3], d)
        return [P(fit([(p.real, p.imag) for p in pts]))]
    reg("Koch flake depth %d" % _d, "Fractal", _flake, True)

for _d in range(1, 6):
    def _kcurve(c, d=_d):
        pts = _koch(complex(0, 0), complex(1, 0), d) + [complex(1, 0)]
        return [L(fit([(p.real, p.imag) for p in pts], keep=False))]
    reg("Koch curve depth %d" % _d, "Fractal", _kcurve, False)


def _tree_segs(spread, depth=7, shrink=.72):
    out = []

    def rec(p, ang, ln, d):
        if d == 0:
            return
        q = (p[0] + ln * math.sin(ang), p[1] - ln * math.cos(ang))
        out.append((p, q))
        rec(q, ang - spread, ln * shrink, d - 1)
        rec(q, ang + spread, ln * shrink, d - 1)
    rec((0., 0.), 0., 1., depth)
    return out


for _a in (15, 20, 25, 30, 35, 40, 45):
    reg("Fractal tree %d deg" % _a, "Fractal", lambda c, a=_a: _norm_segs(_tree_segs(math.radians(a))), True)


def _dragon(n):
    turns = []
    for _ in range(n):
        turns = turns + [1] + [-t for t in reversed(turns)]
    x, y, dx, dy = 1, 0, 1, 0
    pts = [(0, 0), (1, 0)]
    for t in turns:
        dx, dy = -t * dy, t * dx
        x, y = x + dx, y + dy
        pts.append((x, y))
    return pts


for _n in range(6, 12):
    reg("Dragon curve iter %d" % _n, "Fractal", lambda c, n=_n: [L(fit(_dragon(n)))], True)


def _hilbert(order):
    n = 2 ** order
    pts = []
    for d in range(n * n):
        x = y = 0
        t, s = d, 1
        while s < n:
            rx = 1 & (t // 2)
            ry = 1 & (t ^ rx)
            if ry == 0:
                if rx == 1:
                    x, y = s - 1 - x, s - 1 - y
                x, y = y, x
            x += s * rx
            y += s * ry
            t //= 4
            s *= 2
        pts.append((x, y))
    return pts


for _o in range(1, 6):
    reg("Hilbert curve order %d" % _o, "Fractal", lambda c, o=_o: [L(fit(_hilbert(o)))], True)


def _levy(p, q, d):
    if d == 0:
        return [p]
    m = p + (q - p) * complex(.5, -.5)
    return _levy(p, m, d - 1) + _levy(m, q, d - 1)


for _d in range(8, 12):
    def _lv(c, d=_d):
        pts = _levy(complex(0, 0), complex(1, 0), d) + [complex(1, 0)]
        return [L(fit([(p.real, p.imag) for p in pts]))]
    reg("Levy C curve iter %d" % _d, "Fractal", _lv, True)

# ---- G) Badges (100)
_INNER = [
    ("disc", lambda n: [P(circ(.5, .5, .28, 32), 1.2)]),
    ("star5", lambda n: [P(sc(star(5, .4), .62), 1.2)]),
    ("starN", lambda n: [P(sc(star(max(n, 4), .5), .62), 1.2)]),
    ("plus", lambda n: [P(sc(PLUS, .55), 1.2)]),
    ("diamond", lambda n: [P(sc([(.5, 0), (1, .5), (.5, 1), (0, .5)], .6), 1.2)]),
    ("ring", lambda n: [S(circ(.5, .5, .3, 32)), S(circ(.5, .5, .18, 28))]),
    ("cross", lambda n: [L([(.28, .28), (.72, .72)]), L([(.72, .28), (.28, .72)])]),
    ("inner-gon", lambda n: [P(sc(ngon(n, rot=90), .55), 1.2)]),
    ("dots", lambda n: [DOT(circ(.5 + .27 * math.cos(2 * math.pi * i / n), .5 + .27 * math.sin(2 * math.pi * i / n), .06, 12), 1.2)
                        for i in range(n)]),
    ("arrow", lambda n: [P(sc(ARROW_R, .55), 1.2)]),
]
for _n in range(3, 13):
    for _tag, _in in _INNER:
        reg("Badge %d-gon %s" % (_n, _tag), "Badge",
            (lambda c, n=_n, f=_in: [P(ngon(n))] + f(n)), True)

# ---- H) Voxel 3D  (fills the new pack up to exactly +1000)
_need = 1000 - (len(SHAPES) - _BASE_COUNT)
_rs = np.random.RandomState(7)
_seen, _k = set(), 0
while _k < _need:
    _h = tuple(int(v) for v in _rs.randint(0, 4, 9))
    if _h in _seen or sum(1 for v in _h if v) < 3:
        continue
    _seen.add(_h)
    _k += 1
    _vb = (lambda h=_h: [part_box((i - 1) * 1.1, h[i * 3 + j] / 2, (j - 1) * 1.1, 1, h[i * 3 + j], 1)
                        for i in range(3) for j in range(3) if h[i * 3 + j]])
    _view = VIEWS[(0, 1, 10, 4)[_k % 4]][1]
    reg("Voxel city %02d" % _k, "Voxel 3D", _solid_fn("V%d" % _k, _vb, _view), True, wire=_wire_fn("V%d" % _k, _vb, _view))

assert len(SHAPES) == 2000, "expected 2000 shapes, got %d" % len(SHAPES)
assert len({s["name"] for s in SHAPES}) == 2000, "shape names must be unique"

# =====================================================================================
#  SHAPE RENDERING
# =====================================================================================
def box_from(a, b, square):
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    if square:
        s = max(abs(dx), abs(dy))
        bx = ax + math.copysign(s, dx if dx else 1)
        by = ay + math.copysign(s, dy if dy else 1)
    x1, x2 = sorted((ax, bx))
    y1, y2 = sorted((ay, by))
    return (x1, y1, x2, y2), bx < ax, by < ay


def shade_col(col, s):
    if np.isscalar(col):
        return col
    out = []
    for ch in col:
        v = ch * s if s <= 1 else ch + (255 - ch) * (s - 1)
        out.append(int(max(0, min(255, v))))
    return tuple(out)


def render_shape(img, idx, a, b, color, th, mode, is_mask=False):
    """mode: 0 outline, 1 fill, 2 outline+fill. is_mask draws 255 on a single-channel mask."""
    spec = SHAPES[idx]
    (x1, y1, x2, y2), fx, fy = box_from(a, b, spec["square"])
    w, h = x2 - x1, y2 - y1
    if w < 2 and h < 2:
        return
    th = max(1, int(th))
    prims = spec["fn"](Ctx(max(w, 1.0), max(h, 1.0), th))
    has_fill = any(p["shade"] is not None for p in prims)
    base = 255 if is_mask else color
    dark = 255 if is_mask else shade_col(color, .55)
    for p in prims:
        pts = p["pts"].copy()
        if spec["directional"]:
            if fx:
                pts[:, 0] = 1 - pts[:, 0]
            if fy:
                pts[:, 1] = 1 - pts[:, 1]
        pts[:, 0] = x1 + pts[:, 0] * w
        pts[:, 1] = y1 + pts[:, 1] * h
        q = np.round(pts * 16).astype(np.int32).reshape(-1, 1, 2)
        closed, shade = p["closed"], p["shade"]
        if shade is not None and (mode in (1, 2) or p["solid"]):
            cv2.fillPoly(img, [q], shade_col(base, shade), cv2.LINE_AA, 4)
        if p["stroke"] and (not closed or mode in (0, 2)):
            if mode == 0 or not has_fill:
                col = base
            else:
                col = dark
            cv2.polylines(img, [q], closed, col, th, cv2.LINE_AA, 4)


# ---- art styles + wireframe mode: wraps render_shape (everything below calls the wrapper)
_orig_render = render_shape


def _base_draw(img, idx, a, b, color, th, mode, is_mask):
    if mode == 3:                                   # wireframe: 3D shapes show all their edges
        spec = SHAPES[idx]
        w = spec.get("wire")
        if w is not None:
            SHAPES[idx] = dict(spec, fn=w)
            try:
                _orig_render(img, idx, a, b, color, th, 0, is_mask)
            finally:
                SHAPES[idx] = spec
        else:
            _orig_render(img, idx, a, b, color, th, 0, is_mask)
    else:
        _orig_render(img, idx, a, b, color, th, mode, is_mask)


def _mix(col, t):
    return tuple(int(c + (255 - c) * t) for c in col)


def styled_render(img, idx, a, b, color, th, mode, is_mask=False):
    style = STYLES[STATE["style"]]
    if not is_mask and th == 2 and mode == 0 and tuple(color) == (255, 255, 255):
        style = "Clean"                              # palette thumbnails stay plain
    D = _base_draw
    if style == "Clean":
        D(img, idx, a, b, color, th, mode, is_mask)
    elif style == "Neon":
        D(img, idx, a, b, shade_col(color, .35), th * 4 + 2, 0, is_mask)
        D(img, idx, a, b, shade_col(color, .6), th * 2 + 1, 0, is_mask)
        D(img, idx, a, b, _mix(color, .55), th, mode, is_mask)
    elif style == "Sketch":
        rs = np.random.RandomState(idx * 7 + 3)
        for k in range(2):
            o = rs.uniform(-3, 3, 4)
            D(img, idx, (a[0] + o[0], a[1] + o[1]), (b[0] + o[2], b[1] + o[3]), color, max(1, th // 2),
              mode if k == 0 else (3 if mode == 3 else 0), is_mask)
    elif style == "Comic":
        if mode in (1, 2):
            D(img, idx, a, b, color, th, 1, is_mask)
        D(img, idx, a, b, (0, 0, 0), th + 3, 0 if mode != 3 else 3, is_mask)
        D(img, idx, a, b, color, max(1, th - 1), 0 if mode != 3 else 3, is_mask)
    elif style == "Shadow":
        o = 10
        D(img, idx, (a[0] + o, a[1] + o), (b[0] + o, b[1] + o), shade_col(color, .25), th,
          1 if mode in (1, 2) else 0, is_mask)
        D(img, idx, a, b, color, th, mode, is_mask)
    elif style == "Pastel":
        D(img, idx, a, b, _mix(color, .6), th, mode, is_mask)
    elif style == "Blueprint":
        D(img, idx, a, b, (255, 190, 70), max(1, th // 2), 3, is_mask)
    elif style == "Double":
        D(img, idx, a, b, color, th, mode, is_mask)
        sx = math.copysign(1, b[0] - a[0] or 1)
        sy = math.copysign(1, b[1] - a[1] or 1)
        d = min(14, abs(b[0] - a[0]) / 4, abs(b[1] - a[1]) / 4)
        if d >= 3:
            D(img, idx, (a[0] + sx * d, a[1] + sy * d), (b[0] - sx * d, b[1] - sy * d), color, max(1, th // 2),
              3 if mode == 3 else 0, is_mask)


render_shape = styled_render


# =====================================================================================
#  CANVAS  (premultiplied colour layer + coverage mask => correct transparency)
# =====================================================================================
class Stroke:
    def __init__(self, canvas, tool, color, size):
        self.tool, self.color, self.size = tool, color, size
        self.last = None
        if tool == "highlighter":
            self.base_layer = canvas.layer.copy()
            self.base_mask = canvas.mask.copy()
            self.cov = np.zeros((canvas.h, canvas.w), np.uint8)


class Canvas:
    def __init__(self, w, h):
        self.w, self.h = w, h
        self.layer = np.zeros((h, w, 3), np.uint8)
        self.mask = np.zeros((h, w), np.uint8)
        self.undo, self.redo = [], []

    # ---- history
    def push_undo(self):
        self.undo.append((self.layer.copy(), self.mask.copy()))
        del self.undo[:-MAX_UNDO]
        self.redo.clear()

    def undo_step(self):
        if self.undo:
            self.redo.append((self.layer.copy(), self.mask.copy()))
            self.layer, self.mask = self.undo.pop()
            return True
        return False

    def redo_step(self):
        if self.redo:
            self.undo.append((self.layer.copy(), self.mask.copy()))
            self.layer, self.mask = self.redo.pop()
            return True
        return False

    def clear(self):
        self.push_undo()
        self.layer[:] = 0
        self.mask[:] = 0

    # ---- helpers
    def _roi(self, p0, p1, pad):
        x0, y0 = max(min(p0[0], p1[0]) - pad, 0), max(min(p0[1], p1[1]) - pad, 0)
        x1, y1 = min(max(p0[0], p1[0]) + pad + 1, self.w), min(max(p0[1], p1[1]) + pad + 1, self.h)
        return (x0, y0, x1, y1) if x1 > x0 and y1 > y0 else None

    @staticmethod
    def _line_cov(p0, p1, th, r):
        x0, y0, x1, y1 = r
        tmp = np.zeros((y1 - y0, x1 - x0), np.uint8)
        cv2.line(tmp, (p0[0] - x0, p0[1] - y0), (p1[0] - x0, p1[1] - y0), 255, th, cv2.LINE_AA)
        return tmp.astype(np.float32) / 255.0

    def _over(self, r, cov, color, alpha):
        x0, y0, x1, y1 = r
        a = (cov * alpha)[..., None]
        self.layer[y0:y1, x0:x1] = (self.layer[y0:y1, x0:x1] * (1 - a) + np.array(color, np.float32) * a).astype(np.uint8)
        m = self.mask[y0:y1, x0:x1].astype(np.float32)
        self.mask[y0:y1, x0:x1] = (m * (1 - a[..., 0]) + 255 * a[..., 0]).astype(np.uint8)

    # ---- one stroke segment
    def seg(self, st, p0, p1):
        tool, color, size = st.tool, st.color, st.size
        if tool in ("pen", "marker", "eraser"):
            th = {"pen": size, "marker": max(2, int(size * 1.8)), "eraser": max(8, int(size * 3))}[tool]
            col, m = ((0, 0, 0), 0) if tool == "eraser" else (color, 255)
            cv2.line(self.layer, p0, p1, col, th, cv2.LINE_AA)
            cv2.line(self.mask, p0, p1, m, th, cv2.LINE_AA)
        elif tool == "pencil":
            th = max(1, int(size * .6))
            r = self._roi(p0, p1, th + 2)
            if r:
                cov = self._line_cov(p0, p1, th, r)
                cov *= (0.55 + 0.45 * np.random.rand(*cov.shape)).astype(np.float32)
                self._over(r, cov, color, .9)
        elif tool == "brush":
            for wm, al in ((1.0, .10), (.7, .15), (.4, .22)):
                th = max(2, int(size * 2 * wm))
                r = self._roi(p0, p1, int(size * 2) + 2)
                if r:
                    self._over(r, self._line_cov(p0, p1, th, r), color, al)
        elif tool == "highlighter":
            th = max(8, int(size * 3))
            cv2.line(st.cov, p0, p1, 255, th, cv2.LINE_AA)
            r = self._roi(p0, p1, th + 2)
            if r:
                x0, y0, x1, y1 = r
                a = (st.cov[y0:y1, x0:x1].astype(np.float32) / 255.0 * .38)[..., None]
                self.layer[y0:y1, x0:x1] = (st.base_layer[y0:y1, x0:x1] * (1 - a) +
                                            np.array(color, np.float32) * a).astype(np.uint8)
                self.mask[y0:y1, x0:x1] = (st.base_mask[y0:y1, x0:x1].astype(np.float32) * (1 - a[..., 0]) +
                                           255 * a[..., 0]).astype(np.uint8)

    def stamp_shape(self, idx, a, b, color, th, mode):
        self.push_undo()
        render_shape(self.layer, idx, a, b, color, th, mode, False)
        render_shape(self.mask, idx, a, b, color, th, mode, True)

    # ---- output
    def compose(self, bg):
        inv = cv2.cvtColor(cv2.bitwise_not(self.mask), cv2.COLOR_GRAY2BGR)
        return cv2.add(cv2.multiply(bg, inv, scale=1 / 255.0), self.layer)

    def bgra(self):
        a = np.maximum(self.mask, 1).astype(np.float32)[..., None] / 255.0
        bgr = np.clip(self.layer.astype(np.float32) / a, 0, 255).astype(np.uint8)
        bgr[self.mask == 0] = 0
        return np.dstack([bgr, self.mask])


# =====================================================================================
#  APP  (state, UI panel, gesture logic)
# =====================================================================================
def new_hand_state():
    return dict(ptr=None, pinch_prev=False, nopoint=0, stroke=None, missing=0)


class App:
    def __init__(self, w, h):
        self.w, self.h = w, h
        self.canvas = Canvas(w, h)
        self.tool, self.mode = "pen", "draw"
        self.color_idx, self.color = 0, COLORS[0][1]
        self.size, self.outline_th = 6, 4
        self.shape_idx = next(i for i, s in enumerate(SHAPES) if s["name"] == "Circle")
        self.fill_mode = 0
        self.paper, self.skeleton, self.palette, self.page = False, True, False, 0
        self.hs = {"Left": new_hand_state(), "Right": new_hand_state()}
        self.hands = {}
        self.group_open = False
        self.shape_drag = None
        self.preview = None
        self.shape_cd = 0.0
        self.last_click = 0.0
        self.flash = (None, 0.0)
        self.toast = ("", 0.0)
        self.hover = {}
        self.panel_h, self.pal_bbox, self.ui = 0, None, []
        self.paper_img = np.full((h, w, 3), 255, np.uint8)
        cv2.rectangle(self.paper_img, (0, 0), (w - 1, h - 1), (190, 190, 190), 3)
        self._hue_cache = {}
        # shape categories / palette filter
        self.cats = ["All"] + sorted({s["cat"] for s in SHAPES})
        self.cat_i = 0
        self.view_list = list(range(len(SHAPES)))

    # ------------------------------------------------------------ UI layout
    def build_ui(self):
        w = self.w
        btns, pad, gap, rowh = [], 6, 6, 44
        pos = [pad, pad]

        def add(bid, label, wd, kind="btn"):
            if pos[0] + wd > w - pad:
                pos[0] = pad
                pos[1] += rowh + gap
            btns.append(dict(id=bid, label=label, kind=kind,
                             rect=(pos[0], pos[1], pos[0] + wd, pos[1] + rowh)))
            pos[0] += wd + gap

        for tid, tname in TOOLS:
            add("tool:" + tid, tname, 92)
        add("mode", "Mode: " + self.mode.capitalize(), 126)
        add("shapes", "Shape: " + SHAPES[self.shape_idx]["name"][:14], 200)
        add("fill", FILL_NAMES[self.fill_mode], 130)
        add("paper", "Paper: " + ("On" if self.paper else "Off"), 112)
        for bid, lab in (("undo", "Undo"), ("redo", "Redo"), ("clear", "Clear"), ("save", "Save"),
                         ("size-", "Size -"), ("size+", "Size +"), ("skel", "Skeleton")):
            add(bid, lab, 82)
        for i in range(len(COLORS)):
            add("color:%d" % i, "", 34, "color")
        add("hue", "", 150, "hue")
        # extra row: art style, 3D rotation, category filter
        add("pstyle", "Style: " + STYLES[STATE["style"]], 190)
        add("rotx", "Rot X", 84)
        add("roty", "Rot Y", 84)
        add("rotz", "Rot Z", 84)
        add("rotr", "Reset 3D", 110)
        add("cat", "Cat: " + self.cats[self.cat_i][:12], 190)
        self.panel_h = pos[1] + rowh + pad
        self.ui = btns
        self.pal_bbox = None
        if self.palette:
            self._build_palette()
        return btns

    def _palette_geom(self):
        cols = max(4, (self.w - 16) // 100)
        rows = 4
        return cols, rows, (self.w - 16) // cols, 84

    def _build_palette(self):
        cols, rows, cw, ch = self._palette_geom()
        per = cols * rows
        idxs = self.view_list
        pages = max(1, math.ceil(len(idxs) / per))
        self.page = max(0, min(self.page, pages - 1))
        top = self.panel_h + 4
        for k in range(per):
            n = self.page * per + k
            if n >= len(idxs):
                break
            i = idxs[n]
            r, c = divmod(k, cols)
            x, y = 8 + c * cw, top + 8 + r * ch
            self.ui.append(dict(id="shape:%d" % i, label=SHAPES[i]["name"], kind="cell",
                                rect=(x, y, x + cw - 4, y + ch - 4)))
        ny = top + 8 + rows * ch
        for j, (bid, lab) in enumerate((("pal_prev", "< Prev"), ("pal_next", "Next >"), ("pal_close", "Close"))):
            self.ui.append(dict(id=bid, label=lab, kind="btn", rect=(8 + j * 130, ny, 8 + j * 130 + 124, ny + 40)))
        self.pal_bbox = (0, top, self.w, ny + 48)

    def in_ui(self, p):
        if p[1] < self.panel_h:
            return True
        b = self.pal_bbox
        return bool(self.palette and b and b[1] <= p[1] <= b[3])

    @staticmethod
    def hit(btns, p):
        for b in reversed(btns):
            x1, y1, x2, y2 = b["rect"]
            if x1 <= p[0] <= x2 and y1 <= p[1] <= y2:
                return b
        return None

    # ------------------------------------------------------------ actions
    def say(self, text, secs=1.6):
        self.toast = (text, time.time() + secs)

    def change_size(self, d):
        if self.mode == "shape":
            self.outline_th = int(np.clip(self.outline_th + d, 1, 30))
            self.say("Outline thickness: %d" % self.outline_th)
        else:
            self.size = int(np.clip(self.size + 2 * d, 1, 60))
            self.say("Brush size: %d" % self.size)

    def set_color(self, i):
        self.color_idx, self.color = i, COLORS[i][1]
        self.say("Colour: " + COLORS[i][0])

    def next_tool(self):
        ids = [t[0] for t in TOOLS]
        self.tool = ids[(ids.index(self.tool) + 1) % len(ids)]
        self.mode = "draw"
        self.say("Tool: " + self.tool)

    def step_shape(self, d):
        self.shape_idx = (self.shape_idx + d) % len(SHAPES)
        self.mode = "shape"
        self.say("Shape: " + SHAPES[self.shape_idx]["name"])

    def set_cat(self, i):
        self.cat_i = i % len(self.cats)
        cat = self.cats[self.cat_i]
        self.view_list = [k for k, s in enumerate(SHAPES) if cat == "All" or s["cat"] == cat]
        self.page = 0
        self.say("Category: %s (%d shapes)" % (cat, len(self.view_list)))

    def next_style(self):
        STATE["style"] = (STATE["style"] + 1) % len(STYLES)
        self.say("Art style: " + STYLES[STATE["style"]])

    def rotate3d(self, axis, d):
        STATE["rot"][axis] = (STATE["rot"][axis] + d) % 360
        self.say("3D rotation  X %d  Y %d  Z %d" % tuple(STATE["rot"]))

    def cycle_fill(self):
        self.fill_mode = (self.fill_mode + 1) % len(FILL_NAMES)
        self.say("Mode: " + FILL_NAMES[self.fill_mode])

    def save(self):
        os.makedirs(SAVE_DIR, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        if self.paper:
            out = self.canvas.compose(self.paper_img)
        else:
            out = self.canvas.bgra()
        path = os.path.join(SAVE_DIR, "airdraw_%s.png" % stamp)
        cv2.imwrite(path, out)
        self.say("Saved " + path, 2.5)
        return path

    def activate(self, bid):
        if bid.startswith("tool:"):
            self.tool, self.mode = bid[5:], "draw"
        elif bid == "mode":
            self.mode = "shape" if self.mode == "draw" else "draw"
        elif bid == "shapes":
            self.palette = not self.palette
        elif bid == "fill":
            self.cycle_fill()
        elif bid == "paper":
            self.paper = not self.paper
        elif bid == "undo":
            self.say("Undo" if self.canvas.undo_step() else "Nothing to undo")
        elif bid == "redo":
            self.say("Redo" if self.canvas.redo_step() else "Nothing to redo")
        elif bid == "clear":
            self.canvas.clear()
            self.say("Canvas cleared")
        elif bid == "save":
            self.save()
        elif bid == "skel":
            self.skeleton = not self.skeleton
        elif bid == "size-":
            self.change_size(-1)
        elif bid == "size+":
            self.change_size(+1)
        elif bid.startswith("color:"):
            self.set_color(int(bid[6:]))
        elif bid.startswith("shape:"):
            self.shape_idx, self.mode, self.palette = int(bid[6:]), "shape", False
        elif bid == "pal_prev":
            self.page -= 1
        elif bid == "pal_next":
            self.page += 1
        elif bid == "pal_close":
            self.palette = False
        elif bid == "pstyle":
            self.next_style()
        elif bid in ("rotx", "roty", "rotz"):
            self.rotate3d("xyz".index(bid[3]), 15)
        elif bid == "rotr":
            STATE["rot"][:] = [0, 0, 0]
            self.say("3D rotation reset")
        elif bid == "cat":
            self.set_cat(self.cat_i + 1)

    # ------------------------------------------------------------ gesture logic
    def _end_stroke(self, lab):
        st = self.hs[lab]
        st["stroke"] = None
        st["nopoint"] = 0
        if not any(s["stroke"] for s in self.hs.values()):
            self.group_open = False

    def _lost(self, lab):
        st = self.hs[lab]
        self._end_stroke(lab)
        st["ptr"], st["pinch_prev"] = None, False
        if self.shape_drag and self.shape_drag["label"] == lab:
            self.shape_drag = None
        if self.preview and self.preview.get("label") == lab:
            self.preview = None

    def _commit(self, a, b):
        spec = SHAPES[self.shape_idx]
        (x1, y1, x2, y2), _, _ = box_from(a, b, spec["square"])
        if max(x2 - x1, y2 - y1) < 12:
            return
        self.canvas.stamp_shape(self.shape_idx, a, b, self.color, self.outline_th, self.fill_mode)

    def update(self, hands, now=None):
        """hands: {'Left'|'Right': dict(tip=(x,y), idx=bool, mid=bool, pinch=bool)}"""
        now = now or time.time()
        self.build_ui()
        self.hands = hands
        for lab in ("Left", "Right"):
            st = self.hs[lab]
            if lab in hands:
                st["missing"] = 0
            else:
                st["missing"] += 1
                if st["missing"] >= 4:
                    self._lost(lab)
        # pointer smoothing (frozen while pinching, so a click does not shift the target)
        for lab, hd in hands.items():
            st = self.hs[lab]
            tip = np.array(hd["tip"], float)
            if st["ptr"] is None:
                st["ptr"] = tip
            elif not hd["pinch"]:
                st["ptr"] = st["ptr"] * .45 + tip * .55
            hd["ptr"] = st["ptr"].copy()
            hd["in_ui"] = self.in_ui(hd["ptr"])
        # panel interaction
        self.hover = {}
        for lab, hd in hands.items():
            st = self.hs[lab]
            if not hd["in_ui"]:
                continue
            self._end_stroke(lab)
            if self.shape_drag and self.shape_drag["label"] == lab:
                self.shape_drag = None
            b = self.hit(self.ui, hd["ptr"])
            self.hover[lab] = b["id"] if b else None
            if b and hd["pinch"]:
                if b["kind"] == "hue":
                    x1, _, x2, _ = b["rect"]
                    hue = int(np.clip((hd["ptr"][0] - x1) / (x2 - x1), 0, .999) * 180)
                    self.color = tuple(int(v) for v in cv2.cvtColor(np.uint8([[[hue, 255, 255]]]),
                                                                    cv2.COLOR_HSV2BGR)[0, 0])
                    self.color_idx = None
                elif not st["pinch_prev"] and now - self.last_click > .45:
                    self.activate(b["id"])
                    self.last_click, self.flash = now, (b["id"], now)
        # drawing / shape creation
        outside = [lab for lab, hd in hands.items() if not hd["in_ui"]]
        self.preview = None
        two_hand = (self.mode == "shape" and len(hands) == 2 and len(outside) == 2 and
                    all(hands[l]["idx"] or hands[l]["pinch"] for l in outside))
        if two_hand:
            if self.shape_drag:
                self.shape_drag = None
            a, b = hands["Left"]["ptr"], hands["Right"]["ptr"]
            self.preview = dict(a=a, b=b, label="Right")
            r = self.hs["Right"]
            if hands["Right"]["pinch"] and not r["pinch_prev"] and now > self.shape_cd:
                self._commit(a, b)
                self.shape_cd = now + .8
                self.preview = None
        else:
            for lab in outside:
                hd, st = hands[lab], self.hs[lab]
                pointing = hd["idx"] and not hd["mid"]
                if lab == "Left":                      # left hand = eraser
                    self._draw_logic(lab, hd, st, pointing)
                elif self.mode == "draw":
                    self._draw_logic(lab, hd, st, pointing)
                else:
                    self._shape_logic(lab, hd, st, pointing, now)
        for lab, hd in hands.items():
            self.hs[lab]["pinch_prev"] = hd["pinch"]

    def _draw_logic(self, lab, hd, st, pointing):
        if pointing:
            st["nopoint"] = 0
            p = (int(hd["ptr"][0]), int(hd["ptr"][1]))
            if st["stroke"] is None:
                if not self.group_open:
                    self.canvas.push_undo()
                    self.group_open = True
                st["stroke"] = Stroke(self.canvas, "eraser" if lab == "Left" else self.tool, self.color, self.size)
                st["stroke"].last = p
                self.canvas.seg(st["stroke"], p, p)
            else:
                s = st["stroke"]
                if math.hypot(p[0] - s.last[0], p[1] - s.last[1]) >= 2:
                    self.canvas.seg(s, s.last, p)
                    s.last = p
        else:
            st["nopoint"] += 1
            if st["nopoint"] >= 3 and st["stroke"] is not None:
                self._end_stroke(lab)

    def _shape_logic(self, lab, hd, st, pointing, now):
        d = self.shape_drag
        if pointing:
            st["nopoint"] = 0
            if d is None and now > self.shape_cd:
                self.shape_drag = d = dict(label=lab, a=hd["ptr"].copy(), b=hd["ptr"].copy())
            if d and d["label"] == lab:
                d["b"] = hd["ptr"].copy()
        else:
            st["nopoint"] += 1
            if d and d["label"] == lab and st["nopoint"] >= 3:
                self._commit(d["a"], d["b"])
                self.shape_drag = None
                self.shape_cd = now + .4
        if self.shape_drag:
            self.preview = dict(a=self.shape_drag["a"], b=self.shape_drag["b"], label=self.shape_drag["label"])

    # ------------------------------------------------------------ rendering
    def render_base(self, frame):
        bg = self.paper_img.copy() if self.paper else frame
        disp = self.canvas.compose(bg)
        if self.paper:
            pw, ph = self.w // 5, self.h // 5
            x0, y0 = self.w - pw - 10, self.h - ph - 44
            disp[y0:y0 + ph, x0:x0 + pw] = cv2.resize(frame, (pw, ph))
            cv2.rectangle(disp, (x0, y0), (x0 + pw, y0 + ph), (120, 120, 120), 2)
        if self.preview:
            a = tuple(float(v) for v in self.preview["a"])
            b = tuple(float(v) for v in self.preview["b"])
            render_shape(disp, self.shape_idx, a, b, self.color, self.outline_th, self.fill_mode)
        return disp

    def _hue_bar(self, wd, ht):
        key = (wd, ht)
        if key not in self._hue_cache:
            hsv = np.zeros((1, wd, 3), np.uint8)
            hsv[0, :, 0] = np.linspace(0, 179, wd).astype(np.uint8)
            hsv[0, :, 1:] = 255
            self._hue_cache[key] = cv2.resize(cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR), (wd, ht),
                                              interpolation=cv2.INTER_NEAREST)
        return self._hue_cache[key]

    @staticmethod
    def _text(img, txt, rect, scale=.5, col=(255, 255, 255)):
        (tw, th), _ = cv2.getTextSize(txt, FONT, scale, 1)
        x = rect[0] + (rect[2] - rect[0] - tw) // 2
        y = rect[1] + (rect[3] - rect[1] + th) // 2
        cv2.putText(img, txt, (x, y), FONT, scale, col, 1, cv2.LINE_AA)

    def _is_active(self, b):
        i = b["id"]
        if i.startswith("tool:"):
            return self.mode == "draw" and self.tool == i[5:]
        if i == "mode":
            return self.mode == "shape"
        if i == "shapes":
            return self.palette
        if i == "paper":
            return self.paper
        if i == "skel":
            return self.skeleton
        if i.startswith("color:"):
            return self.color_idx == int(i[6:])
        if i.startswith("shape:"):
            return int(i[6:]) == self.shape_idx
        return False

    def render_overlay(self, disp):
        now = time.time()
        w, h = self.w, self.h
        ov = disp.copy()
        cv2.rectangle(ov, (0, 0), (w, self.panel_h), (24, 24, 24), -1)
        if self.pal_bbox:
            b = self.pal_bbox
            cv2.rectangle(ov, (b[0], b[1]), (b[2], b[3]), (24, 24, 24), -1)
        cv2.rectangle(ov, (0, h - 34), (w, h), (24, 24, 24), -1)
        disp[:] = cv2.addWeighted(ov, .78, disp, .22, 0)
        hovered = set(v for v in self.hover.values() if v)
        for b in self.ui:
            x1, y1, x2, y2 = [int(v) for v in b["rect"]]
            active, hov = self._is_active(b), b["id"] in hovered
            flash = self.flash[0] == b["id"] and now - self.flash[1] < .25
            if b["kind"] == "color":
                cv2.rectangle(disp, (x1, y1), (x2, y2), COLORS[int(b["id"][6:])][1], -1)
                bc = (255, 255, 255) if active else (110, 110, 110)
                cv2.rectangle(disp, (x1, y1), (x2, y2), bc, 3 if (active or hov) else 1)
            elif b["kind"] == "hue":
                disp[y1:y2, x1:x2] = self._hue_bar(x2 - x1, y2 - y1)
                cv2.rectangle(disp, (x1, y1), (x2, y2), (255, 255, 255) if hov else (110, 110, 110), 2)
                self._text(disp, "Hue", b["rect"], .5, (0, 0, 0))
            else:
                if kind_cell := (b["kind"] == "cell"):
                    fill = (58, 58, 58)
                else:
                    fill = (62, 62, 62)
                if active:
                    fill = (70, 150, 70)
                if hov:
                    fill = (180, 130, 40)
                if flash:
                    fill = (255, 255, 255)
                cv2.rectangle(disp, (x1, y1), (x2, y2), fill, -1)
                cv2.rectangle(disp, (x1, y1), (x2, y2), (120, 120, 120), 1)
                if kind_cell:
                    i = int(b["id"][6:])
                    s = min(x2 - x1 - 20, y2 - y1 - 28)
                    cx = (x1 + x2) // 2
                    render_shape(disp, i, (cx - s // 2, y1 + 6), (cx + s // 2, y1 + 6 + s),
                                 (255, 255, 255), 2, 0)
                    cv2.putText(disp, b["label"][:16], (x1 + 4, y2 - 6), FONT, .34, (230, 230, 230), 1, cv2.LINE_AA)
                    if "3D" in SHAPES[i]["cat"]:
                        cv2.putText(disp, "3D", (x2 - 22, y1 + 14), FONT, .38, (0, 220, 255), 1, cv2.LINE_AA)
                else:
                    self._text(disp, b["label"], b["rect"], .48 if len(b["label"]) < 16 else .44)
        if self.palette and self.pal_bbox:
            cols, rows, _, _ = self._palette_geom()
            pages = max(1, math.ceil(len(self.view_list) / (cols * rows)))
            cv2.putText(disp, "Page %d/%d  -  %d shapes  [%s]" % (self.page + 1, pages, len(self.view_list),
                                                                   self.cats[self.cat_i]),
                        (410, self.pal_bbox[3] - 18), FONT, .55, (255, 255, 255), 1, cv2.LINE_AA)
        # status bar
        status = "%s | Size %d | Line %d | %s | %s | Paper %s | L:%s R:%s" % (
            self.tool if self.mode == "draw" else "shape", self.size, self.outline_th,
            SHAPES[self.shape_idx]["name"], FILL_NAMES[self.fill_mode], "on" if self.paper else "off",
            "ok" if "Left" in self.hands else "-", "ok" if "Right" in self.hands else "-")
        cv2.rectangle(disp, (8, h - 27), (28, h - 7), self.color, -1)
        cv2.rectangle(disp, (8, h - 27), (28, h - 7), (255, 255, 255), 1)
        cv2.putText(disp, status, (38, h - 12), FONT, .5, (235, 235, 235), 1, cv2.LINE_AA)
        cv2.putText(disp, "Style %s | 3D rot %d/%d/%d" % ((STYLES[STATE["style"]],) + tuple(STATE["rot"])),
                    (w - 330, h - 12), FONT, .5, (235, 235, 235), 1, cv2.LINE_AA)
        # pointers
        for lab, hd in self.hands.items():
            p = (int(hd["ptr"][0]), int(hd["ptr"][1]))
            ring = (255, 200, 0) if lab == "Left" else (255, 0, 200)
            erasing = lab == "Left" or self.tool == "eraser"
            if not hd["in_ui"] and (self.mode == "draw" or lab == "Left"):
                rad = max(4, int((self.size * 3 if erasing else self.size) / 2) + 3)
                cv2.circle(disp, p, rad, (255, 255, 255) if erasing else self.color, 1, cv2.LINE_AA)
            cv2.circle(disp, p, 11, ring, -1 if hd["pinch"] else 2, cv2.LINE_AA)
            cv2.circle(disp, p, 3, (255, 255, 255), -1, cv2.LINE_AA)
        # toast
        txt, until = self.toast
        if txt and now < until:
            (tw, th), _ = cv2.getTextSize(txt, FONT, .7, 2)
            x, y = (w - tw) // 2, h - 60
            cv2.rectangle(disp, (x - 12, y - th - 10), (x + tw + 12, y + 12), (20, 20, 20), -1)
            cv2.putText(disp, txt, (x, y), FONT, .7, (255, 255, 255), 2, cv2.LINE_AA)
        return disp

    # ------------------------------------------------------------ keyboard
    def on_key(self, k):
        if k in (ord("q"), 27):
            return False
        if k == 26:                                   # Ctrl+Z
            self.activate("undo")
            return True
        if k == 25:                                   # Ctrl+Y
            self.activate("redo")
            return True
        raw = chr(k) if 0 <= k < 256 else ""
        if raw and raw in "xyzXYZ":
            self.rotate3d("xyz".index(raw.lower()), 15 if raw.islower() else -15)
            return True
        c = raw.lower()
        if c == "u":
            self.activate("undo")
        elif c == "r":
            self.activate("redo")
        elif c == "c":
            self.activate("clear")
        elif c == "s":
            self.save()
        elif c == "f":
            self.cycle_fill()
        elif c == "w":
            if self.fill_mode == 3:
                self.fill_mode = STATE["prev_fill"]
            else:
                STATE["prev_fill"], self.fill_mode = self.fill_mode, 3
            self.say("Mode: " + FILL_NAMES[self.fill_mode])
        elif c == "a":
            self.next_style()
        elif c == "v":
            self.activate("rotr")
        elif c == "k":
            self.set_cat(self.cat_i + 1)
        elif c == "p":
            self.paper = not self.paper
            self.say("Paper mode " + ("on" if self.paper else "off"))
        elif c == "[":
            self.change_size(-1)
        elif c == "]":
            self.change_size(+1)
        elif c == ",":
            self.step_shape(-1)
        elif c == ".":
            self.step_shape(+1)
        elif c == ";":
            self.step_shape(-25)
        elif c == "'":
            self.step_shape(+25)
        elif c == "t":
            self.next_tool()
        elif c == "l":
            self.skeleton = not self.skeleton
        elif c == "m":
            self.activate("mode")
        elif c == "g":
            self.palette = not self.palette
        elif c.isdigit():
            self.set_color(9 if c == "0" else int(c) - 1)
        return True


# =====================================================================================
#  CAMERA + MEDIAPIPE  (new Tasks API - works with mediapipe 0.10.x and 1.x)
# =====================================================================================
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
             "hand_landmarker/float16/1/hand_landmarker.task")
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")


def ensure_model():
    if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 1_000_000:
        return
    import urllib.request
    print("Downloading hand tracking model (one time, ~8 MB)...")
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    except Exception as e:
        sys.exit("Could not download the model (%s).\nDownload it manually from:\n  %s\n"
                 "and save it next to AirDraw.py as hand_landmarker.task" % (e, MODEL_URL))


def analyse(lms, w, h):
    pts = [(p.x * w, p.y * h) for p in lms]
    d = lambda a, b: math.hypot(a[0] - b[0], a[1] - b[1])
    wrist = pts[0]
    ext = lambda tip, pip: d(wrist, pts[tip]) > d(wrist, pts[pip]) * 1.08
    palm = d(pts[0], pts[9]) + 1e-6
    return dict(tip=pts[8], idx=ext(8, 6), mid=ext(12, 10),
                pinch=d(pts[4], pts[8]) < .32 * palm, wrist_x=pts[0][0], pts=pts)


def draw_skeleton(img, pts, connections):
    for a, b in connections:
        cv2.line(img, (int(pts[a][0]), int(pts[a][1])), (int(pts[b][0]), int(pts[b][1])),
                 (255, 255, 255), 2, cv2.LINE_AA)
    for x, y in pts:
        cv2.circle(img, (int(x), int(y)), 4, (0, 200, 0), -1, cv2.LINE_AA)


def main():
    try:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision
    except ImportError:
        sys.exit("mediapipe is not installed:  pip install mediapipe opencv-python numpy")

    ensure_model()
    landmarker = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO, num_hands=2,
        min_hand_detection_confidence=.6, min_hand_presence_confidence=.6,
        min_tracking_confidence=.6))
    connections = [(c.start, c.end) for c in vision.HandLandmarksConnections.HAND_CONNECTIONS]

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) if os.name == "nt" else cv2.VideoCapture(0)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not cap.isOpened():
        sys.exit("Could not open the webcam.")
    ok, frame = cap.read()
    if not ok:
        sys.exit("Could not read from the webcam.")
    h, w = frame.shape[:2]

    app = App(w, h)
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    print(__doc__)
    t0 = time.time()
    last_ts = -1

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)  # mirror view -> labels match the user's real hands
        ts = int((time.time() - t0) * 1000)
        ts = max(ts, last_ts + 1)
        last_ts = ts
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        res = landmarker.detect_for_video(mp_img, ts)

        found = []
        for lms, hd in zip(res.hand_landmarks, res.handedness):
            info = analyse(lms, w, h)
            info["label"] = hd[0].category_name
            found.append(info)
        hands = {}
        if len(found) == 2:                       # two hands: decide by screen position
            found.sort(key=lambda i: i["wrist_x"])
            hands = {"Left": found[0], "Right": found[1]}
        elif len(found) == 1:
            hands = {found[0]["label"]: found[0]}

        app.update(hands)
        disp = app.render_base(frame)
        if app.skeleton:
            for info in hands.values():
                draw_skeleton(disp, info["pts"], connections)
        app.render_overlay(disp)
        cv2.imshow(WIN, disp)
        if not app.on_key(cv2.waitKey(1) & 0xFF):
            break
        if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    landmarker.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
    