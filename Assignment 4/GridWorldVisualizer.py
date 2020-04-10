"""
Course:         CS 7641 Assignment 4, Spring 2020
Date:           March 31st, 2020
Author:         Maziar Mardan
GT Username:    mmardan3
Comment:        This file is part of a library called gridworld-visualizer, with modifications by me for showing the
policy and trajectory, and can be found here:
https://github.com/mvcisback/gridworld-visualizer
"""

from itertools import product
import svgwrite


#https://github.com/mvcisback/gridworld-visualizer
BOARD_SIZE = ("200", "200")
CSS_STYLES = """
    .background { fill: white; }
    .line { stroke: firebrick; stroke-width: .1mm; }
    .lava { fill: #ff8b8b; }
    .dry { fill: #f4a460; }
    .water { fill: #FF0000; }
    .recharge {fill: #ffff00; }
    .normal {fill: white; }
    .goal {fiLL: #00FF1F; } 
    rect {
       stroke: black;
       stroke-width: 1;
    }
    .marker {
      fill: black;
      stroke-width: 2;
      stroke: grey;
    }
    .agent {
      fill: black;
      stroke-width: 2;
      stroke: grey;
      animation: blinker 4s linear infinite;
      animation: move 5s ease forwards;
    }
    @keyframes blinker {
       50% {
         opacity: 0.5;
       }
    }
"""


def draw_board(n=3, tile2classes=None, policyList=[]):
    dwg = svgwrite.Drawing(size=(f"{n+0.05}cm", f"{n+0.05}cm"))
    dwg.add(dwg.rect(size=('100%', '100%'), class_='background'))

    def group(classname):
        return dwg.add(dwg.g(class_=classname))

    # draw squares
    for x, y in product(range(n), range(n)):
        kwargs = {
            'insert': (f"{x+0.1}cm", f"{y+0.1}cm"),
            'size': (f"0.9cm", f"0.9cm"),
        }
        if tile2classes is not None and tile2classes(y, x):
            kwargs["class_"] = tile2classes(y, x)

        dwg.add(dwg.rect(**kwargs))

    return dwg


N = (0, -1)
S = (0, 1)
W = (-1, 0)
E = (1, 0)


def gen_offsets(actions):
    dx, dy = 0, 0
    for ax, ay in actions:
        dx += ax
        dy += ay
        yield dx, dy


def move_keyframe(dx, dy, ratio):
    return f"""{ratio*100}% {{
    transform: translate({dx}cm, {dy}cm);
}}"""


def gridworld(n=10, actions=None, tile2classes=None, start=(0,0), extra_css="", policyList=[]):
    policy = []
    dwg = draw_board(n=n, tile2classes=tile2classes, policyList=policyList)

    for i in range(n):
        ind0 = i * n
        ind1 = (i + 1) * n
        policy.append(policyList[ind0:ind1])

    css_styles = CSS_STYLES
    if actions is not None:
        # Add agent.
        x, y = start[0], start[1]  # start position.
        cx, cy = y + 0.55, x + 0.55
        dwg.add(svgwrite.shapes.Circle(
            r="0.2cm",
            center=(f"{cx}cm", f"{cy}cm"),
            class_="agent",
        ))

        dwg.add(svgwrite.shapes.Circle(
            r="0.1cm",
            center=(f"{cx}cm", f"{cy}cm"),
            class_="agent0",
        ))

        offsets = gen_offsets(actions)
        keyframes = [move_keyframe(x, y, (i + 1) / len(actions)) for i, (x, y)
                     in enumerate(offsets)]
        move_css = "\n@keyframes move {\n" + '\n'.join(keyframes) + "\n}"
        css_styles += move_css


    if actions is not None:
        offsets = gen_offsets(actions)
        for i, (x, y) in enumerate(offsets):
            dwg.add(svgwrite.shapes.Circle(
                r="0.1cm",
                center=(f"{cx+x}cm", f"{cy+y}cm"),
                class_="agent"+str(i),
            ))

    for x, y in product(range(n), range(n)):
        if policy[x][y] == 0 and tile2classes(x, y) != "water" and tile2classes(x, y) != "goal":
            dwg.add(svgwrite.shapes.Line(start=(f"{y+0.75*1.1}cm", f"{x+0.25*1.1}cm"), end=(f"{y+0.25*1.1}cm", f"{x+0.5*1.1}cm"),
                                         class_="marker"))
            dwg.add(svgwrite.shapes.Line(start=(f"{y+0.75*1.1}cm", f"{x+0.75*1.1}cm"), end=(f"{y+0.25*1.1}cm", f"{x+0.5*1.1}cm"),
                                         class_="marker"))
        elif policy[x][y] == 1 and tile2classes(x, y) != "water" and tile2classes(x, y) != "goal":
            dwg.add(svgwrite.shapes.Line(start=(f"{y+0.25*1.1}cm", f"{x+0.75*1.1}cm"), end=(f"{y+0.5*1.1}cm", f"{x+0.25*1.1}cm"),
                                         class_="marker"))
            dwg.add(svgwrite.shapes.Line(start=(f"{y+0.75*1.1}cm", f"{x+0.75*1.1}cm"), end=(f"{y+0.5*1.1}cm", f"{x+0.25*1.1}cm"),
                                         class_="marker"))
        elif policy[x][y] == 2 and tile2classes(x, y) != "water" and tile2classes(x, y) != "goal":
            dwg.add(svgwrite.shapes.Line(start=(f"{y+0.25*1.1}cm", f"{x+0.25*1.1}cm"), end=(f"{y+0.75*1.1}cm", f"{x+0.5*1.1}cm"),
                                         class_="marker"))
            dwg.add(svgwrite.shapes.Line(start=(f"{y+0.25*1.1}cm", f"{x+0.75*1.1}cm"), end=(f"{y+0.75*1.1}cm", f"{x+0.5*1.1}cm"),
                                         class_="marker"))
        elif policy[x][y] == 3 and tile2classes(x, y) != "water" and tile2classes(x, y) != "goal":
            dwg.add(svgwrite.shapes.Line(start=(f"{y+0.25*1.1}cm", f"{x+0.25*1.1}cm"), end=(f"{y+0.5*1.1}cm", f"{x+0.75*1.1}cm"),
                                         class_="marker"))
            dwg.add(svgwrite.shapes.Line(start=(f"{y+0.75*1.1}cm", f"{x+0.25*1.1}cm"), end=(f"{y+0.5*1.1}cm", f"{x+0.75*1.1}cm"),
                                         class_="marker"))

    dwg.defs.add(dwg.style(css_styles + extra_css))
    return dwg


if __name__ == '__main__':
    def tile2classes(x, y):
        if (3 <= x <= 4) and (2 <= y <= 5):
            return "water"
        elif (x in (0, 7)) and (y in (0, 7)):
            return "recharge"
        elif (2 <= x <= 5) and y in (0, 7):
            return "dry"
        elif x in (1, 6) and (y in (4, 5) or y <= 1):
            return "lava"
        elif (x in (0, 7)) and (y in (1, 4, 5)):
            return "lava"

        return "normal"


    actions = [E, N, N, N, N, W, W, W]
    dwg = gridworld(n=8, tile2classes=tile2classes, actions=actions)
    dwg.saveas("example.svg", pretty=True)