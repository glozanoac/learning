"""
Watch the agent play live, switch the hidden phase, and see it adapt.

    python watch.py                               # trained agent, fortress map, smart aliens
    python watch.py --weights out/a2c_fortress_smart_u500.npz   # an early snapshot
    python watch.py --no-learn                    # frozen policy

Buttons Fortress / Open / Pillars change the map, and Rain / Sniper / Trap / Smart
change the hidden phase, on the fly.

Manual: press the "Manual" button (or the M key, or start with --manual)
to fly the ship yourself:
    arrow keys or W A S D = move,  space = light shot,  X = grenade.
If the ship does not react, click once on the game picture so the window
has the keyboard focus; the panel shows the last key it received.
Press it again to give control back to the agent. While you play, the agent
keeps training in the background (if "learn online" is on).
With "learn online" on, 16 invisible copies of the game keep training the
same A2C agent in the current phase between frames. Switch to Trap and you
will see the agent walk into the trap at first, then stop going for the bait
after roughly 10-30 seconds (the "background" stats on the right show it).
"""
import argparse
import tkinter as tk
from collections import deque

import numpy as np
from PIL import Image, ImageTk

from invaders_env import Invaders, render
from train_a2c import A2C, Tracker

N_ENVS, N_STEPS = 16, 8
KEY_ACTION = {"Left": 0, "Right": 1, "Up": 2, "Down": 3,
              "a": 0, "d": 1, "w": 2, "s": 3, "A": 0, "D": 1, "W": 2, "S": 3}


class Viewer:
    def __init__(self, root, agent, phase, learn, updates_per_frame, map_name="fortress"):
        self.root, self.agent, self.phase, self.map = root, agent, phase, map_name
        self.upf = updates_per_frame
        root.title("Mini Invaders - A2C agent")
        self.env = Invaders(None, phase, map_name)
        self.obs = self.env.features()
        self.bg = [Invaders(1000 + i, phase, map_name) for i in range(N_ENVS)]
        self.bg_obs = np.stack([e.features() for e in self.bg])
        self.tracker = Tracker(self.bg)
        self.window = deque(maxlen=150)          # per-update deltas of the stats
        self.n_updates = 0
        self.manual = False                      # True: the human flies the ship
        self.held = set()                        # keys currently held down
        self.taps = []                           # keys pressed since the last frame
        root.bind_all("<KeyPress>", self.on_key_down)
        root.bind_all("<KeyRelease>", self.on_key_up)

        # the game picture scales with the window (drag the window edges)
        base = render(self.env, "", cell=36)
        self.canvas = tk.Canvas(root, width=base.width, height=base.height, bg="#0c0e20",
                                highlightthickness=0, takefocus=1)
        self.canvas.grid(row=0, column=0, rowspan=2, sticky="nsew")
        self.canvas.bind("<Button-1>", lambda _e: self.root.focus_force())
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)
        root.rowconfigure(1, weight=1)
        root.resizable(True, True)
        root.minsize(500, 400)
        self.last_key = "-"
        side = tk.Frame(root, padx=10)
        side.grid(row=0, column=1, sticky="n")
        self.b_manual = tk.Button(side, text="Manual: OFF", width=12, bg="#d5dbdb",
                                  font=("TkDefaultFont", 11, "bold"), takefocus=0,
                                  command=self.toggle_manual)
        self.b_manual.pack(anchor="w", pady=(0, 2))
        tk.Label(side, text="arrows / WASD = move\nspace = light shot, X = grenade\nM = manual on/off", fg="#555",
                 justify="left").pack(anchor="w", pady=(0, 2))
        self.l_key = tk.Label(side, text="last key: -", fg="#555")
        self.l_key.pack(anchor="w", pady=(0, 8))
        tk.Label(side, text="Map", font=("TkDefaultFont", 11, "bold")).pack(anchor="w")
        self.mbtns = {}
        for m in ("fortress", "open", "pillars"):
            b = tk.Button(side, text=m.capitalize(), width=12, bg="#a9dfbf", takefocus=0,
                          command=lambda m=m: self.set_map(m))
            b.pack(anchor="w", pady=2)
            self.mbtns[m] = b
        tk.Label(side, text="Hidden phase", font=("TkDefaultFont", 11, "bold")).pack(anchor="w", pady=(8, 0))
        self.btns = {}
        for p, col in [("rain", "#5dade2"), ("sniper", "#f5b041"), ("trap", "#ec7063"),
                       ("smart", "#bb8fce")]:
            b = tk.Button(side, text=p.capitalize(), width=12, bg=col, takefocus=0,
                          command=lambda p=p: self.set_phase(p))
            b.pack(anchor="w", pady=2)
            self.btns[p] = b
        self.learn = tk.BooleanVar(value=learn)
        tk.Checkbutton(side, text="learn online", variable=self.learn,
                       takefocus=0).pack(anchor="w", pady=(10, 0))
        tk.Label(side, text="speed").pack(anchor="w", pady=(8, 0))
        self.speed = tk.Scale(side, from_=10, to=200, orient="horizontal", length=160, takefocus=0)
        self.speed.set(60)
        self.speed.pack(anchor="w")
        self.stats = tk.Label(side, text="", justify="left", font=("TkFixedFont", 10))
        self.stats.pack(anchor="w", pady=(12, 0))
        self.set_phase(phase)
        self.set_map(map_name)
        self.tick()

    # ------------------------------------------------------------ manual mode --
    def toggle_manual(self):
        self.manual = not self.manual
        self.held.clear()
        self.taps.clear()
        if self.manual:
            self._agent_speed = self.speed.get()
            self.speed.set(max(self._agent_speed, 110))     # slower, so a human can react
            self.b_manual.config(text="Manual: ON", bg="#f7dc6f", relief="sunken")
        else:
            self.speed.set(getattr(self, "_agent_speed", 60))
            self.b_manual.config(text="Manual: OFF", bg="#d5dbdb", relief="raised")
        self.root.focus_force()                            # make sure the keys reach the game

    def on_key_down(self, ev):
        self.last_key = ev.keysym
        self.l_key.config(text=f"last key: {ev.keysym}")
        if ev.keysym in ("m", "M"):
            self.toggle_manual()
            return "break"
        if ev.keysym in KEY_ACTION or ev.keysym in ("space", "x", "X"):
            self.held.add(ev.keysym)
            self.taps.append(ev.keysym)
            return "break"                                 # don't let space press a button

    def on_key_up(self, ev):
        self.held.discard(ev.keysym)

    def human_action(self):
        """X throws a grenade, space a light shot (if one is available);
        otherwise the arrow pressed last (tapped since the previous frame, or
        still held) moves the ship."""
        env = self.env
        pressed = self.taps + [k for k in self.held if k not in self.taps]
        self.taps = []
        if ("x" in pressed or "X" in pressed) and (env.can_fire_grenade() or env.ammo_g == 0):
            return 6
        if "space" in pressed and (env.can_fire_light() or env.ammo_l == 0):
            return 5
        arrows = [k for k in pressed if k in KEY_ACTION]
        return KEY_ACTION[arrows[-1]] if arrows else 4

    def set_map(self, m):
        self.map = m
        self.env.set_map(m)
        for e in self.bg:
            e.set_map(m)
        self.window.clear()
        for q, b in self.mbtns.items():
            b.config(relief="sunken" if q == m else "raised")

    def set_phase(self, p):
        self.phase = p
        self.env.phase = p
        for e in self.bg:
            e.phase = p
        self.window.clear()
        for q, b in self.btns.items():
            b.config(relief="sunken" if q == p else "raised")

    def learn_step(self):
        ag = self.agent
        before = self.tracker.totals()
        X, A, R, D = [], [], [], []
        for _ in range(N_STEPS):
            a = ag.act(self.bg_obs)
            nxt, r, d = [], np.zeros(N_ENVS), np.zeros(N_ENVS)
            for i, e in enumerate(self.bg):
                o, r[i], d[i] = e.step(int(a[i]))
                if d[i]:
                    self.tracker.bank(e)
                    o = e.reset()
                nxt.append(o)
            X.append(self.bg_obs); A.append(a); R.append(r); D.append(d)
            self.bg_obs = np.stack(nxt)
        G = np.zeros((N_STEPS, N_ENVS))
        g = ag.value(self.bg_obs)
        for t in reversed(range(N_STEPS)):
            g = R[t] + ag.gamma * g * (1 - D[t])
            G[t] = g
        ag.update(np.concatenate(X), np.concatenate(A), G.reshape(-1))
        self.n_updates += 1
        after = self.tracker.totals()
        self.window.append([after[k] - before[k] for k in ("ret", "collected", "dropped", "hits")])

    def tick(self):
        if self.learn.get():
            for _ in range(self.upf):
                self.learn_step()
        a = self.human_action() if self.manual else int(self.agent.act(self.obs)[0])
        self.obs, _, done = self.env.step(a)
        who = "YOU" if self.manual else "AGENT"
        title = f"{who} | map: {self.map.upper()}  phase: {self.phase.upper()}"
        img = render(self.env, title, cell=36)
        cw, ch = max(self.canvas.winfo_width(), 50), max(self.canvas.winfo_height(), 50)
        scale = min(cw / img.width, ch / img.height)            # keep the aspect ratio
        img = img.resize((max(1, int(img.width * scale)), max(1, int(img.height * scale))),
                         Image.LANCZOS if scale < 1 else Image.NEAREST)
        self.photo = ImageTk.PhotoImage(img)
        self.canvas.delete("all")
        self.canvas.create_image(cw // 2, ch // 2, image=self.photo, anchor="center")
        if done:
            self.env.reset()
            self.obs = self.env.features()
        if self.window:
            b = np.array(self.window)
            steps = len(b) * N_ENVS * N_STEPS
            rate = b[:, 1].sum() / max(b[:, 2].sum(), 1)
            txt = (f"background play (last {steps} steps)\n"
                   f"  reward/1000 steps {1000 * b[:, 0].sum() / steps:6.1f}\n"
                   f"  bonuses taken     {100 * rate:5.0f} %\n"
                   f"  hits/1000 steps   {1000 * b[:, 3].sum() / steps:6.2f}\n")
        else:
            txt = "background play: -\n\n\n\n"
        txt += f"\nA2C updates: {self.n_updates}"
        self.stats.config(text=txt)
        self.root.after(int(self.speed.get()), self.tick)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="out/a2c_final.npz")
    ap.add_argument("--phase", default="smart", choices=["rain", "sniper", "trap", "smart"])
    ap.add_argument("--map", default="fortress", choices=["fortress", "open", "pillars"])
    ap.add_argument("--no-learn", action="store_true")
    ap.add_argument("--updates-per-frame", type=int, default=4)
    ap.add_argument("--manual", action="store_true", help="start with you in control")
    args = ap.parse_args()
    agent = A2C(Invaders.N_FEAT, np.random.default_rng())
    agent.load(args.weights)
    root = tk.Tk()
    v = Viewer(root, agent, args.phase, not args.no_learn, args.updates_per_frame, args.map)
    if args.manual:
        root.after(200, v.toggle_manual)
    root.after(300, root.focus_force)
    root.mainloop()


if __name__ == "__main__":
    main()
