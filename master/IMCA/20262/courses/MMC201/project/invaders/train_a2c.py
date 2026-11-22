"""
Actor-critic agents for Mini Invaders, written to follow
    S. Zhao, "Mathematical Foundations of Reinforcement Learning" (Springer, 2025)
Tags of the form [Zhao 10.2, Alg. 10.2] or [Zhao (8.13)] point to sections,
algorithms and equations of the book. READING_PLAN.md maps every part of this
file to the book and to the papers behind the practical choices.

  algorithms
    --algo a2c   Advantage actor-critic        [Zhao 10.2, Alg. 10.2, (10.7)-(10.8)]
                 actor  : theta += a_th * A_t * grad ln pi(a_t|s_t)
                 critic : v(s, w) learned by TD with function approx. [Zhao 8.2, (8.13)]
    --algo qac   Q actor-critic (the simplest) [Zhao 10.1, Alg. 10.1, (10.2)]
                 actor  : theta += a_th * q(s_t, a_t, w) * grad ln pi(a_t|s_t)
                 critic : q(s, a, w) learned by Sarsa with function approx. [Zhao 8.3.1, (8.35)]

  networks       actor  pi(a|s, theta): MLP 311 -> 64 (tanh) -> 7, softmax  [Zhao 9.1, (9.12)]
                 critic v(s, w) or q(s, ., w): MLP 311 -> 64 (tanh) -> 1 or 7  [Zhao 8.2.3]

  targets        n-step return  G_t^(n) = r_{t+1} + ... + g^{n-1} r_{t+n} + g^n v(s_{t+n})
                                                                        [Zhao 7.3]
                 --nstep 1 gives the TD error of Alg. 10.2:
                 delta_t = r_{t+1} + g v(s_{t+1}) - v(s_t)

  beyond the book (see READING_PLAN.md for the papers)
                 * N parallel environments -> mini-batch stochastic gradient  [Zhao 6.4.4]
                 * entropy bonus beta * H(pi(.|s))          (Williams & Peng 1991; Mnih et al. 2016)
                 * Adam with constant step sizes + gradient clipping (Kingma & Ba 2015)
                   (constant steps deliberately break the Robbins-Monro conditions of
                   [Zhao Thm 6.1 / Thm 6.4] so that the agent keeps tracking a changing world)

  Algorithm 10.2 exactly as printed (one sample per update, TD error, no entropy):
      python train_a2c.py --nstep 1 --envs 1 --entropy 0
  (the book uses plain SGD steps; this code always uses Adam)

Schedule: a sequence of stages (map, hidden phase); currently a single stage,
fortress map with smart aliens. Snapshots are saved at CHECKPOINTS updates so
the learning progress can be replayed (make_gifs.py, watch.py --weights ...).

Run:  python train_a2c.py                         (A2C, defaults used in the report)
      python train_a2c.py --algo qac              (QAC, Algorithm 10.1)
      python train_a2c.py --resume out/a2c_final.npz --seed 1   (continue training)
Writes: out/<algo>_<k>_<map>_<phase>.npz (end of each stage), out/<algo>_final.npz,
        out/<algo>_<map>_<phase>_u<N>.npz (snapshots during training),
        out/curves.png, out/eval.png, out/eval.txt
"""
import argparse
import os
import time

import numpy as np

from invaders_env import ACTIONS, Invaders, heuristic


# ----------------------------------------------------------------------------
# Function approximators  [Zhao 8.1 "from table to function", 8.2.3, 9.1]
# ----------------------------------------------------------------------------
class MLP:
    """Two-layer network f(x) = W2^T tanh(W1^T x + b1) + b2 with manual backprop.

    [Zhao 8.2.3] lists two choices of approximator, linear phi(s)^T w or a
    neural network; this is the neural-network option. backward() is the chain
    rule (report, Prop. "back-propagation"); tanh'(x) = 1 - tanh(x)^2.
    """
    def __init__(self, n_in, n_hid, n_out, rng, out_scale=1.0):
        self.p = {
            "W1": rng.normal(0, 1 / np.sqrt(n_in), (n_in, n_hid)),
            "b1": np.zeros(n_hid),
            "W2": rng.normal(0, out_scale / np.sqrt(n_hid), (n_hid, n_out)),
            "b2": np.zeros(n_out),
        }

    def forward(self, X):
        h = np.tanh(X @ self.p["W1"] + self.p["b1"])
        return h @ self.p["W2"] + self.p["b2"], h

    def backward(self, X, h, dout):
        g = {"W2": h.T @ dout, "b2": dout.sum(0)}
        dh = (dout @ self.p["W2"].T) * (1 - h ** 2)
        g["W1"] = X.T @ dh
        g["b1"] = dh.sum(0)
        return g


class Adam:
    """Adam on a dict of parameters (constant step size, gradient clipping).

    The book optimises with (mini-batch) stochastic gradient steps
    w <- w - a_k grad f(w, x_k)  [Zhao (6.13), 6.4.4]. Adam replaces the scalar
    a_k by a per-coordinate step built from running averages of the gradient
    and of its square (Kingma & Ba 2015). The step size stays CONSTANT, so the
    Robbins-Monro conditions sum a_k = inf, sum a_k^2 < inf of
    [Zhao Thm 6.1, Thm 6.4] do not hold: we trade asymptotic convergence for
    the ability to track a non-stationary target (report, Prop. "tracking").
    """

    def __init__(self, params, lr, b1=0.9, b2=0.999, eps=1e-8, clip=1.0):
        self.p, self.lr, self.b1, self.b2, self.eps, self.clip = params, lr, b1, b2, eps, clip
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, g):
        norm = np.sqrt(sum((x ** 2).sum() for x in g.values()))
        scale = min(1.0, self.clip / (norm + 1e-12))           # clip by global norm (Pascanu et al. 2013)
        self.t += 1
        for k in self.p:
            gk = g[k] * scale
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * gk
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * gk ** 2
            mh = self.m[k] / (1 - self.b1 ** self.t)           # bias correction (report, Lemma)
            vh = self.v[k] / (1 - self.b2 ** self.t)
            self.p[k] -= self.lr * mh / (np.sqrt(vh) + self.eps)


# ----------------------------------------------------------------------------
# Agents  [Zhao Ch. 10]
# ----------------------------------------------------------------------------
class A2C:
    """Advantage actor-critic  [Zhao 10.2, Alg. 10.2].

    gamma : discount rate of the return  [Zhao 1.6, (1.3)]
    beta  : weight of the entropy bonus (beyond the book; 0 = Alg. 10.2 as printed)
    """

    def __init__(self, n_in, rng, hid=64, lr_actor=7e-4, lr_critic=1e-3, gamma=0.99,
                 beta=0.02):
        self.actor = MLP(n_in, hid, len(ACTIONS), rng, out_scale=0.01)
        self.critic = MLP(n_in, hid, 1, rng)
        self.oa = Adam(self.actor.p, lr_actor)
        self.oc = Adam(self.critic.p, lr_critic)
        self.gamma, self.beta, self.rng = gamma, beta, rng

    def pi(self, X):
        """Softmax policy pi(a|s, theta) = exp(h(s,a,theta)) / sum_a' exp(h(s,a',theta))
        [Zhao (9.12)]; h = actor logits. Strictly positive, hence exploratory."""
        z, _ = self.actor.forward(X)
        z = z - z.max(1, keepdims=True)
        p = np.exp(z)
        return p / p.sum(1, keepdims=True)

    def act(self, X, greedy=False):
        """Sample a ~ pi(.|s, theta) (on-policy: the same policy that is learned)."""
        p = self.pi(np.atleast_2d(X))
        if greedy:
            return p.argmax(1)
        u = self.rng.random((p.shape[0], 1))
        return (p.cumsum(1) > u).argmax(1)

    def value(self, X):
        """Critic v(s, w) ~ v_pi(s)  [Zhao 8.2]."""
        return self.critic.forward(X)[0][:, 0]

    def bootstrap(self, X, A_next=None):
        """Tail of the n-step target: v(s_{t+n}, w)  [Zhao 7.3]."""
        return self.value(X)

    def update(self, X, A, G):
        """One A2C update from a batch of N samples (s_i, a_i, G_i).

        advantage : A_i = G_i - v(s_i, w)  estimates q_pi - v_pi = delta_pi(s, a)
                    [Zhao 10.2.2, (10.7)]; with n = 1 it is the TD error of Alg. 10.2.
                    Subtracting v is the baseline of [Zhao 10.2.1, (10.3)]: it leaves
                    the gradient unbiased and reduces its variance ([Zhao (10.5)]
                    gives the optimal baseline; v_pi is the practical one).
        critic    : semi-gradient TD step w += a_w * A_i * grad_w v(s_i, w)
                    [Zhao (8.13), Alg. 10.2 "critic"]  = gradient of 1/2 (G - v)^2 with
                    the target G held fixed  (objective [Zhao (8.3)]).
        actor     : theta += a_th * A_i * grad ln pi(a_i|s_i, theta)  [Zhao (10.8)],
                    the stochastic version of the policy gradient theorem
                    [Zhao Thm 9.1, (9.9); Thm 9.2 for J = v-bar_pi^0], plus beta * grad H.
        grad ln pi w.r.t. the logits is e_a - pi (report, Lemma "softmax score").
        """
        N = len(A)
        # critic  [Zhao (8.13)]: minimise 1/2 (G - v)^2 with G fixed (semi-gradient)
        v, hc = self.critic.forward(X)
        v = v[:, 0]
        adv = G - v                                               # advantage / TD error
        self.oc.step(self.critic.backward(X, hc, (-(adv) / N)[:, None]))
        # actor  [Zhao (10.8)]: maximise A * ln pi(a|s) + beta * entropy
        z, ha = self.actor.forward(X)
        z = z - z.max(1, keepdims=True)
        p = np.exp(z)
        p /= p.sum(1, keepdims=True)
        logp = np.log(p + 1e-12)
        ent = -(p * logp).sum(1)
        onehot = np.eye(len(ACTIONS))[A]
        d_pg = -(adv[:, None]) * (onehot - p)                    # d(-A ln pi)/dz, [Zhao (9.11)]
        d_ent = self.beta * p * (logp + ent[:, None])             # d(-beta H)/dz
        self.oa.step(self.actor.backward(X, ha, (d_pg + d_ent) / N))
        return float(ent.mean())

    def save(self, f):
        np.savez(f, **{"a_" + k: v for k, v in self.actor.p.items()},
                 **{"c_" + k: v for k, v in self.critic.p.items()})

    def load(self, f):
        d = np.load(f)
        for k in self.actor.p:
            self.actor.p[k][...] = d["a_" + k]
        for k in self.critic.p:
            self.critic.p[k][...] = d["c_" + k]


class QAC(A2C):
    """The simplest actor-critic, Q actor-critic  [Zhao 10.1, Alg. 10.1].

    critic : q(s, a, w) for all actions at once (MLP 311 -> 64 -> 7), learned by
             Sarsa with function approximation  [Zhao 8.3.1, (8.35)]; with n > 1 the
             target is the n-step Sarsa return  [Zhao 7.3].
    actor  : theta += a_th * q(s_t, a_t, w_t) * grad ln pi(a_t|s_t)  [Zhao (10.2)]
             -- NO baseline, so the gradient estimate has higher variance than A2C
             (compare [Zhao 10.2.1]).
    Sarsa needs the NEXT action a_{t+1} ~ pi(.|s_{t+1}, theta_t) sampled with the
    parameters before the update; train() keeps it and actually plays it.
    """

    def __init__(self, n_in, rng, hid=64, lr_actor=7e-4, lr_critic=1e-3, gamma=0.99,
                 beta=0.02):
        super().__init__(n_in, rng, hid, lr_actor, lr_critic, gamma, beta)
        self.critic = MLP(n_in, hid, len(ACTIONS), rng)          # q(s, ., w)
        self.oc = Adam(self.critic.p, lr_critic)

    def qvalues(self, X):
        return self.critic.forward(X)[0]

    def value(self, X):
        """v(s) = sum_a pi(a|s) q(s, a)  [Zhao (2.13)] (only used for reporting)."""
        return (self.pi(X) * self.qvalues(X)).sum(1)

    def bootstrap(self, X, A_next):
        """Tail of the (n-step) Sarsa target: q(s_{t+n}, a_{t+n}, w)  [Zhao (8.35), 7.3]."""
        q = self.qvalues(X)
        return q[np.arange(len(A_next)), A_next]

    def update(self, X, A, G):
        N = len(A)
        idx = np.arange(N)
        q, hc = self.critic.forward(X)
        q_sa = q[idx, A]                                          # q(s_t, a_t, w_t)
        # actor  [Zhao Alg. 10.1]: weight = q(s_t, a_t, w_t), computed BEFORE the critic step
        z, ha = self.actor.forward(X)
        z = z - z.max(1, keepdims=True)
        p = np.exp(z)
        p /= p.sum(1, keepdims=True)
        logp = np.log(p + 1e-12)
        ent = -(p * logp).sum(1)
        onehot = np.eye(len(ACTIONS))[A]
        d_pg = -(q_sa[:, None]) * (onehot - p)
        d_ent = self.beta * p * (logp + ent[:, None])
        self.oa.step(self.actor.backward(X, ha, (d_pg + d_ent) / N))
        # critic [Zhao (8.35)]: semi-gradient Sarsa on the chosen action only
        dq = np.zeros_like(q)
        dq[idx, A] = -(G - q_sa) / N
        self.oc.step(self.critic.backward(X, hc, dq))
        return float(ent.mean())


AGENTS = {"a2c": A2C, "qac": QAC}


# ----------------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------------
STAT_KEYS = ("ret", "hits", "kills", "own_blocks", "collected", "dropped",
             "got_life", "drop_life", "got_gun", "drop_gun", "got_repair", "drop_repair",
             "got_ammo", "drop_ammo", "got_grenades", "drop_grenades",
             "shots_light", "shots_grenade", "alien_hits", "dry", "empty_steps", "rammed", "damage")


class Tracker:
    """Running totals over a set of envs that survive resets (episodes have no
    fixed length any more, so we measure per step instead of per episode)."""

    def __init__(self, envs):
        self.envs = envs
        self.banked = {k: 0.0 for k in STAT_KEYS}
        self.banked["deaths"] = 0.0

    def bank(self, env):                       # call right before env.reset()
        for k in STAT_KEYS:
            self.banked[k] += env.stats[k]
        self.banked["deaths"] += 1

    def totals(self):
        t = dict(self.banked)
        for e in self.envs:
            for k in STAT_KEYS:
                t[k] += e.stats[k]
        return t


def summarize(d, steps, cover=0, above=0):
    """Turn raw counts over `steps` steps into readable rates."""
    per_k = 1000 / steps
    rate = lambda a, b: d[a] / max(d[b], 1)
    return dict(ret=d["ret"] * per_k, hits=d["hits"] * per_k, kills=d["kills"] * per_k,
                deaths=d["deaths"] * per_k, own=d["own_blocks"] * per_k,
                coll=rate("collected", "dropped"), life=rate("got_life", "drop_life"),
                gun=rate("got_gun", "drop_gun"), repair=rate("got_repair", "drop_repair"),
                ammo=rate("got_ammo", "drop_ammo"), grn=rate("got_grenades", "drop_grenades"),
                shots=d["shots_light"] * per_k, grenades=d["shots_grenade"] * per_k,
                acc=100 * d["alien_hits"] / max(d["shots_light"] + d["shots_grenade"], 1),
                dry=d["dry"] * per_k, empty=100 * d["empty_steps"] / steps,
                rammed=d["rammed"] * per_k,
                cover=100 * cover / steps, above=100 * above / steps)


def evaluate(policy, phase, map_name, steps=4000, seed=10_000):
    """Play `steps` steps (restarting after each game over) and report rates per
    1000 steps. policy: 'greedy', 'cautious' or an A2C agent (stochastic policy)."""
    from invaders_env import BUNKER_ROWS
    env = Invaders(seed, phase, map_name)
    tr = Tracker([env])
    obs = env.features()
    cover = above = shots = peeks = 0
    recent_cover = [False] * 4
    if hasattr(policy, "rng"):
        policy.rng = np.random.default_rng(seed)
    for _ in range(steps):
        c = env.covered()
        cover += c
        above += env.sy < BUNKER_ROWS[0]
        recent_cover = recent_cover[1:] + [c]
        if policy == "greedy":
            a = heuristic(env, True)
        elif policy == "cautious":
            a = heuristic(env, False)
        else:
            a = int(policy.act(obs)[0])
        n_before = len(env.pbullets)
        obs, _, done = env.step(a)
        if a == 5 and (len(env.pbullets) > n_before or env.stats["kills"]):
            shots += 1
            peeks += (not env.covered()) and any(recent_cover)   # stepped out of cover to shoot
        if done:
            tr.bank(env)
            obs = env.reset()
            recent_cover = [False] * 4
    out = summarize(tr.totals(), steps, cover, above)
    out["peek"] = 100 * peeks / max(shots, 1)
    return out


# ----------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------
SCHEDULE = (("fortress", "smart"),)
CHECKPOINTS = (500, 2000, 8000)        # also save snapshots at these updates


def train(schedule=SCHEDULE, updates_per_phase=60000,
          n_envs=16, n_steps=8, seed=0, outdir="out", init=None, algo="a2c", beta=0.02):
    """On-policy actor-critic training loop, the outer loop of [Zhao Alg. 10.1 / 10.2].

    Each update: every one of the n_envs environments plays n_steps steps with the
    current policy, the n-step targets G_t^(n) are formed [Zhao 7.3], and ONE
    gradient step is taken on the average over all n_envs * n_steps samples
    (a mini-batch stochastic gradient, [Zhao 6.4.4]). n_envs = n_steps = 1 and
    beta = 0 reproduce Alg. 10.2 (or Alg. 10.1 for --algo qac) sample by sample.
    """
    os.makedirs(outdir, exist_ok=True)
    rng = np.random.default_rng(seed)
    envs = [Invaders(seed * 1000 + i) for i in range(n_envs)]
    obs = np.stack([e.features() for e in envs])
    agent = AGENTS[algo](obs.shape[1], rng, beta=beta)
    if init:                                   # continue training a saved agent
        agent.load(init)
    # QAC needs a_{t+1} drawn before the update [Alg. 10.1]; A2C does not (keeps its
    # random stream identical to the run reported in the document)
    a_next = agent.act(obs) if algo == "qac" else None
    tracker = Tracker(envs)
    log = []               # per update: (env_step, stage, reward, hits, collected, dropped, deaths)
    evals = []             # per phase: before / after
    env_steps, t0 = 0, time.time()

    for k, (mp, phase) in enumerate(schedule):
        for e in envs:
            e.phase = phase
            e.set_map(mp)
        obs = np.stack([e.features() for e in envs])
        before = evaluate(agent, phase, mp)
        a_next = agent.act(obs) if algo == "qac" else None
        for u in range(updates_per_phase):
            t_before = tracker.totals()
            X, A, R, D = [], [], [], []
            for j in range(n_steps):
                # a_t ~ pi(.|s_t, theta_t); for QAC the first one is the a_{t+1} already
                # sampled for the previous Sarsa target, so the target is truly on-policy
                a = a_next if (algo == "qac" and j == 0) else agent.act(obs)
                nxt, r, d = [], np.zeros(n_envs), np.zeros(n_envs)
                for i, e in enumerate(envs):
                    o, r[i], d[i] = e.step(int(a[i]))
                    if d[i]:
                        tracker.bank(e)
                        o = e.reset()
                    nxt.append(o)
                X.append(obs); A.append(a); R.append(r); D.append(d)
                obs = np.stack(nxt)
                env_steps += n_envs
            # n-step returns [Zhao 7.3], computed backwards:
            #   G_t = r_{t+1} + gamma * G_{t+1},  G_{t+n} = v(s_{t+n}) (A2C) or q(s_{t+n}, a_{t+n}) (QAC)
            # a terminal state (no lives left) has value 0, so the bootstrap is cut there.
            a_next = agent.act(obs) if algo == "qac" else None
            G = np.zeros((n_steps, n_envs))
            g = agent.bootstrap(obs, a_next)
            for t in reversed(range(n_steps)):
                g = R[t] + agent.gamma * g * (1 - D[t])
                G[t] = g
            ent = agent.update(np.concatenate(X), np.concatenate(A), G.reshape(-1))
            t_after = tracker.totals()
            log.append((env_steps, k) + tuple(t_after[q] - t_before[q] for q in
                                              ("ret", "hits", "collected", "dropped", "deaths")))
            if init is None and (u + 1) in CHECKPOINTS:
                agent.save(os.path.join(outdir, f"{algo}_{mp}_{phase}_u{u + 1}.npz"))
            if (u + 1) % 1000 == 0:
                L = np.array(log[-1000:])
                n = n_envs * n_steps * len(L)
                print(f"[{mp:8s}/{phase:6s}] update {u + 1:5d}  steps {env_steps:8d}  "
                      f"reward/1000 steps {1000 * L[:, 2].sum() / n:6.1f}  "
                      f"deaths/1000 steps {1000 * L[:, 6].sum() / n:5.2f}  entropy {ent:.2f}  "
                      f"{time.time() - t0:5.0f}s", flush=True)
        after = evaluate(agent, phase, mp)
        evals.append((f"{mp}/{phase}", before, after,
                      evaluate("greedy", phase, mp), evaluate("cautious", phase, mp)))
        agent.save(os.path.join(outdir, f"{algo}_{k}_{mp}_{phase}.npz"))
    agent.save(os.path.join(outdir, f"{algo}_final.npz"))
    np.save(os.path.join(outdir, f"log_{algo}.npy" if algo != "a2c" else "log.npy"), np.array(log))
    return agent, np.array(log), evals


# ----------------------------------------------------------------------------
# Plots and table
# ----------------------------------------------------------------------------
def report(log, evals, schedule, outdir="out", steps_per_update=16 * 8, tag="A2C"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"rain": "#d6eaf8", "sniper": "#fdebd0", "trap": "#fadbd8", "smart": "#e8daef"}
    step, stage = log[:, 0], log[:, 1]
    n = max(1, min(300, len(log) // 5))            # updates per moving window

    def ma(x):
        c = np.cumsum(np.insert(x, 0, 0))
        return np.concatenate([np.full(n - 1, np.nan), (c[n:] - c[:-n]) / n])

    per_k = 1000 / steps_per_update
    series = [(ma(log[:, 2]) * per_k, "reward per 1000 steps"),
              (ma(log[:, 4]) / np.maximum(ma(log[:, 5]), 1e-9), "bonuses collected / dropped"),
              (ma(log[:, 3]) * per_k, "hits per 1000 steps")]
    fig, axs = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    for ax, (y, lab) in zip(axs, series):
        for k, (mp, p) in enumerate(schedule):
            idx = np.where(stage == k)[0]
            if len(idx):
                ax.axvspan(step[idx[0]], step[idx[-1]], color=colors[p], zorder=0)
                ax.axvline(step[idx[0]], color="k", lw=0.5, alpha=0.4)
                if ax is axs[0]:
                    ax.text(step[idx[0]], 1.02, f"{mp}\n{p}", transform=ax.get_xaxis_transform(),
                            fontsize=9, va="bottom")
        ax.plot(step, y, color="#c0392b", lw=1.2)
        ax.set_ylabel(lab)
    axs[2].set_xlabel("environment steps")
    axs[0].set_title(f"{tag} on Mini Invaders (moving average over {n} updates); "
                     "colour = hidden phase (blue rain, orange sniper, red trap, purple smart)", fontsize=11, pad=34)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "curves.png"), dpi=150)
    plt.close(fig)

    head = (f"{'stage':16s} {'policy':22s} {'reward':>7s} {'hits':>5s} {'deaths':>6s} {'kills':>6s}"
            f" {'shots':>6s} {'gren':>5s} {'acc%':>5s} {'dry':>5s} {'empty%':>6s}"
            f" {'A-crt':>5s} {'G-crt':>5s} {'cover%':>6s} {'rammed':>6s}")
    lines = ["rates per 1000 steps. shots / gren = light shots / grenades thrown; acc% = direct hits",
             "an alien; dry = trigger pulls with no ammo; empty% = time with no ammo at all;",
             "A-crt / G-crt = share of ammo / grenade crates collected; cover% = time under a bunker;",
             "rammed = times a kamikaze diver crashed into the ship",
             head]
    labels, vals_b, vals_a, vals_g, vals_c = [], [], [], [], []
    for k, (p, b, a, g, c) in enumerate(evals):
        for name, d in [(f"{tag} right after switch", b), (f"{tag} end of phase", a),
                        ("Greedy (scripted)", g), ("Cautious (scripted)", c)]:
            lines.append(f"{p:16s} {name:22s} {d['ret']:7.1f} {d['hits']:5.2f} {d['deaths']:6.2f}"
                         f" {d['kills']:6.1f} {d['shots']:6.1f} {d['grenades']:5.1f} {d['acc']:5.1f}"
                         f" {d['dry']:5.1f} {d['empty']:6.1f} {d['ammo']:5.2f} {d['grn']:5.2f}"
                         f" {d['cover']:6.1f} {d['rammed']:6.2f}")
        labels.append(f"{k + 1}. {p.replace('/', chr(10))}")
        vals_b.append(b["ret"]); vals_a.append(a["ret"]); vals_g.append(g["ret"]); vals_c.append(c["ret"])
    txt = "\n".join(lines)
    print("\n" + txt)
    open(os.path.join(outdir, "eval.txt"), "w").write(txt + "\n")

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(12, 4.4))
    wd = 0.2
    ax.bar(x - 1.5 * wd, vals_b, wd, label=f"{tag} right after switch", color="#f5b7b1")
    ax.bar(x - 0.5 * wd, vals_a, wd, label=f"{tag} end of phase", color="#c0392b")
    ax.bar(x + 0.5 * wd, vals_g, wd, label="Greedy (always grabs bonuses)", color="#f1c40f")
    ax.bar(x + 1.5 * wd, vals_c, wd, label="Cautious (never grabs)", color="#5d6d7e")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(x, labels)
    ax.set_ylabel("reward per 1000 steps")
    ax.legend(fontsize=8, ncol=2)
    ax.set_title("Does the agent adapt to each map and hidden phase?")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "eval.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--updates", type=int, default=60000, help="updates per stage")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", default=None, help="continue training from a saved .npz")
    ap.add_argument("--algo", default="a2c", choices=sorted(AGENTS),
                    help="a2c = [Zhao Alg. 10.2], qac = [Zhao Alg. 10.1]")
    ap.add_argument("--nstep", type=int, default=8,
                    help="n of the n-step target [Zhao 7.3]; 1 = TD error of Alg. 10.2")
    ap.add_argument("--envs", type=int, default=16,
                    help="parallel environments (mini-batch size per step) [Zhao 6.4.4]")
    ap.add_argument("--entropy", type=float, default=0.02,
                    help="entropy weight beta; 0 = the book's algorithms as printed")
    ap.add_argument("--out", default="out", help="output folder")
    args = ap.parse_args()
    _, lg, ev = train(SCHEDULE, args.updates, n_envs=args.envs, n_steps=args.nstep,
                      seed=args.seed, init=args.resume, algo=args.algo, beta=args.entropy,
                      outdir=args.out)
    report(lg, ev, SCHEDULE, outdir=args.out, steps_per_update=args.envs * args.nstep,
           tag=args.algo.upper())
