# Slurm Cluster with Ansible: Provisioning Guide

These Ansible playbooks build a working Slurm cluster on Ubuntu 24.04, with
one controller (`slurmctld`), compute nodes (`slurmd`) and MUNGE for
authentication. You run them from your own machine over SSH.

```
your machine (Ansible) --SSH-->  controller (slurmctld)
                       \--SSH-->  compute node(s) (slurmd)
```

**Run every command in this guide from the repository root**, where
`ansible.cfg` lives.

---

## 0. Requirements on your machine

| Tool | Needed for |
|---|---|
| ansible-core >= 2.15 (`pipx install ansible-core`) | provisioning |
| OpenSSH client (`ssh`, `ssh-keygen`) | reaching the nodes |
| Docker + Compose v2 | only if you simulate the machines (step 1) |

---

## 1. Get the machines

You need one controller and at least one compute node. Each needs:

- Ubuntu 24.04
- reachable over SSH from your machine
- an account that is root or has `sudo`

If you have such machines, skip to step 2.

**No hardware? Simulate the machines.** `infrastructure/docker-compose.yml`
starts two containers that behave like plain Ubuntu servers. They run
systemd and SSH, share a private network, and have **no Slurm software
installed**.

```bash
# SSH key that the simulated machines will accept for root
mkdir -p infrastructure/ssh
ssh-keygen -t ed25519 -N '' -C slurm-sim -f infrastructure/ssh/id_ed25519

# build the image and start the machines
docker compose -f infrastructure/docker-compose.yml up -d --build
docker compose -f infrastructure/docker-compose.yml ps     # slurmctl and c1: "Up"
```

| Machine | Hostname | SSH from your machine |
|---|---|---|
| controller | `slurmctl` | `127.0.0.1:2221` |
| compute node | `c1` | `127.0.0.1:2222` |

---

## 2. Set the Ansible inventory

`inventory/hosts.yml` says which machine has which role and how to reach it:

- `slurm_controller`: the host that runs `slurmctld`
- `slurm_compute`: the hosts that run `slurmd`

Host names must equal each machine's short hostname (`hostname -s`), because
Slurm identifies nodes by hostname. Only the host names and connection
variables differ between the two examples below.

**Example A: real nodes on your network.** Ansible logs in as a sudo user
with your SSH key:

```yaml
all:
  children:
    slurm_controller:
      hosts:
        hpc-ctl01:
          ansible_host: 192.168.1.10
    slurm_compute:
      hosts:
        hpc-c01:
          ansible_host: 192.168.1.21
    slurm_cluster:
      children:
        slurm_controller:
        slurm_compute:
      vars:
        ansible_user: admin            # account with sudo on every node
        ansible_become: true
```

**Example B: simulated nodes from step 1.** This is the file as shipped:

```yaml
all:
  children:
    slurm_controller:
      hosts:
        slurmctl:
          ansible_host: 127.0.0.1
          ansible_port: 2221
    slurm_compute:
      hosts:
        c1:
          ansible_host: 127.0.0.1
          ansible_port: 2222
    slurm_cluster:
      children:
        slurm_controller:
        slurm_compute:
      vars:
        ansible_user: root
        ansible_ssh_private_key_file: infrastructure/ssh/id_ed25519
        # the simulated nodes get new SSH host keys whenever they are recreated
        ansible_ssh_common_args: -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null
```

Check that Ansible reaches every node:

```bash
ansible slurm_cluster -m ping
```

Every host must answer `pong` before you continue.

**Cluster settings** live in `inventory/group_vars/all.yml`: cluster name,
partitions, and cluster users. Anything not set there falls back to
`ansible/roles/slurm_defaults/defaults/main.yml`.

When the nodes are containers, the playbooks adjust these settings
automatically:

| Setting | Machines | Containers |
|---|---|---|
| `ProctrackType` / `TaskPlugin` | `cgroup` / `cgroup,affinity` | `linuxproc` / `affinity` |
| `JobAcctGatherType` | `jobacct_gather/cgroup` | `jobacct_gather/linux` |
| `SlurmdParameters` | (none) | `config_overrides` |
| CPU / memory per node | as detected (minus 512 MB for the OS) | capped at 4 CPUs / 2 GB |
| chrony, `/etc/hosts` | managed by Ansible | left to the container host |

The container settings exist because containers share the host's kernel,
clock and cgroups, and every container sees all of the host's CPUs.

---

## 3. Provision the cluster

```bash
ansible-playbook ansible/site.yml
```

Add `-K` if your `ansible_user` needs a sudo password.

`site.yml` runs these plays in order, following the Slurm Quick Start
Administrator Guide:

| # | Play | Hosts | What it does |
|---|---|---|---|
| 1 | Wait for SSH | all | waits until each node accepts connections |
| 2 | `common` | all | base packages, chrony, `/etc/hosts`, and a `slurm` user with UID/GID 64030 |
| 3 | `munge` | all | creates one key on your machine (`inventory/.secrets/munge.key`), copies it to every node, starts `munged`, and checks it with `munge -n \| unmunge` |
| 4 | `slurm_users` | all | cluster users with the same UID on every node |
| 5 | `slurmd` (detect) | compute | installs `slurmd` and gcc, and reads the hardware with `slurmd -C` |
| 6 | `slurm_config` | all | writes the same `slurm.conf` and `cgroup.conf` to every node, with node lines built from the detected hardware |
| 7 | `slurmctld` | controller | starts the controller and waits for `scontrol ping` to report UP |
| 8 | `slurmd` | compute | starts `slurmd` and waits until the node is `idle` |
| 9 | Smoke test | controller | runs `srun hostname` on every compute node as `hpcuser` |

A successful run ends with `Job ran on: <compute nodes>` and `failed=0`.

The playbook is **idempotent**: running it again reports `changed=0`. After
you change the inventory or the settings, run it again. A daemon restarts only
when its configuration changed.

---

## 4. Verify the cluster

```bash
ansible slurm_controller -a "sinfo"
ansible slurm_controller -a "scontrol show nodes"
ansible slurm_controller -a "cat /etc/slurm/slurm.conf"
```

You should see something like:

```
PARTITION AVAIL  TIMELIMIT  NODES  STATE NODELIST
debug*       up    1:00:00      1   idle c1
```

---

## 5. Run the OpenMP job

The job lives in `jobs/openmp/`:

- **`pi_omp.c`** computes pi by numerical integration, first serially and then
  with `#pragma omp parallel for reduction`. It prints which CPU each thread
  is pinned to, plus the speedup.
- **`pi_omp.sbatch`** requests 1 node, 1 task and N CPUs, and sets
  `OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK`, `OMP_PLACES=cores` and
  `OMP_PROC_BIND=close`.

Run it with:

```bash
ansible-playbook ansible/playbooks/openmp_job.yml
ansible-playbook ansible/playbooks/openmp_job.yml -e omp_threads=8   # more threads
```

There is no shared filesystem yet, so the playbook:

1. Compiles `pi_omp.c` on every compute node, in `/home/hpcuser/openmp`.
2. Submits the job from the controller with `sbatch --wait`, as `hpcuser`. It
   asks for `omp_threads` CPUs (default 4), capped at the node's CPU count.
3. Reads the job's output file on the node that ran it and prints it:

```
Node: c1 | CPUs allocated: 2 | OMP_NUM_THREADS=2
  thread  0 of  2 on CPU 0
  thread  1 of  2 on CPU 1
version                   pi    abs error   time [s]
serial     3.141592653589552    2.411e-13     0.4268
openmp     3.141592653590022    2.287e-13     0.2151
Speedup: 1.98x with 2 threads (efficiency 99%)
State: COMPLETED  ExitCode: 0:0
```

After the playbook has run once, you can also submit the job by hand on the
controller (`ssh` into it, then `su - hpcuser`):

```bash
cd openmp
sbatch --cpus-per-task=2 pi_omp.sbatch
squeue
```

The output file `pi-omp-<jobid>.out` is written on the compute node.

---

## 6. Troubleshooting

| Symptom | Check |
|---|---|
| Node `down` or `drained` | `ansible slurm_controller -a "sinfo -R"`, then `ansible slurm_controller -a "scontrol update NodeName=<node> State=RESUME"` |
| Daemon won't start | `ansible slurm_cluster -a "tail -n 30 /var/log/slurm/slurmctld.log"` (or `slurmd.log`) |
| Authentication errors | `ansible slurm_cluster -m shell -a "munge -n \| unmunge"`; clocks must be in sync |
| Node marked invalid (wrong CPUs or memory) | compare `slurmd -C` on the node with its `NodeName` line |

---

## 7. Remove the simulated machines

```bash
docker compose -f infrastructure/docker-compose.yml down -v
rm -rf infrastructure/ssh inventory/.secrets
```

---

## 8. Growing the cluster in later iterations

- **More compute nodes**: add hosts under `slurm_compute` (if simulated, also
  add services to `infrastructure/docker-compose.yml`), then rerun
  `site.yml`. Node lines are generated automatically.
- **High availability**: add a second host under `slurm_controller`. The
  template already writes one `SlurmctldHost` line per controller, and the
  first one is the primary. `StateSaveLocation` must then be on storage that
  both controllers share.
- **Database node**: add a `slurm_dbd` group and a `slurmdbd` role, plus
  `AccountingStorageType=accounting_storage/slurmdbd` in `slurm.conf`.
- **Login node**: add a `slurm_login` group that runs `common`, `munge`,
  `slurm_users` and `slurm_config`, with no daemons. A shared `/home` removes
  the per-node compile step of the job playbook.

---

## Repository layout

```
ansible.cfg                        Ansible settings (run commands from here)
inventory/
  hosts.yml                        machines and how to reach them (step 2)
  group_vars/all.yml               cluster settings
ansible/
  site.yml                         provisions the cluster (step 3)
  playbooks/openmp_job.yml         runs the OpenMP job (step 5)
  roles/                           slurm_defaults, common, munge, slurm_users,
                                   slurm_config, slurmctld, slurmd
infrastructure/                    optional simulated machines (step 1)
  docker-compose.yml               controller + compute node containers
  node/Dockerfile                  Ubuntu 24.04 + systemd + sshd
jobs/openmp/                       pi_omp.c, pi_omp.sbatch
```

Software: Ubuntu packages, Slurm 23.11 and MUNGE 0.5.15. References: the
[Quick Start Admin Guide](https://slurm.schedmd.com/quickstart_admin.html)
and [slurm.conf](https://slurm.schedmd.com/slurm.conf.html).
