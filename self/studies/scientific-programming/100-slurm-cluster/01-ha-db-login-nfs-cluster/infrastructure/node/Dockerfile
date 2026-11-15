# Simulated "bare metal" node: a plain Ubuntu 24.04 machine running
# systemd + sshd, with nothing Slurm-related installed. Ansible does the rest,
# exactly as it would on a real server.
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive \
    container=docker

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      systemd systemd-sysv dbus openssh-server python3 sudo acl kmod \
      iproute2 iputils-ping procps less ca-certificates \
 && apt-get clean && rm -rf /var/lib/apt/lists/*

# Trim systemd units that make no sense inside a container
RUN cd /lib/systemd/system/sysinit.target.wants/ \
 && find . -mindepth 1 ! -name 'systemd-tmpfiles-setup.service' -delete \
 && rm -f /lib/systemd/system/multi-user.target.wants/* \
          /etc/systemd/system/*.wants/* \
          /lib/systemd/system/local-fs.target.wants/* \
          /lib/systemd/system/sockets.target.wants/*udev* \
          /lib/systemd/system/sockets.target.wants/*initctl* \
 && systemctl mask getty.target console-getty.service systemd-udevd.service \
          systemd-udev-trigger.service systemd-logind.service

# At every boot: create this machine's own SSH host keys and authorize the
# key mounted at /keys/authorized_keys for root.
RUN rm -f /etc/ssh/ssh_host_* \
 && printf '%s\n' \
      'PermitRootLogin prohibit-password' \
      'PasswordAuthentication no' \
      'UseDNS no' > /etc/ssh/sshd_config.d/10-sim-cluster.conf \
 && printf '%s\n' '#!/bin/sh' 'set -e' \
      'ssh-keygen -A' \
      'install -d -m 0700 /root/.ssh' \
      'if [ -f /keys/authorized_keys ]; then install -m 0600 /keys/authorized_keys /root/.ssh/authorized_keys; fi' \
      > /usr/local/sbin/sim-node-bootstrap \
 && chmod +x /usr/local/sbin/sim-node-bootstrap \
 && printf '%s\n' \
      '[Unit]' 'Description=Simulated node bootstrap (SSH host keys, root key)' 'Before=ssh.service' \
      '[Service]' 'Type=oneshot' 'ExecStart=/usr/local/sbin/sim-node-bootstrap' \
      '[Install]' 'WantedBy=multi-user.target' \
      > /etc/systemd/system/sim-node-bootstrap.service \
 && systemctl enable sim-node-bootstrap.service ssh.socket

STOPSIGNAL SIGRTMIN+3
CMD ["/sbin/init"]
