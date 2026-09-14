# Linux LXC deployment

The first install is **clone → configure → install dependencies → download models →
GPU smoke test → run**. Later starts only need the installed server or systemd.
This guide targets a Debian/Ubuntu x86_64 container with an RTX 5090. CUDA inference
and these Linux service commands still require verification on the target machine;
Mac tests do not establish GPU compatibility, memory usage or generation speed.

## 1. Prepare the container

On the host, install a driver that supports the RTX 5090 and the CUDA 12.8 runtime.
Expose the GPU's compute devices and compatible NVIDIA driver libraries to the
container using your container manager. The kernel driver belongs to the host.
The container needs access to the GPU, control and CUDA UVM devices, not just the
graphics/render device. Device permissions must permit the eventual service user.

Host configuration differs between Proxmox, LXD and plain LXC. For LXD, see
Canonical's [GPU device documentation](https://canonical.com/lxd/docs/default/reference/devices_gpu/)
and [NVIDIA runtime option](https://documentation.ubuntu.com/lxd/stable-5.0/reference/instance_options/).
LXD commands are not Proxmox container configuration commands. This project does
not change host drivers or container isolation settings.

Inside the container, confirm:

```bash
uname -m       # x86_64
nvidia-smi     # must show the RTX 5090
```

`nvidia-smi` checks driver visibility; the CUDA computation below is a separate
check. Set the container's RAM limit substantially above the roughly **50 GiB of
retained transformer/text weights**, leaving room for loading buffers, activations
and the OS. That estimate is not a measured peak. The five model files occupy
**66.2 GiB**; allow additional disk space for the Python/CUDA environment, download
caches, uploaded media and generated videos. Use a local filesystem for mutable
storage; its directories must be separate, on the same filesystem and not symlinks.

## 2. Clone and install

Commands below run inside the container, using an administrator with `sudo`.
If logged in as root, omit `sudo`. Replace `YOUR_REPOSITORY_URL` with the Git remote
where you publish this commit. A local Mac commit must be pushed or transferred
before another machine can clone it.

```bash
sudo apt-get update
sudo apt-get install -y git curl ca-certificates ffmpeg libgomp1

sudo useradd --system --user-group --home-dir /var/lib/ltx-server \
  --shell /usr/sbin/nologin ltx-server
sudo install -d -o ltx-server -g ltx-server -m 0700 /var/lib/ltx-server
sudo install -d -o "$(id -un)" -g "$(id -gn)" -m 0755 /opt/ltx-server
git clone YOUR_REPOSITORY_URL /opt/ltx-server
cd /opt/ltx-server

curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
export UV_PYTHON_INSTALL_DIR=/opt/ltx-server/.python
umask 022
uv python install 3.12
uv sync --locked --extra inference --no-dev --python 3.12 --managed-python
```

Skip account creation if that account already exists. Follow the
[official uv installer instructions](https://docs.astral.sh/uv/getting-started/installation/)
if uv is already managed another way. The explicit
[Python installation directory](https://docs.astral.sh/uv/reference/storage/)
keeps the interpreter outside `/home` and `/root`, which the systemd service hides.
Keep the checkout at this path after installation; virtual environments contain
absolute paths. Recreate the environment on Linux rather than copying the Mac's.

The lockfile pins LTX source and matching PyTorch/torchaudio/torchvision CUDA 12.8
wheels. The default configuration does not require SageAttention, NATTEN or a
separate CUDA toolkit installation. Use `--extra inference` on every subsequent
`uv sync` or `uv run` command; omitting it can remove the inference dependencies.
The commands below invoke the installed `.venv` directly.

## 3. Configure and download

```bash
sudo install -o root -g ltx-server -m 0640 .env.example .env
sudoedit /opt/ltx-server/.env
```

Set these values in `.env`:

```dotenv
DATA_DIR=/var/lib/ltx-server
INFERENCE_BACKEND=ltx
HOST=0.0.0.0
PORT=8000
API_KEY=replace-with-a-long-random-key
USE_FP8=true
WARM_MODEL_ON_START=true
```

Keep the other defaults for the initial test. Set `HF_TOKEN` if access to
[Lightricks/LTX-2.5](https://huggingface.co/Lightricks/LTX-2.5) requires it, and
complete any applicable repository access/license flow. Do not paste secrets
into Git. The example above creates `.env` only on first install; do not overwrite
it with the template during upgrades.

Run the checks and downloader as the account that will run the service:

```bash
sudo -u ltx-server -H nvidia-smi
sudo -u ltx-server -H .venv/bin/python - <<'PY'
import torch
from ltx_server.config import Settings

device = Settings().gpu_device
assert torch.cuda.is_available(), "CUDA is unavailable inside this container"
x = torch.ones((32, 32), device=device)
assert (x @ x)[0, 0].item() == 32
print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(device))
PY

sudo -u ltx-server -H .venv/bin/python scripts/download_models.py --list
sudo -u ltx-server -H .venv/bin/python scripts/download_models.py
sudo -u ltx-server -H .venv/bin/python scripts/smoke_test.py --gpu
```

The downloader verifies every component's pinned size and SHA-256 as it completes.
Re-running it verifies/skips valid files. To verify without downloading, add
`--verify-only`. The server never downloads checkpoints itself. LoRA weights are
optional, separately supplied files; see the [LoRA setup](../README.md) before
using workflows that require them.

The GPU smoke test generates a one-second video with audio, validates its media
properties and removes its temporary output. Run it **with the server stopped**:
they must not compete for the GPU or the storage locks. A passed smoke test covers
that generation only, not all resolutions, workflows or peak memory scenarios.

## 4. Start the server

For a foreground first run:

```bash
sudo -u ltx-server -H .venv/bin/ltx-server
```

From another terminal in `/opt/ltx-server`:

```bash
curl http://127.0.0.1:8000/v1/health
sudo -u ltx-server -H .venv/bin/python scripts/smoke_test.py
```

The API smoke test reads `API_KEY` from `.env` (an exported environment value takes
precedence); it checks the HTTP API without running another GPU generation.
Health remains available while weights preload. Check `model_state` and
`model_ready`; a successful HTTP health response alone does not prove readiness.
If `HEALTH_REQUIRES_AUTH=true`, supply the bearer key to curl as well.

Open `http://CONTAINER_IP:8000/docs` for the interactive API, then use **Authorize**
with your configured key. Allow TCP port 8000 through the relevant container/host
firewall for the clients that should reach it. The built-in server speaks HTTP;
use a TLS reverse proxy if serving it across an untrusted network.

Stop the foreground server with Ctrl-C before enabling the persistent service:

```bash
sudo install -m 0644 deploy/systemd/ltx-server.service \
  /etc/systemd/system/ltx-server.service
sudo systemctl daemon-reload
sudo systemctl enable --now ltx-server
sudo journalctl -u ltx-server -f
```

The unit runs one process as `ltx-server`, restarts after a crash, and stores all
writable state/caches below `/var/lib/ltx-server`. Model path overrides must be
readable by this account and outside the hidden home directories. Mutable path
overrides must remain inside the writable state directory unless you adjust the
unit. Enable container autostart separately in your container manager if desired.

Useful service commands:

```bash
sudo systemctl status ltx-server
sudo systemctl stop ltx-server
sudo systemctl restart ltx-server
sudo journalctl -u ltx-server -n 100 --no-pager
```

## Troubleshooting and updates

- **Driver works on the host, but not in the container:** check the host's GPU
  device/library passthrough and container device permissions. If `nvidia-smi`
  works but the tensor check fails, check CUDA UVM access and driver/runtime
  compatibility. Reinstalling the Python environment cannot repair missing host
  device access.
- **Service cannot find Python:** inspect `.venv/bin/python` and its symlink
  target. Recreate the environment with `UV_PYTHON_INSTALL_DIR` as above if the
  interpreter lives under a hidden home directory.
- **`226/NAMESPACE` before Python starts:** the container may disallow the mount
  namespaces used by the unit's `ProtectSystem`, `ProtectHome` or `PrivateTmp`
  settings. Inspect the journal and your container manager's systemd support;
  the foreground command can isolate this from an application startup failure.
- **Killed process or memory error:** inspect the container's RAM limit and host
  OOM logs as well as VRAM. `CACHE_TEXT_ENCODER=false` reduces retained host RAM
  at the cost of reloading the encoder for each prompt. Other offload profiles
  are described in [performance validation](performance.md); their suitability
  still needs measurement on this machine.
- **`Storage is in use`:** stop the other server or smoke/benchmark process using
  those paths. Use one server process and one worker for this GPU.

To update, stop the service, pull the intended commit in `/opt/ltx-server`, export
`UV_PYTHON_INSTALL_DIR=/opt/ltx-server/.python` again and repeat the locked inference
sync command. Preserve `.env` and `/var/lib/ltx-server`. Reinstall the unit and run
`systemctl daemon-reload` if it changed. Run the downloader if the model manifest
changed, run the GPU smoke test, then start the service. Pending jobs and job
records are in memory and are not resumed after restart.
