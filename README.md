# Accessing H100 on Dogo -- Client Setup

After getting an account, do the following on your local machine:

Put the following in `~/.ssh/config`:

    CanonicalizeHostname always
    CanonicalDomains csail.mit.edu

    Host jump.csail.mit.edu
      GSSAPIAuthentication yes
      GSSAPIDelegateCredentials yes
      VerifyHostKeyDNS yes
      # optional: uncomment and edit if your CSAIL username doesn't match your local username
      # User akeley98
      ForwardX11 yes

    Host *.csail.mit.edu !jump.csail.mit.edu !login.csail.mit.edu 128.52.* 128.30.* 128.31.*
      GSSAPIAuthentication yes
      GSSAPIDelegateCredentials yes
      ProxyJump jump.csail.mit.edu
      # optional: uncomment and edit if your CSAIL username doesn't match your local username
      User akeley98
      # optional: uncomment if you need X11 forwarding
      ForwardX11 yes

Set up the jump server (this expires every 12? hours)

    kinit $USER@CSAIL.MIT.EDU

SSH into kennel; this is the GPU-free entrypoint into Yoon's cluster.
Do not SSH into any server besides `kennel` directly.

    ssh -X -J $USER@jump.csail.mit.edu $USER@kennel.csail.mit.edu

You may wish to alias both of these in your local `~/.bashrc`

    alias kennel_kinit="kinit $USER@CSAIL.MIT.EDU"
    alias kennel="ssh -X -J $USER@jump.csail.mit.edu $USER@kennel.csail.mit.edu"


# Accessing H100 on Dogo -- Server Setup

Once you are on kennel, use `emacs -nw ~/.bashrc` or whatever editor and add the following

    export DFS=/data/cl/u/$USER/
    alias h100="srun --partition debug --gres=gpu:h100:1 --nodelist=dogo"
    alias h100cpu="srun --partition debug --gres=gpu:h100:0 --nodelist=dogo"
    alias h100bash="TMOUT=300 srun --partition debug --gres=gpu:h100:1 --nodelist=dogo --pty bash"

The `$DFS` directory is shared between `kennel` and `dogo` (the H100 server).
`h100 $FOO` runs `$FOO` on `dogo`.
`h100bash` gives you an interactive shell on `dogo` that's configured to kick you after 5 minutes of inactivity (so the admins will not flag you for wasting GPU time if you forget to exit).


# Exo-GPU setup

Make a directory in `$DFS` for this and run all this on `kennel` (you don't need GPUs for this yet).

    git clone --recurse-submodules https://github.com/exo-lang/exo.git
    cd exo
    git checkout akeley98/wgmma
    python3 -m venv ../venv
    source ../venv/bin/activate
    python3 -m pip install -U pip setuptools wheel

Test that Exo-GPU is working.
This runs CUDA code on one of `dogo's` H100s.
If you are adapting these instructions for Ubuchan, omit `--cuda-run-Sm90a` and modify the `EXO_NVCC` path.

    EXO_NVCC=/usr/local/cuda-12.6/bin/nvcc h100 pytest --cuda-run-Sm80 --cuda-run-Sm90a tests/cuda/


# Build this Repo (GEMM test)

Install `exocc`

    source ../venv/bin/activate  # if your venv from before is not yet activated
    pip3 install dist/*.whl
    # The following will be needed before installing again
    # pip3 install dist/*.whl

Compile and run the gemm testbed.

NOTE: this runs on dogo despite being a CPU-only task since `nvcc` on `kennel` is too out of date (CUDA 12.0 which has compiler bugs for the H100).

    PATH=/usr/local/cuda-12.6/bin/:$PATH h100cpu ninja
    h100 gemm/gemm

For ubuchan, `gemm/gemm` will automatically detect it is not running on an H100 and will not run `sm_90a` kernels.
