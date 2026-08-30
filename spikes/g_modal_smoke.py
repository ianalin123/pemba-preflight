"""G-MODAL kill-test: JAX-on-GPU + MJX batched stepping + headless EGL rendering
inside a Modal A100 container. If all three pass, the D3 corpus pipeline is unblocked.
"""

import modal

app = modal.App("himalaya-gmodal-smoke")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libegl1", "libgles2", "libglvnd0", "libglx0", "libopengl0", "git")
    .pip_install("jax[cuda12]==0.4.38", "mujoco==3.2.7", "mujoco-mjx==3.2.7", "numpy")
    .env({"MUJOCO_GL": "egl", "XLA_PYTHON_CLIENT_PREALLOCATE": "false"})
)

TEST_XML = """
<mujoco>
  <option timestep="0.002"/>
  <worldbody>
    <light pos="0 0 3"/>
    <geom type="plane" size="2 2 .1"/>
    <body pos="0 0 1">
      <joint name="j" type="hinge" axis="0 1 0" frictionloss="0.3"/>
      <geom type="capsule" size=".05 .3" euler="0 90 0"/>
    </body>
  </worldbody>
  <actuator><position name="j" joint="j" kp="40" kv="2"/></actuator>
</mujoco>
"""


@app.function(image=image, gpu="A100", timeout=600)
def smoke():
    import time

    import jax
    import jax.numpy as jnp
    import mujoco
    import numpy as np
    from mujoco import mjx

    results = {}

    # 1. GPU visible to JAX
    devs = jax.devices()
    results["jax_devices"] = str(devs)
    assert any("cuda" in str(d).lower() or "gpu" in str(d).lower() for d in devs), "no GPU"

    # 2. MJX batched stepping (the D3 corpus workhorse)
    m = mujoco.MjModel.from_xml_string(TEST_XML)
    mx = mjx.put_model(m)
    dx = mjx.make_data(mx)

    def step(dx, _):
        return mjx.step(mx, dx), None

    batch = 4096
    rng = jax.random.PRNGKey(0)
    qpos0 = jax.random.uniform(rng, (batch, m.nq), minval=-0.3, maxval=0.3)
    dxs = jax.vmap(lambda q: dx.replace(qpos=q))(qpos0)
    vstep = jax.jit(jax.vmap(lambda d: mjx.step(mx, d)))
    t0 = time.time()
    dxs = vstep(dxs)  # compile
    jax.block_until_ready(dxs.qpos)
    compile_s = time.time() - t0
    t0 = time.time()
    n_steps = 500
    for _ in range(n_steps):
        dxs = vstep(dxs)
    jax.block_until_ready(dxs.qpos)
    steps_per_s = batch * n_steps / (time.time() - t0)
    results["mjx"] = f"batch={batch}, compile={compile_s:.1f}s, {steps_per_s/1e6:.2f}M steps/s"

    # 3. Headless EGL render (demo videos)
    renderer = mujoco.Renderer(m, 240, 320)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    renderer.update_scene(d)
    px = renderer.render()
    assert px.std() > 1.0, f"black frame (std={px.std():.3f}) — EGL broken"
    results["egl"] = f"rendered 240x320, pixel std={px.std():.1f}"

    return results


@app.local_entrypoint()
def main():
    r = smoke.remote()
    for k, v in r.items():
        print(f"  {k}: {v}")
    print("G-MODAL PASSES")
