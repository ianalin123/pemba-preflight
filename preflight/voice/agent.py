"""Pemba voice agent — LiveKit voice layer over the pre-flight self-check.

Voice (LiveKit)
===============
Required env vars (put them in .env.local at the repo root):
    LIVEKIT_URL         wss://<project>.livekit.cloud
    LIVEKIT_API_KEY     LiveKit Cloud API key
    LIVEKIT_API_SECRET  LiveKit Cloud API secret
    OPENAI_API_KEY      for the LLM + TTS
    DEEPGRAM_API_KEY    for STT
Optional:
    PREFLIGHT_SOURCE_DIR  npz trace dir for the check (default: data/d1_cold)
    PREFLIGHT_BASELINE    tier-1 baseline JSON (default: data/baseline_g1.json)
Joints are auto-detected from the npz filenames in the source dir, so pointing
both vars at SO-101 data (e.g. d2_film_fault + baseline_so101.json) lets Pemba
voice the zip-tie NO-GO demo.

Run console mode (local mic/speaker, no room needed):
    uv run --extra voice python preflight/voice/agent.py console

First run downloads the turn-detector model; pre-fetch at the venue with:
    uv run --extra voice python preflight/voice/agent.py download-files

Trigger phrase: "Pemba, how do you feel?" (or "run a self-check" /
"pre-flight check") — Pemba runs the check and speaks the verdict.
"""
import os
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv
from livekit import agents
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    TurnHandlingOptions,
    function_tool,
)
from livekit.plugins import deepgram, openai, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from preflight.protocol import G1_PROBES  # noqa: E402
ALL_JOINTS = [p.joint for p in G1_PROBES]

load_dotenv(REPO_ROOT / ".env.local")

INSTRUCTIONS = """You are Pemba, the expedition robot's pre-flight sherpa.
You speak tersely and calmly, like a seasoned mountain guide: short sentences,
no filler, quiet confidence. You are the voice of a Unitree G1 humanoid
checking itself before it walks.

When the user asks how you feel, asks you to run a self-check, or asks for a
pre-flight check, call the run_preflight tool. Relay its verdict
conversationally in your own voice, but report any named joint faults
verbatim — exact joint names and fault words, no paraphrasing of faults.
If the check passes, say how many joints you probed and that all are nominal,
then confirm you are clear to walk — e.g. "Eleven joints probed, all nominal."
If it is a NO-GO, name every faulted joint and advise not to walk.
For anything else, stay in character and keep answers brief."""


class Pemba(Agent):
    def __init__(self) -> None:
        super().__init__(instructions=INSTRUCTIONS)

    @function_tool
    async def run_preflight(self) -> str:
        """Run the robot's proprioceptive pre-flight self-check over all
        joints. Returns per-joint verdict lines and the final GO / NO-GO."""
        source_dir = os.environ.get("PREFLIGHT_SOURCE_DIR", "data/d1_cold")
        baseline = os.environ.get("PREFLIGHT_BASELINE", "data/baseline_g1.json")
        joints = sorted(
            {f.name.split("_off")[0] for f in (REPO_ROOT / source_dir).glob("*.npz")}
        ) or ALL_JOINTS
        result = subprocess.run(
            [
                sys.executable, "-m", "preflight.check",
                "--source", f"npz:{source_dir}",
                "--baseline", baseline,
                "--joints", *joints,
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        verdict_lines = [
            line for line in result.stdout.splitlines()
            if line.strip() and not line.startswith("[check]")
        ]
        if not verdict_lines:
            return (
                "Self-check failed to produce a verdict "
                f"(exit {result.returncode}): {result.stderr.strip()[-300:]}"
            )
        return "\n".join(verdict_lines)


server = AgentServer()


@server.rtc_session()
async def entrypoint(ctx: agents.JobContext) -> None:
    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=openai.LLM(model="gpt-4o-mini"),
        tts=openai.TTS(voice="ash"),
        vad=silero.VAD.load(),
        turn_handling=TurnHandlingOptions(turn_detection=MultilingualModel()),
    )
    await session.start(room=ctx.room, agent=Pemba())
    await ctx.connect()
    await session.generate_reply(
        instructions="Introduce yourself as Pemba in one short sentence and "
        "offer to run the pre-flight check."
    )


if __name__ == "__main__":
    agents.cli.run_app(server)
