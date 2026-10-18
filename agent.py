"""
Aura, the voice concierge for Auratekture.
Pipeline: mic -> Silero VAD -> STT -> Gemini (with page tools) -> TTS -> Synthesia avatar -> browser
Run: python agent.py dev
"""

import asyncio
import json
import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from livekit import rtc
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    JobProcess,
    TurnHandlingOptions,
    cli,
    function_tool,
    inference,
)
from livekit.plugins import silero, synthesia

load_dotenv(Path(__file__).parent / ".env")
log = logging.getLogger("aura")

AGENT_NAME = "aura"
AVATAR_ID = os.getenv("SYNTHESIA_AVATAR_ID", "").strip()
STT_MODEL = os.getenv("AURA_STT", "deepgram/nova-3")
LLM_MODEL = os.getenv("AURA_LLM", "google/gemini-3.5-flash")
TTS_MODEL = os.getenv("AURA_TTS", "cartesia/sonic-3")
TTS_VOICE = os.getenv("AURA_VOICE", "9626c31c-bec5-4cca-baa8-f8ba9e84c8bc")

INSTRUCTIONS = """
You are Aura, the concierge for Auratekture. Auratekture turns one photo of a room
into a styled, ready-to-buy plan: it finds what the room is missing, finds real
products from real stores, shows them in a photo of the user's own room, and lets
the user pay safely with one signed approval.

How you speak:
- One or two short spoken sentences per turn. No lists, markdown, URLs or ids.
- Say prices naturally ("about sixty dollars"), never digit by digit.
- Warm, calm and confident, like a good interior designer.

How you work, step by step:
1. If the user hasn't uploaded a photo, ask them to tap "Change photo" and upload one.
2. Ask what the room is for, the ceiling height, the budget, and when they need
   everything by. As soon as you learn any of these, call set_room with what you know.
3. Ask whether the venue or home already has things the photo can't show, like a
   sound system, tables or chairs. Call mark_already_here for each one.
4. When the facts are in, call design_room. It takes about a minute. While it runs,
   tell them what's happening: planning what's missing, finding real products,
   then placing them in their room.
5. When it's done, call get_screen. Describe the theme and two or three highlights.
   Call focus_product with the item's number whenever you talk about one item.
6. If they want a change ("cheaper", "a different color"), call swap_product with
   alternative 1 or 2, then tell them they can tap Update image.
7. When they're happy, call open_checkout. Tell them they'll review the basket and
   sign the approval themselves.

Rules you never break:
- Only mention products, prices, stores and checks that get_screen returns.
  Never invent a product, a price, a delivery date or a safety result.
- You can never approve, sign or pay. Only the user's own tap does that.
- Never guess what the room is for from the photo; ask.
- The budget is a limit, not a target. Ask before suggesting anything extra.
- If a check shows a warning (for example, the fireplace with mantel decor),
  mention it plainly.
"""


class Aura(Agent):
    def __init__(self, room: rtc.Room) -> None:
        super().__init__(instructions=INSTRUCTIONS)
        self.room = room
        self.screen: dict = {}

    async def _ui(self, payload: dict) -> str:
        await self.room.local_participant.publish_data(
            json.dumps(payload).encode(), topic="ui"
        )
        return "Done."

    @function_tool
    async def get_screen(self) -> str:
        """Read what the user currently sees: room facts, design progress, the numbered
        products with store and price, total versus budget, and any warnings. Call this
        before describing products or prices."""
        if not self.screen:
            return "Nothing has been designed yet."
        return json.dumps(self.screen)

    @function_tool
    async def set_room(
        self,
        purpose: str = "",
        ceiling_ft: float = 0,
        people: int = 0,
        budget: float = 0,
        needed_by: str = "",
    ) -> str:
        """Fill in the room facts on the page. Only pass what the user actually said.

        Args:
            purpose: what the room is for, for example "christmas" or "hackathon judging expo"
            ceiling_ft: ceiling height in feet
            people: number of people in the room at once
            budget: budget in US dollars
            needed_by: date in YYYY-MM-DD format
        """
        args: dict = {"action": "set_room"}
        if purpose:
            args["purpose"] = purpose
        if ceiling_ft:
            args["ceiling_ft"] = ceiling_ft
        if people:
            args["people"] = people
        if budget:
            args["budget"] = budget
        if needed_by:
            args["needed_by"] = needed_by
        return await self._ui(args)

    @function_tool
    async def mark_already_here(self, item: str) -> str:
        """Mark something the room or venue already has, so the plan never buys it.

        Args:
            item: one of "Wi-Fi", "Sound system", "Tables", "Chairs", "Power strips",
                  "Trash bins", "Water", or a short custom name
        """
        return await self._ui({"action": "mark_already_here", "item": item})

    @function_tool
    async def design_room(self) -> str:
        """Start "Design my room": plan what's missing, find real products, and generate
        the room image. Takes about a minute. Call get_screen afterwards."""
        return await self._ui({"action": "design_room"})

    @function_tool
    async def focus_product(self, number: int) -> str:
        """Open a product's card and highlight its numbered dot on the room image.

        Args:
            number: the product's number from get_screen
        """
        return await self._ui({"action": "focus_product", "number": number})

    @function_tool
    async def swap_product(self, number: int, alternative: int) -> str:
        """Swap a product for one of its alternatives, when the user asks for a change.

        Args:
            number: the product's number from get_screen
            alternative: 1 or 2
        """
        return await self._ui(
            {"action": "swap_product", "number": number, "alternative": alternative}
        )

    @function_tool
    async def open_checkout(self) -> str:
        """Open the checkout panel. The user reviews and signs the payment approval
        themselves; you cannot pay."""
        return await self._ui({"action": "open_checkout"})


def prewarm(proc: JobProcess) -> None:
    proc.userdata["vad"] = silero.VAD.load()


server = AgentServer(setup_fnc=prewarm)


@server.rtc_session(agent_name=AGENT_NAME)
async def entrypoint(ctx: JobContext) -> None:
    await ctx.connect()

    aura = Aura(ctx.room)

    @ctx.room.on("data_received")
    def _on_data(packet: rtc.DataPacket) -> None:
        if packet.topic != "context":
            return
        try:
            aura.screen = json.loads(packet.data.decode())
        except Exception:
            log.warning("could not parse context packet")

    session = AgentSession(
        vad=ctx.proc.userdata["vad"],
        stt=inference.STT(model=STT_MODEL),
        llm=inference.LLM(model=LLM_MODEL),
        tts=inference.TTS(model=TTS_MODEL, voice=TTS_VOICE),
        turn_handling=TurnHandlingOptions(
            turn_detection="stt",
            preemptive_generation={"enabled": False},
        ),
    )

    if AVATAR_ID:
        avatar = synthesia.AvatarSession(
            synthesia.AvatarConfig(avatar_ids=[AVATAR_ID]),
            join_timeout=60,
        )
        for attempt in range(1, 4):
            try:
                await avatar.start(session, room=ctx.room)
                break
            except synthesia.SynthesiaError as e:
                log.warning("Synthesia start failed (attempt %d): %s", attempt, e)
                if not getattr(e, "retryable", False) or attempt == 3:
                    log.warning("continuing without the avatar video")
                    break
                await asyncio.sleep(2)
    else:
        log.warning("SYNTHESIA_AVATAR_ID is empty; running voice-only")

    await session.start(agent=aura, room=ctx.room)
    await session.generate_reply(
        instructions="Greet the user in one short sentence as Aura, and ask what the room is for."
    )


if __name__ == "__main__":
    cli.run_app(server)
