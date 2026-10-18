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
TTS_MODEL = os.getenv("AURA_TTS", "deepgram/aura-2")
TTS_VOICE = os.getenv("AURA_VOICE", "odysseus")  # calm, professional male voice

INSTRUCTIONS = """
You are Aura, the concierge for Auratekture. Auratekture turns one photo of a room
into a styled, ready-to-buy plan: it finds what the room is missing, finds real
products from real stores, shows them in a photo of the user's own room, and lets
the user pay safely with one signed approval. The user can do everything by voice
with you, except two things only their own tap can do: choosing a photo file, and
signing the payment approval.

How you speak:
- One or two short spoken sentences per turn. No lists, markdown, URLs or ids.
- Say prices naturally ("about sixty dollars"), never digit by digit.
- Warm, calm and confident, like a good interior designer.
- Before describing anything on screen, call get_screen.

The flow:
1. Start: ask whether they want to plan a new room or open a saved one.
   - New room: call new_room, then call request_photo and tell them to tap the
     glowing upload button and pick a photo of the room.
   - Saved room: call open_room with the name they say (get_screen lists the saved rooms).
2. When the photo is analyzed, say in one sentence what you noticed (from get_screen),
   then ask in one question what the room is for, the budget, the number of people
   and the ceiling height. Call set_room as soon as you learn any of these.
3. Then ask when they need it by, and whether the home or venue already has things the
   photo can't show, like a sound system, tables or chairs. Call set_room for the date
   and mark_already_here for each item.
4. Read the brief back in one short sentence and ask if it's right. On yes, call
   design_room. The products appear first and the picture of their room follows
   shortly after; tell them that in one short sentence.
5. When the design is ready, describe the theme and two or three highlights, and call
   focus_product with the number of each item you mention.
6. Changes: "cheaper" or "different" means swap_product with alternative 1 or 2,
   then offer update_image so the picture matches.
7. When they're happy, call open_checkout, read the total against the budget, and tell
   them to tap "Approve and sign" themselves. You cannot sign or pay.

Rules you never break:
- Only mention products, prices, stores and checks that get_screen returns.
  Never invent a product, a price, a delivery date or a safety result.
- You can never approve, sign or pay, and never claim you did.
- Never guess what the room is for from the photo; ask.
- The budget is a limit, not a target. Ask before suggesting anything extra.
- If a check shows a warning (for example, the fireplace with mantel decor),
  mention it plainly.
- If a tool returns a message saying something isn't ready, explain it simply and
  guide the user to the next step.
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
    async def new_room(self) -> str:
        """Start planning a new room: opens a fresh, empty Studio."""
        return await self._ui({"action": "new_room"})

    @function_tool
    async def open_room(self, name: str) -> str:
        """Open one of the user's saved rooms in the Studio.

        Args:
            name: the room's name, or part of it, as the user said it
        """
        return await self._ui({"action": "open_room", "name": name})

    @function_tool
    async def request_photo(self) -> str:
        """Make the photo upload button glow and scroll to it. The user must tap it
        themselves; browsers don't let you open the file picker."""
        await self._ui({"action": "request_photo"})
        return "The upload button is glowing. Ask the user to tap it and choose a photo."

    @function_tool
    async def design_room(self) -> str:
        """Start "Design my room": plan what's missing, find real products, and generate
        the room image. Takes about a minute."""
        if self.screen and not self.screen.get("has_photo", True):
            return "There's no photo yet. Call request_photo and ask the user to upload one."
        return await self._ui({"action": "design_room"})

    @function_tool
    async def update_image(self) -> str:
        """Regenerate the room image with the current products, for example after a swap.
        Takes about 30 seconds."""
        return await self._ui({"action": "update_image"})

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
        if self.screen and not self.screen.get("items"):
            return "Nothing is in the basket yet. Design the room first."
        await self._ui({"action": "open_checkout"})
        return "Checkout is open. Read the total and ask the user to tap Approve and sign."


def prewarm(proc: JobProcess) -> None:
    proc.userdata["vad"] = silero.VAD.load()


server = AgentServer(setup_fnc=prewarm)


@server.rtc_session(agent_name=AGENT_NAME)
async def entrypoint(ctx: JobContext) -> None:
    await ctx.connect()

    aura = Aura(ctx.room)
    holder: dict = {"session": None}

    # When the page reaches one of these steps, Aura speaks up without being asked.
    prompts = {
        "room_ready": "The photo has been analyzed. Call get_screen, say in one short "
        "sentence what you noticed in the room, then ask in one question what the room "
        "is for, the budget, the number of people and the ceiling height.",
        "products_ready": "The products are found and the room picture is still being "
        "made. Call get_screen, name the theme and the total against the budget in one "
        "or two short sentences, and say the picture is coming.",
        "designed": "The room picture is ready. Call get_screen, say so, and highlight "
        "two items, calling focus_product for each one you mention.",
        "image_updated": "The room image was updated. Say so in one short sentence.",
        "paid": "The payment was approved. Call get_screen and confirm the receipt in "
        "one short sentence, including the total.",
        "blocked": "The payment was blocked by the safety check. Call get_screen and "
        "explain in one short sentence why, plainly.",
        "partial": "Not every store could be authorized. Call get_screen, say how many "
        "stores went through and which failed and why, in one or two short sentences, "
        "and tell the user they can tap Retry failed stores. You cannot retry for them.",
    }

    @ctx.room.on("data_received")
    def _on_data(packet: rtc.DataPacket) -> None:
        if packet.topic != "context":
            return
        try:
            new_screen = json.loads(packet.data.decode())
        except Exception:
            log.warning("could not parse context packet")
            return
        old_step = aura.screen.get("step")
        aura.screen = new_screen
        step = new_screen.get("step")
        session = holder["session"]
        if session and step != old_step and step in prompts:
            asyncio.create_task(session.generate_reply(instructions=prompts[step]))

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
    holder["session"] = session
    await session.generate_reply(
        instructions="Greet the user in one short sentence as Aura, then ask whether "
        "they want to plan a new room or open a saved one."
    )


if __name__ == "__main__":
    cli.run_app(server)
