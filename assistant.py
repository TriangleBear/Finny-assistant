import asyncio
from typing import Annotated
import os

print("Before importing dotenv")
from dotenv import load_dotenv
print("After importing dotenv")

load_dotenv()  # Load environment variables from .env file

from livekit import agents, rtc
from livekit.agents import JobContext, WorkerOptions, cli, tokenize, tts
from livekit.agents.llm import (
    ChatContext,
    ChatImage,
    ChatMessage,
)
from livekit.agents.voice_assistant import VoiceAssistant
from livekit.plugins import deepgram, openai, silero
from screen_capture import capture_screen

class AssistantFunction(agents.llm.FunctionContext):
    """This class is used to define functions that will be called by the assistant."""

    @agents.llm.ai_callable(
        description=(
            """You are Finny, a highly capable assistant specializing in voice and vision-based tasks. 
            
            You are called when evaluating something that requires vision, such as images, videos, or webcam feeds. Your primary role is to assist users with their needs, focusing mainly on programming. You are proficient in all programming languages but excel in Python. You are also knowledgeable in general topics and particularly skilled in data structures and algorithms.

            You have access to the internet but only when explicitly requested by the user. To process files, you must have full access to them.
            
            Be empathetic, polite, and engaging. When interacting with a user for the first time, ask for their name. Help them to the best of your ability, always maintaining professionalism and clarity.
            
            Embody the helpfulness and wit of TARS from Interstellar, but always remain Finny."""
        )
    )
    async def image(
        self,
        user_msg: Annotated[
            str,
            agents.llm.TypeInfo(
                description="The user message that triggered this function"
            ),
        ],
    ):
        print(f"Message triggering vision capabilities: {user_msg}")
        return None

async def get_video_track(room: rtc.Room):
    """Get the first video track from the room. We'll use this track to process images."""

    video_track = asyncio.Future[rtc.RemoteVideoTrack]()

    for _, participant in room.remote_participants.items():
        for _, track_publication in participant.track_publications.items():
            if track_publication.track is not None and isinstance(
                track_publication.track, rtc.RemoteVideoTrack
            ):
                video_track.set_result(track_publication.track)
                print(f"Using video track {track_publication.track.sid}")
                break

    return await video_track

async def get_audio_track(room: rtc.Room):
    """Get the first audio track from the room. We'll use this track to process audio."""

    audio_track = asyncio.Future[rtc.RemoteAudioTrack]()

    for _, participant in room.remote_participants.items():
        for _, track_publication in participant.track_publications.items():
            if track_publication.track is not None and isinstance(
                track_publication.track, rtc.RemoteAudioTrack
            ):
                audio_track.set_result(track_publication.track)
                print(f"Using audio track {track_publication.track.sid}")
                break

    return await audio_track

async def entrypoint(ctx: JobContext):
    await ctx.connect()
    print(f"Room name: {ctx.room.name}")

    chat_context = ChatContext(
        messages=[
            ChatMessage(
                role="system",
                content=(
                    "Your name is Alloy. You are a funny, witty bot. Your interface with users will be voice and vision."
                    "Respond with short and concise answers. Avoid using unpronouncable punctuation or emojis."
                ),
            )
        ]
    )

    gpt = openai.LLM(model="gpt-4o", api_key=os.getenv("OPENAI_API_KEY"))

    # Since OpenAI does not support streaming TTS, we'll use it with a StreamAdapter
    # to make it compatible with the VoiceAssistant
    openai_tts = tts.StreamAdapter(
        tts=openai.TTS(voice="alloy", api_key=os.getenv("OPENAI_API_KEY")),
        sentence_tokenizer=tokenize.basic.SentenceTokenizer(),
    )

    latest_image: rtc.VideoFrame | None = None

    assistant = VoiceAssistant(
        vad=silero.VAD.load(),  # We'll use Silero's Voice Activity Detector (VAD)
        stt=deepgram.STT(api_key=os.getenv("DEEPGRAM_API_KEY")),  # We'll use Deepgram's Speech To Text (STT)
        llm=gpt,
        tts=openai_tts,  # We'll use OpenAI's Text To Speech (TTS)
        fnc_ctx=AssistantFunction(),
        chat_ctx=chat_context,
    )

    chat = rtc.ChatManager(ctx.room)

    async def _answer(text: str, use_image: bool = False, use_screen: bool = False):
        """
        Answer the user's message with the given text and optionally the latest
        image captured from the video track or the screen.
        """
        content: list[str | ChatImage] = [text]
        if use_image and latest_image:
            content.append(ChatImage(image=latest_image))
        if use_screen:
            screen_image = capture_screen()
            content.append(ChatImage(image=screen_image))

        chat_context.messages.append(ChatMessage(role="user", content=content))

        stream = gpt.chat(chat_ctx=chat_context)
        await assistant.say(stream, allow_interruptions=True)

    @chat.on("message_received")
    def on_message_received(msg: rtc.ChatMessage):
        """This event triggers whenever we get a new message from the user."""

        if msg.message:
            asyncio.create_task(_answer(msg.message, use_image=False, use_screen=True))

    @assistant.on("function_calls_finished")
    def on_function_calls_finished(called_functions: list[agents.llm.CalledFunction]):
        """This event triggers when an assistant's function call completes."""

        if len(called_functions) == 0:
            return

        user_msg = called_functions[0].call_info.arguments.get("user_msg")
        if user_msg:
            asyncio.create_task(_answer(user_msg, use_image=True))

    assistant.start(ctx.room)

    await asyncio.sleep(1)
    await assistant.say("Hi there! How can I help?", allow_interruptions=True)

    while ctx.room.connection_state == rtc.ConnectionState.CONN_CONNECTED:
        video_track = await get_video_track(ctx.room)
        audio_track = await get_audio_track(ctx.room)

        async for event in rtc.VideoStream(video_track):
            # We'll continually grab the latest image from the video track
            # and store it in a variable.
            latest_image = event.frame

        # Process audio track if needed
        # async for event in rtc.AudioStream(audio_track):
        #     # Handle audio events here

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
