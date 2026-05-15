import argparse
import asyncio
import shutil
import uuid
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

TRUNCATE_LINES = 8


def truncate(text: str) -> str:
    lines = text.strip().splitlines()
    if len(lines) <= TRUNCATE_LINES:
        return text.strip()
    visible = "\n".join(lines[:TRUNCATE_LINES])
    return f"{visible}\n  … +{len(lines) - TRUNCATE_LINES} lines (ctrl+o to expand)"


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="piccolo-sprite",
        description="Pixel-art sprite generation agent",
    )
    parser.add_argument("-p", "--prompt", help="Run a one-shot prompt and exit")
    parser.add_argument(
        "--run-dir",
        default="run",
        metavar="DIR",
        help="Run directory (default: run)",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Wipe the run directory before starting",
    )
    return parser.parse_args(argv)


async def run_stream(agent, user_input: str, thread_id: str) -> None:
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 64}
    in_reasoning = False

    async for event in agent.astream_events(
        {"messages": [("human", user_input)]},
        config=config,
        version="v2",
    ):
        kind = event["event"]

        if kind == "on_chat_model_stream":
            chunk = event["data"]["chunk"]
            content = chunk.content if isinstance(chunk.content, str) else ""
            if content:
                if not in_reasoning:
                    print("\nThinking: ", end="", flush=True)
                    in_reasoning = True
                print(content, end="", flush=True)

        elif kind == "on_tool_start":
            if in_reasoning:
                print()
                in_reasoning = False
            name = event["name"]
            inputs = event["data"].get("input", {})
            args_str = ", ".join(f"{k}={v}" for k, v in inputs.items())
            print(f"\n⏺ {name}({args_str})")

        elif kind == "on_tool_end":
            output = event["data"].get("output", "")
            print(f"  ⎿  {truncate(str(output))}")

    if in_reasoning:
        print()


async def repl(agent) -> None:
    thread_id = str(uuid.uuid4())
    print("piccolo-sprite  (type 'exit' or ctrl+c to quit)")
    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            break
        if not user_input or user_input.lower() in ("exit", "quit"):
            break
        await run_stream(agent, user_input, thread_id)


async def one_shot(agent, prompt: str) -> None:
    thread_id = str(uuid.uuid4())
    await run_stream(agent, prompt, thread_id)


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)

    if args.clean and run_dir.exists():
        shutil.rmtree(run_dir)

    from .agent import build_agent
    agent = build_agent()

    if args.prompt:
        prompt = args.prompt
        manifest_path = run_dir / "run-manifest.json"
        if run_dir.exists() and manifest_path.exists():
            prompt = (
                f"Resuming existing run in '{args.run_dir}'. "
                f"Read the run-manifest.json first and continue any incomplete work. "
                f"Original request: {args.prompt}"
            )
        asyncio.run(one_shot(agent, prompt))
    else:
        asyncio.run(repl(agent))


if __name__ == "__main__":
    main()
