"""One terminal reader for concurrent adjudication and manual takeover requests."""
from concurrent.futures import Future
import os
import select
import sys
import threading

from .contracts import HumanDecisionPending


class TerminalDialogue:
    def __init__(self, *, read=input, write=print):
        self.read, self.write = read, write
        self.lock = threading.RLock()
        self.pending = {}
        self.sequence = 0
        self.started = False
        self.closed = False
        self.reader = None
        self.stop_reader = threading.Event()

    def _read_line(self):
        if self.read is not input:
            return self.read("")
        # A daemon blocked in input() holds stdin's buffered-reader lock and
        # can abort Python during shutdown. Read the terminal fd interruptibly
        # instead; close() must not require a final operator keystroke.
        fd, data = sys.stdin.fileno(), bytearray()
        while not self.stop_reader.is_set():
            if not select.select([fd], [], [], .1)[0]:
                continue
            char = os.read(fd, 1)
            if not char:
                if data:
                    return data.decode(sys.stdin.encoding or "utf-8", errors="replace")
                raise EOFError
            if char == b"\n":
                return data.decode(sys.stdin.encoding or "utf-8", errors="replace")
            data.extend(char)
        raise EOFError

    def request(self, kind, prompt, parse=lambda text: text):
        future = Future()
        with self.lock:
            if self.closed:
                future.set_exception(HumanDecisionPending("Terminal dialogue is closed"))
                return future
            self.sequence += 1
            key = kind + str(self.sequence)
            self.pending[key] = (future, parse)
            self.write(f"\n[{key}] {prompt}\n回复格式：{key} 你的回答（只有一个问题时可直接回答）")
            if not self.started:
                self.started = True
                self.reader = threading.Thread(target=self._reader, name="human-terminal", daemon=True)
                self.reader.start()
        return future

    def read_takeover(self, prompt):
        return self.request("T", prompt).result()

    def _reader(self):
        while not self.closed:
            try:
                line = self._read_line()
            except (EOFError, KeyboardInterrupt):
                self.close()
                return
            with self.lock:
                words = line.strip().split(maxsplit=1)
                if words and words[0] in self.pending and len(words) == 2:
                    key, answer = words
                elif len(self.pending) == 1:
                    key, answer = next(iter(self.pending)), line.strip()
                else:
                    self.write("请用问题编号回复；当前问题：" + ", ".join(self.pending))
                    continue
                future, parse = self.pending[key]
                try:
                    result = parse(answer)
                except (ValueError, TypeError) as exc:
                    self.write(f"[{key}] {exc}；请重新回答。")
                    continue
                del self.pending[key]
                if not future.done():
                    future.set_result(result)

    def cancel_pending(self, reason="This episode has ended"):
        """End a sequential episode's questions without creating a new reader."""
        with self.lock:
            for future, _ in self.pending.values():
                if not future.done():
                    future.set_exception(HumanDecisionPending(reason))
            self.pending.clear()

    def close(self):
        # A blocked terminal read must not prevent a robot stop or process exit.
        with self.lock:
            self.closed = True
            self.stop_reader.set()
            self.cancel_pending("Operator input is unavailable")
        if self.reader is not None and self.reader is not threading.current_thread():
            self.reader.join(timeout=1)
