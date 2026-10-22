"""Work-stealing scheduler that hands nonce blocks to devices and collects shares."""
import heapq
import threading
from dataclasses import dataclass, field


@dataclass(order=True)
class Block:
    priority: int
    start: int = field(compare=False)
    size: int = field(compare=False)
    attempts: int = field(default=0, compare=False)


class Scheduler:
    def __init__(self, nonce_space: int, block_size: int, devices: int, target_bits: int):
        self.block_size = block_size
        self.target_bits = target_bits
        self.queues = [[] for _ in range(devices)]
        self.locks = [threading.Lock() for _ in range(devices)]
        self.best = 0
        self.found = None
        self._done = threading.Event()
        for index, start in enumerate(range(0, nonce_space, block_size)):
            heapq.heappush(self.queues[index % devices], Block(index, start, block_size))

    def next_block(self, device: int):
        with self.locks[device]:
            if self.queues[device]:
                return heapq.heappop(self.queues[device])
        return self._steal(device)

    def _steal(self, thief: int):
        victims = sorted(range(len(self.queues)), key=lambda d: -len(self.queues[d]))
        for victim in victims:
            if victim == thief:
                continue
            with self.locks[victim]:
                if len(self.queues[victim]) > 1:
                    return self.queues[victim].pop()  # take from the far end
        return None

    def requeue(self, device: int, block: Block) -> None:
        block.attempts += 1
        block.priority = -block.attempts  # retried blocks go first
        with self.locks[device]:
            heapq.heappush(self.queues[device], block)

    def report(self, nonce: int, zero_bits: int) -> None:
        self.best = max(self.best, zero_bits)
        if zero_bits >= self.target_bits and self.found is None:
            self.found = nonce
            self._done.set()

    def wait(self, timeout: float | None = None) -> bool:
        return self._done.wait(timeout)

    def pending(self) -> int:
        return sum(len(queue) for queue in self.queues)
