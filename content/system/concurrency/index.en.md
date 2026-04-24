---
title: "Concurrency in Python"
date: 2026-04-24
---

Python concurrency is mostly about choosing the right execution model and understanding the synchronization primitives behind it. In practice, the first distinction is between threads and processes.

### Process vs Thread

Threads in the same process share memory. Processes do not. That difference affects synchronization, communication, and whether CPU work can run in parallel.

| Feature | Thread | Process |
| --- | --- | --- |
| Memory | Shared | Separate |
| GIL | Shared | Separate |
| True CPU parallelism | No | Yes |
| IPC required | No | Yes |
| Overhead | Lower | Higher |

In CPython, threads share one Global Interpreter Lock (GIL), so only one thread executes Python bytecode at a time. As a result:

- threads are usually a good fit for I/O-bound work
- processes are a better fit for CPU-bound work

### Multiprocessing

`multiprocessing` gives each worker its own Python interpreter and memory space. This avoids the GIL for CPU-heavy tasks, but communication becomes more expensive because data must cross process boundaries.

```python
from multiprocessing import Process, Queue, Lock


def worker(lock, q):
    with lock:
        print("critical section")
    item = q.get()
    print(item)


if __name__ == "__main__":
    q = Queue(maxsize=2)
    q.put("A")

    lock = Lock()
    p = Process(target=worker, args=(lock, q))
    p.start()
    p.join()
```

Key points:

- `Queue` is process-safe and supports blocking `put()` and `get()`.
- `Lock` can coordinate access across processes.
- Process isolation improves parallelism, but startup and IPC cost more than threads.

### Threading

`threading` is lighter weight than `multiprocessing`, but all Python threads in one process share the same GIL in CPython.

```python
import threading
import time


def worker():
    print("start")
    time.sleep(1)
    print("end")


t = threading.Thread(target=worker)
t.start()
t.join()
```

Each thread can also have its own thread-local state:

```python
import threading


local_data = threading.local()


def worker():
    local_data.x = 1
```

### Locks and Conditions

A `Lock` protects a critical section:

```python
import threading


lock = threading.Lock()
with lock:
    pass
```

This does not mean threads busy-wait in user code. If a lock is unavailable, the waiting thread blocks until it can acquire it.

A `Condition` is used when a thread must wait for a state change, not just mutual exclusion:

```python
import threading


lock = threading.Lock()
cond = threading.Condition(lock)

with cond:
    cond.wait()
    cond.notify()
```

The usual pattern is:

- hold the lock
- check a predicate in a `while` loop
- call `wait()` if the predicate is false
- call `notify()` or `notify_all()` after changing the shared state

For producer-consumer code:

```python
with cond:
    while not queue:
        cond.wait()
    item = queue.pop(0)

with cond:
    queue.append(item)
    cond.notify()
```

Python also provides `queue.Queue`, a thread-safe FIFO queue built on top of a deque, a lock, and condition variables such as `not_empty` and `not_full`.

### Readers-Writer Lock

A readers-writer lock allows concurrent readers but requires writers to be exclusive:

```python
import threading


class RWLock:
    def __init__(self):
        self._cond = threading.Condition(threading.Lock())
        self._readers = 0
        self._writer = False

    def acquire_read(self):
        with self._cond:
            while self._writer:
                self._cond.wait()
            self._readers += 1

    def release_read(self):
        with self._cond:
            self._readers -= 1
            if self._readers == 0:
                self._cond.notify_all()

    def acquire_write(self):
        with self._cond:
            while self._writer or self._readers > 0:
                self._cond.wait()
            self._writer = True

    def release_write(self):
        with self._cond:
            self._writer = False
            self._cond.notify_all()
```

This implementation is simple but not strictly fair. Under contention, either readers or writers may starve unless the policy is extended.

### Bounded Blocking Queue

A bounded blocking queue is a common concurrency primitive for producer-consumer systems. It supports multiple producers and consumers while applying backpressure when the buffer reaches capacity:

- `put(item)` blocks when the queue is full
- `get()` blocks when the queue is empty

The usual implementation uses one mutex and two condition variables:

```python
import threading
from collections import deque


class BoundedQueue:
    def __init__(self, capacity):
        self.capacity = capacity
        self.queue = deque()
        self.lock = threading.Lock()
        self.not_empty = threading.Condition(self.lock)
        self.not_full = threading.Condition(self.lock)

    def put(self, item):
        with self.not_full:
            while len(self.queue) >= self.capacity:
                self.not_full.wait()
            self.queue.append(item)
            self.not_empty.notify()

    def get(self):
        with self.not_empty:
            while not self.queue:
                self.not_empty.wait()
            item = self.queue.popleft()
            self.not_full.notify()
            return item
```

This design works because:

- The mutex protects the queue state.
- `not_full` blocks producers when capacity is exhausted.
- `not_empty` blocks consumers when no data is available.
- The `while` loop is required because condition waits may wake spuriously.

Practical considerations:

- Timeout support: compute a deadline, then call `wait(remaining)` until the condition is satisfied or the timeout expires.
- `deque` instead of `list`: `popleft()` is `O(1)`, while `pop(0)` is `O(n)`.
- `notify()` vs `notify_all()`: `notify()` wakes one waiter and is usually better for throughput; `notify_all()` wakes everyone and can cause unnecessary contention.
- Starvation and fairness: Python condition variables are not strictly fair. If fairness matters, use an explicit FIFO waiter policy or a semaphore-based design.

### Thread Pool

A thread pool is a simple execution model built from a shared task queue and a fixed set of worker threads:

- the main thread submits tasks
- worker threads pull tasks and execute them

```python
import threading
import queue


class ThreadPool:
    def __init__(self, num_workers):
        self.tasks = queue.Queue()
        self.workers = []

        for _ in range(num_workers):
            t = threading.Thread(target=self._worker)
            t.start()
            self.workers.append(t)

    def _worker(self):
        while True:
            task = self.tasks.get()
            if task is None:
                self.tasks.task_done()
                break
            try:
                task()
            finally:
                self.tasks.task_done()

    def submit(self, fn):
        self.tasks.put(fn)

    def shutdown(self):
        for _ in self.workers:
            self.tasks.put(None)
        for t in self.workers:
            t.join()
```

At a high level:

```text
ThreadPool = task queue + worker threads
```

Practical considerations:

- CPU-bound work: threads are a poor fit in CPython because of the GIL; use `multiprocessing` for parallel CPU execution.
- Starvation: a plain FIFO queue is simple and predictable, but it cannot prioritize urgent work.
- Priority scheduling: use a priority queue, and if starvation is a concern, combine priorities with aging so long-waiting tasks gradually become more important.
- Multi-level queues: another option is weighted fair scheduling, for example serving high-, medium-, and low-priority queues in a fixed ratio.
