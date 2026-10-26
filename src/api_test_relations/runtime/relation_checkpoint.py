"""Process-local periodic durability; no test context is assigned to the writer."""
import json
import os
import threading
import time
from pathlib import Path


class Checkpoints:
    def __init__(self, plugin, interval=1.0):
        self.plugin = plugin
        self.pid = os.getpid()
        self.interval = interval
        self.stop = threading.Event()
        self.done = threading.Event()
        self.folder = Path(os.environ['API_RELATION_SHARDS'])
        self.state = self.folder / 'checkpoints' / (str(self.pid) + '.json')
        self.state.parent.mkdir(exist_ok=True)
        self.publish('starting')
        # The collector's saved primitive bypasses context propagation and guard
        # installation for this internal writer only, never for workload threads.
        plugin.collector.old_start(self.run, ())

    def publish(self, state, **extra):
        value = dict(pid=self.pid, state=state, at=time.time(), **extra)
        temp = self.state.with_suffix('.tmp')
        temp.write_text(json.dumps(value))
        temp.replace(self.state)

    def run(self):
        try:
            while os.getpid() == self.pid and not self.stop.wait(self.interval):
                try:
                    request = self.folder / 'checkpoint-request.json'
                    barrier = json.loads(request.read_text()) if request.exists() else None
                    # Freeze new record callbacks after committing everything
                    # accepted before the timeout barrier. Parent/child owners
                    # remain on the records themselves, never on this thread.
                    with self.plugin.lock:
                        self.plugin.flush()
                        self.publish('quiesced' if barrier else 'committed',
                                     request=barrier.get('token') if barrier else None)
                        if barrier:
                            self.stop.wait()
                            break
                except Exception as exc:
                    # flush keeps its pending batch on rollback. Expose the
                    # failure independently of SQLite, then retry next interval.
                    error=type(exc).__name__ + ': ' + str(exc)[:500]
                    with self.plugin.lock:
                        for node,phase,_ in self.plugin.pending:
                            self.plugin.gaps.add((node,phase,'checkpoint_write_failed:'+error))
                    try:self.publish('error', error=error)
                    except OSError:pass
        finally:
            self.done.set()

    def close(self):
        if os.getpid() != self.pid:return
        self.stop.set()
        if not self.done.wait(30):
            raise RuntimeError('Checkpoint writer did not stop; evidence must remain incomplete')
