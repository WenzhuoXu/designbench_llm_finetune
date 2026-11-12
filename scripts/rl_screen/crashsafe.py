"""A process pool that survives a worker dying (a simulator that segfaults or is killed for memory).

multiprocessing.Pool waits for ever on a dead worker's task, and a ProcessPoolExecutor is broken for
every task left once one worker dies. run_jobs runs the jobs in one pool; if a worker dies, the jobs
without a result are run again with one fresh process each, so the job that kills its process is
reported as failed and every other job still gets its result.
"""
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool

DIED = "worker process died"


def run_jobs(fn, jobs, workers, initializer=None, initargs=()):
    """Yield (index, result, error) once for every job, in completion order. error is None, the
    repr of the exception fn raised, or DIED."""
    ctx = mp.get_context("spawn")
    pending = set(range(len(jobs)))
    broken = False
    ex = ProcessPoolExecutor(max(1, workers), mp_context=ctx, initializer=initializer, initargs=initargs)
    futs = {ex.submit(fn, jobs[i]): i for i in sorted(pending)}
    try:
        for fu in as_completed(futs):
            i = futs[fu]
            try:
                res = fu.result()
            except BrokenProcessPool:
                broken = True
                break
            except Exception as e:                           # noqa: BLE001
                pending.discard(i)
                yield i, None, repr(e)[:400]
                continue
            pending.discard(i)
            yield i, res, None
        if broken:                                           # results that were ready before the pool broke
            for fu, i in futs.items():
                if i in pending and fu.done() and not fu.cancelled() and fu.exception() is None:
                    pending.discard(i)
                    yield i, fu.result(), None
    finally:
        ex.shutdown(wait=False, cancel_futures=True)
    if not pending:
        return

    def alone(i):
        one = ProcessPoolExecutor(1, mp_context=ctx, initializer=initializer, initargs=initargs)
        try:
            return i, one.submit(fn, jobs[i]).result(), None
        except BrokenProcessPool:
            return i, None, DIED
        except Exception as e:                               # noqa: BLE001
            return i, None, repr(e)[:400]
        finally:
            one.shutdown(wait=False, cancel_futures=True)

    with ThreadPoolExecutor(max(1, workers)) as tp:
        for out in tp.map(alone, sorted(pending)):
            yield out
