from tqdm import tqdm
import asyncio
import aiohttp
import nest_asyncio

nest_asyncio.apply()


class TaskQueueWorker:
    def __init__(self, worker_method, result_callback=None, concurrency=200):
        self.worker = worker_method
        self.result_callback = result_callback
        self.concurrency = concurrency

    async def process(self, records: list):
        record_iter = iter(records)
        pending_tasks = set()

        with tqdm(total=len(records), desc="Processing records... ") as pbar:
            while True:
                while len(pending_tasks) < self.concurrency:
                    try:
                        record = next(record_iter)
                        task = asyncio.create_task(self.worker(record))
                        pending_tasks.add(task)
                    except StopIteration:
                        break
                if not pending_tasks:
                    break

                done, pending_tasks = await asyncio.wait(
                    pending_tasks, return_when=asyncio.FIRST_COMPLETED
                )

                for completed_task in done:
                    pbar.update(1)
                    if self.result_callback:
                        completed_result = completed_task.result()
                        self.result_callback(completed_result)
