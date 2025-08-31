import asyncio
from typing import Any, Dict
import logging
import functools
from concurrent.futures import ThreadPoolExecutor

from contanos.utils.format_results import format_to_list

class BaseWorker:
    def __init__(self, worker_id: int, device: str,
                 model_config: Dict,
                 input_interface, 
                 output_interface):
        self.worker_id = worker_id
        self.input_interface = input_interface
        self.output_interface = output_interface
        self.model_config = model_config
        self.device = device
        self._model_init()
        self._executor = ThreadPoolExecutor(max_workers=1)

    def _model_init(self):
        pass

    def _predict(self, input: Any, metadata: Any) -> Any:
        return None

    def _format_results(self, results: Any, metadata: dict = None) -> dict:
        """Format the results for output."""
        output = format_to_list(results)
        
        final_metadata = {}
        if isinstance(metadata, list):
            for it in metadata:
                final_metadata.update(it)
            metadata = final_metadata

        output.update(metadata)
        return output

    async def run(self):
        loop = asyncio.get_running_loop()

        """Run the worker, reading from input and writing to output."""
        # logging.info(f"Worker {self.worker_id} started on {self.device}")
        while True:
            try:
                # logging.debug(f"Worker {self.worker_id} waiting for input data...")
                input, metadata = await self.input_interface.read_data()
                # logging.debug(f"Worker {self.worker_id} received input: type={type(input)}, metadata keys={list(metadata.keys()) if metadata else 'None'}")

                results = await loop.run_in_executor(
                    self._executor,
                    functools.partial(self._predict, input, metadata)
                )

                if results is not None:
                    output = self._format_results(results, metadata)
                    await self.output_interface.write_data(output)
                    # self.results.append(results)  # Store for verification
                    # logging.debug(f"Worker {self.worker_id} on {self.device} processed input -> output")
            except asyncio.TimeoutError:
                logging.info(f"Worker {self.worker_id} on {self.device} timed out, stopping")
                break
            except Exception as e:
                logging.error(f"Worker {self.worker_id} on {self.device} error: {e}")
                import traceback
                # logging.error(f"Traceback: {traceback.format_exc()}")
                break
        logging.info(f"Worker {self.worker_id} on {self.device} finished")