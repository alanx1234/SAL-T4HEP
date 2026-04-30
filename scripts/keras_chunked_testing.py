import logging
import os

import numpy as np


def resolve_test_chunk_size(batch_size):
    chunk_size = int(os.environ.get("TEST_CHUNK_SIZE", max(batch_size * 64, batch_size)))
    return max(chunk_size, batch_size)


def predict_in_chunks(model, n_events, batch_size, prepare_chunk):
    chunk_size = resolve_test_chunk_size(batch_size)
    logging.info("Running chunked TEST prediction with chunk_size=%d", chunk_size)

    preds = None
    for start in range(0, n_events, chunk_size):
        end = min(start + chunk_size, n_events)
        x_chunk = prepare_chunk(start, end)
        chunk_preds = model.predict(x_chunk, batch_size=batch_size, verbose=0)
        chunk_preds = np.asarray(chunk_preds)

        if preds is None:
            preds = np.empty((n_events,) + chunk_preds.shape[1:], dtype=chunk_preds.dtype)
        preds[start:end] = chunk_preds

    return preds
