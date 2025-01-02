# Copyright (c) Meta Platforms, Inc. and affiliates.
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import torch
import typer
from rich.progress import Progress, TextColumn

from bytelatent.data.iterators.arrow_iterator import ArrowFileIterator


def main(
    input_file: str,
    output_file: str,
    patching_device: str = "cuda",
    log_step: int = 10_000,
    entropy_model_checkpoint_dir: str = "entropy_checkpoint_dir",
    dry_run: bool = False,
):
    # TODO: Modify this to work with the new code
    raise NotImplementedError()
    iterator = ArrowFileIterator(
        file_path=input_file,
        worker_id=0,
        num_workers=1,
    )
    tokenization_mode = "bytes"
    print(f"Preprocessing entropies, input: {input_file}, output: {output_file}")
    print("Loading entropy model", entropy_model_checkpoint_dir)
    if dry_run:
        return
    entropy_model = load_entropy_model(
        entropy_model_checkpoint_dir, device=patching_device
    )
    entropy_model, _ = to_device(entropy_model, patching_device)
    print("Creating patcher")
    patching_batch_size = 32
    print("Creating tokenizer")
    tokenizer = Tokenizer(
        model_path="/home/artidoro/tokenizers/llama_v2.tokenizer.model",
        tokenization_mode=tokenization_mode,
        # BYTE_UNITS
        vocab_size_unit_1=256,
        bos=True,
        eos=True,
        bpe_delim=False,
        # This isn't used, just stores a reference for other calls we don't use
        patcher=None,
    )
    step = 0
    print("starting")
    start_time = time.time()
    patch_time = 0
    entropy_field = pa.field("entropies", pa.list_(pa.float16()), nullable=False)
    sample_id_field = pa.field("sample_id", pa.string(), nullable=False)
    text_field = pa.field("text", pa.string(), nullable=False)
    schema = pa.schema([sample_id_field, text_field, entropy_field])
    arrow_batch_size = 1_000

    try:
        with pa.OSFile(output_file, "wb") as sink:
            with pa.ipc.new_file(sink, schema) as writer:
                id_buffer = []
                entropies_buffer = []
                text_buffer = []
                with Progress(
                    *Progress.get_default_columns(),
                    TextColumn("Completed: {task.completed}"),
                ) as progress:
                    task = progress.add_task(
                        "[green]Calculating entropies...", total=None
                    )
                    for doc in iterator:
                        sample_id = get_id_from_doc(doc)

                        if "text" in doc:
                            text = doc["text"]
                        elif "content" in doc:
                            text = doc["content"]
                        else:
                            raise ValueError(
                                f"Could not find a text key from: {doc.keys()}"
                            )
                        tokens = torch.tensor(tokenizer.encode(text))
                        patch_start = time.time()
                        scores = calculate_entropies(
                            tokens,
                            entropy_model,
                            patching_batch_size,
                            patching_device,
                        )
                        entropies_buffer.append(
                            np.array(scores.tolist(), dtype=np.float16)
                        )
                        id_buffer.append(sample_id)
                        text_buffer.append(text)
                        if len(entropies_buffer) == arrow_batch_size:
                            batch = pa.record_batch(
                                {
                                    "entropies": entropies_buffer,
                                    "sample_id": id_buffer,
                                    "text": text_buffer,
                                },
                                schema,
                            )
                            writer.write(batch)
                            entropies_buffer = []
                            id_buffer = []
                            text_buffer = []
                        patch_time += time.time() - patch_start
                        step += 1
                        if step % log_step == 0:
                            print("Completed steps:", step)
                        progress.update(task, advance=1)
                    if len(entropies_buffer) > 0:
                        # Write last things
                        batch = pa.record_batch(
                            {
                                "entropies": entropies_buffer,
                                "sample_id": id_buffer,
                                "text": text_buffer,
                            },
                            schema,
                        )
                        writer.write(batch)
                        entropies_buffer = []
                        id_buffer = []
                        text_buffer = []
        Path(f"{output_file}.complete").touch()
    except:
        Path(output_file).unlink(missing_ok=True)
        raise
    elapsed = time.time() - start_time
    print("steps", step)
    print("done in:", elapsed)


if __name__ == "__main__":
    typer.run(main)
