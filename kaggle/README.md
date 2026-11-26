# Running the GPU stages on Kaggle

The pipeline writes a bundle of pending jobs. This worker runs them on a free Kaggle GPU (2× T4, about 30 GPU hours a week, phone-verified account).

1. On your machine: `make shots` (or `make music`). It prints the bundle path, for example `outputs/jobs/shots/bundle/`.
2. Zip the bundle and upload it as a private Kaggle dataset. Upload `kaggle/gpu_worker.py` too, as part of the same dataset or as a second one.
3. Create a notebook with **GPU T4 ×2** and internet on, add the dataset(s), and run:

   ```
   !pip install -q -U diffusers transformers accelerate ftfy imageio imageio-ffmpeg
   !python /kaggle/input/<dataset>/gpu_worker.py /kaggle/input/<dataset>/bundle /kaggle/working/results --gpus 2
   ```

4. Download `/kaggle/working/results/` and import it on your machine:

   ```
   uv run python -m pipeline.gpujobs import shots ~/Downloads/results
   make film
   ```

If the session times out, run the same cell again. Finished jobs are skipped.

Draft first. Set `models.video.params` in the storyboard to fewer steps (for example 20) for drafts. Record the latency per clip from `results/runs.jsonl` in COSTS.md, then finish only the chosen shots at full settings.
