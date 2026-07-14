# ComfyUI workflows

This directory is mounted into the ComfyUI container at `/app/user/default/workflows`.
Workflow JSON files saved from the ComfyUI web interface persist here.

## Getting starter workflows

ComfyUI v0.27.1 ships official workflow templates for every model in the stack. In the
web interface at http://localhost:8188, open **Workflow > Browse Templates** and select:

| Template | Model | Purpose |
|----------|-------|---------|
| FLUX.2 klein | `FLUX.2-klein-4B` | Fast image iteration (4-step generation) |
| FLUX.2 klein base | `FLUX.2-klein-base-9b-fp8` | Higher-quality images |
| Wan 2.2 T2V / I2V | Wan 2.2 14B FP8 | Photorealistic text/image-to-video |
| LTX-2 | `LTX-2.3-fp8` | Video with synchronized audio |

After loading a template, point each loader node at the matching file under
`/app/models` and save the workflow; it is then versioned in this directory.

Workflows committed to the repository must reference model paths relative to
`/app/models` so they work on any checkout.
