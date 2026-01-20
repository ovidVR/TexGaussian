import os
import shutil
import tempfile

import typer
from fastapi import FastAPI, File, Form, UploadFile, APIRouter
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask

from core.options import config_defaults
from texture import Converter, str2bool

app = FastAPI()
router = APIRouter(prefix="/meshAPI/hunyuan")
cli = typer.Typer()


# Load the model once during startup
@app.on_event("startup")
async def load_model():
    global converter

    config = config_defaults["objaverse"]
    config.use_checkpoint = str2bool(config.use_checkpoint)
    config.use_material = str2bool(config.use_material)
    config.save_image = False  # Disable saving intermediate images
    config.gaussian_loss = str2bool(config.gaussian_loss)
    config.use_local_pretrained_ckpt = str2bool(config.use_local_pretrained_ckpt)
    config.ckpt_path = os.getenv(
        "MODEL_CKPT_PATH", config.ckpt_path
    )  # Use environment variable or default
    converter = Converter(config).cuda()
    converter.load_ckpt(config.ckpt_path)


@router.post("/texturize")
async def texturize(mesh: UploadFile = File(...), prompt: str = Form(...)):
    # Create a temporary directory for processing
    temp_dir = tempfile.TemporaryDirectory().name
    # Save the uploaded mesh file
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    mesh_path = os.path.join(temp_dir, mesh.filename)
    with open(mesh_path, "wb") as f:
        shutil.copyfileobj(mesh.file, f)

    # Load the mesh and process it
    converter.opt.text_prompt = prompt
    converter.load_mesh(mesh_path)
    converter.fit_mesh_uv(iters=1000)

    # Export the textured mesh
    output_dir = os.path.join(temp_dir, "output")
    converter.export_mesh(output_dir)

    # Create a zip archive of the output folder
    zip_path = os.path.join(temp_dir, "textured_mesh.zip")
    shutil.make_archive(zip_path.replace(".zip", ""), "zip", output_dir)

    return FileResponse(
        zip_path,
        media_type="application/zip",
        filename="textured_mesh.zip",
        background=BackgroundTask(shutil.rmtree, temp_dir),
    )


@cli.command()
def run(ckpt_path: str = typer.Option(..., help="Path to the model checkpoint")):
    """
    Start the FastAPI application with the specified model checkpoint.
    """
    os.environ["MODEL_CKPT_PATH"] = ckpt_path  # Pass checkpoint path via environment variable
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)

app.include_router(router)

if __name__ == "__main__":
    cli()
