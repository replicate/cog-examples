# /// script
# requires-python = ">=3.11"
# dependencies = ["huggingface_hub"]
# ///
"""Download Z-Image-Turbo weights from HuggingFace into weights/ directory."""

from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Tongyi-MAI/Z-Image-Turbo",
    local_dir="weights",
    ignore_patterns=["assets/*", "README.md", ".gitattributes"],
)
