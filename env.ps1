# Dot-source this before running any uv/bytx command:
#   . .\env.ps1
#
# Points every tool, cache, and model store at project-local folders on this
# drive instead of C:\ (uv, ffmpeg, Deno binaries; uv's Python + package
# cache; the HuggingFace download cache; Ollama's model store). Nothing here
# touches your system or user environment - it only affects this shell.

$root = $PSScriptRoot

$env:PATH = "$root\tools\uv;$root\tools\ffmpeg\bin;$root\tools\deno;$env:PATH"
$env:UV_CACHE_DIR = "$root\tools\uv-cache"
$env:UV_PYTHON_INSTALL_DIR = "$root\tools\uv-python"
$env:HF_HOME = "$root\tools\hf-cache"
$env:OLLAMA_MODELS = "$root\ollama-models"

Write-Output "bytx environment loaded (uv, ffmpeg, deno on PATH; caches/models redirected to $root)"
