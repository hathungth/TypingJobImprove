# TypingJobImprove

A browser-based toolkit with three tools:

1. **Image enhancer** — basic sharpening, brightness and contrast; optional ONNX AI super-resolution runner.
2. **Image to text** — OCR using Tesseract.js, with English and Vietnamese language support.
3. **Audio cutter** — select start/end times and export the selected audio segment as WAV.

All normal processing happens in the browser. The site does not upload selected files to a server.

## Run locally

Install Node.js 20 or newer, then:

```bash
npm install
npm run dev
```

Open the local URL printed by Vite.

## Publish to GitHub Pages

1. Create a new **public** repository named `TypingJobImprove` on your GitHub account.
2. Upload all files and folders from this project to the repository root (not the ZIP file itself).
3. Ensure the default branch is named `main`.
4. In the repository, open **Settings → Pages** and set the build/deployment source to **GitHub Actions**.
5. The workflow at `.github/workflows/deploy.yml` will build and deploy the site after a push to `main`.
6. Your site URL will be `https://YOUR-GITHUB-USERNAME.github.io/TypingJobImprove/`.

If you choose a different repository name, change `base` in `vite.config.js` to `"/YOUR-REPOSITORY-NAME/"` before uploading.

## Image enhancement notes

### Basic mode
Uses browser Canvas APIs for brightness, contrast and a 3×3 sharpening kernel. This is a conventional image filter, not generative restoration.

### AI mode
The AI runner is an optional generic ONNX Runtime Web implementation. It expects a model with:

- Input: float32 tensor `[1, 3, H, W]` (RGB, NCHW, values normalized to `[0, 1]`).
- Output: float32 tensor `[1, 3, H_out, W_out]` (RGB, NCHW, values normalized to `[0, 1]`).

Choose a compatible `.onnx` file in the AI mode. **Not every Real-ESRGAN ONNX export follows this interface.** Some models require different normalization, input/output names, output clamping, padding, or tiled inference to avoid browser memory limits. If the model does not match this contract, the AI run will fail and the model-specific adapter must be implemented.

No model weights are bundled in this project. This keeps the repository small and avoids redistributing third-party model files without checking their license. For a production-quality AI enhancement feature, choose a model with a clear license and test its exact ONNX input/output contract.

## OCR notes

The first OCR run downloads Tesseract worker and language data from the Tesseract.js CDN. The browser needs an internet connection for this initial download. OCR itself runs locally in the browser after the assets load.

## Audio notes

The cutter uses the browser's Web Audio API to decode the source and exports WAV. Browser codec support varies by file type. Very long or high-resolution audio can require substantial memory.

## Privacy and limitations

- The app has no backend and does not store uploaded files.
- OCR language data is fetched from the configured CDN on first use.
- The ONNX model is selected locally by the user and runs in the browser.
- This is a starter implementation. Test it with your target browsers and representative files before relying on it for critical work.
