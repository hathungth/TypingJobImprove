import { useRef, useState } from "react";
import { Image as ImageIcon, ScanText, AudioLines, Upload, Download, WandSparkles, Scissors, Copy, Check, Info } from "lucide-react";
import { createWorker } from "tesseract.js";
import * as ort from "onnxruntime-web";

const tools = [
  { id: "image", label: "Image enhancer", icon: ImageIcon, desc: "Sharpen, adjust and upscale images." },
  { id: "ocr", label: "Image to text", icon: ScanText, desc: "Extract text from images with OCR." },
  { id: "audio", label: "Audio cutter", icon: AudioLines, desc: "Select a section and export it as WAV." },
];

function saveBlob(blob, name) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url; a.download = name; a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function ImageTool() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState("");
  const [mode, setMode] = useState("basic");
  const [sharpness, setSharpness] = useState(1.2);
  const [brightness, setBrightness] = useState(100);
  const [contrast, setContrast] = useState(100);
  const [busy, setBusy] = useState(false);
  const [status, setStatus] = useState("");
  const [model, setModel] = useState(null);
  const inputRef = useRef(null);
  const canvasRef = useRef(null);

  function loadImage(f) {
    if (!f || !f.type.startsWith("image/")) return;
    setFile(f); setPreview(URL.createObjectURL(f)); setStatus("");
  }

  function drawBasic() {
    if (!preview) return;
    const img = new Image();
    img.onload = () => {
      const canvas = canvasRef.current;
      const ctx = canvas.getContext("2d", { willReadFrequently: true });
      canvas.width = img.naturalWidth; canvas.height = img.naturalHeight;
      ctx.filter = `brightness(${brightness}%) contrast(${contrast}%)`;
      ctx.drawImage(img, 0, 0);
      ctx.filter = "none";
      if (sharpness > 0) {
        const src = ctx.getImageData(0, 0, canvas.width, canvas.height);
        const out = ctx.createImageData(canvas.width, canvas.height);
        const s = sharpness;
        const w = canvas.width, h = canvas.height, d = src.data, o = out.data;
        // 3x3 unsharp kernel; edge pixels are copied unchanged.
        for (let y = 1; y < h - 1; y++) for (let x = 1; x < w - 1; x++) {
          const i = (y * w + x) * 4;
          for (let c = 0; c < 3; c++) {
            const center = d[i + c] * (1 + 4 * s);
            const neighbors = d[i - 4 + c] + d[i + 4 + c] + d[i - w * 4 + c] + d[i + w * 4 + c];
            o[i + c] = Math.max(0, Math.min(255, center - s * neighbors));
          }
          o[i + 3] = d[i + 3];
        }
        // Preserve untouched boundary pixels.
        for (let x = 0; x < w; x++) for (const y of [0, h - 1]) {
          const i = (y * w + x) * 4; o.set(d.slice(i, i + 4), i);
        }
        for (let y = 0; y < h; y++) for (const x of [0, w - 1]) {
          const i = (y * w + x) * 4; o.set(d.slice(i, i + 4), i);
        }
        ctx.putImageData(out, 0, 0);
      }
      setStatus("Preview updated. Download to save the processed image.");
    };
    img.src = preview;
  }

  async function runAI() {
    if (!preview || !model) { setStatus("Choose an ONNX super-resolution model first."); return; }
    setBusy(true); setStatus("Running model…");
    try {
      // Generic RGB NCHW super-resolution model contract:
      // input tensor [1, 3, H, W], float32 values normalized to [0,1].
      // Output tensor [1, 3, H*scale, W*scale], float32 values [0,1].
      const session = await ort.InferenceSession.create(URL.createObjectURL(model), { executionProviders: ["wasm"] });
      const img = new Image();
      img.src = preview;
      await img.decode();
      const w = img.naturalWidth, h = img.naturalHeight;
      const c = document.createElement("canvas"); c.width = w; c.height = h;
      const cx = c.getContext("2d"); cx.drawImage(img, 0, 0);
      const pixels = cx.getImageData(0, 0, w, h).data;
      const input = new Float32Array(3 * w * h);
      for (let i = 0; i < w * h; i++) {
        input[i] = pixels[i * 4] / 255;
        input[w * h + i] = pixels[i * 4 + 1] / 255;
        input[2 * w * h + i] = pixels[i * 4 + 2] / 255;
      }
      const inputName = session.inputNames[0];
      const result = await session.run({ [inputName]: new ort.Tensor("float32", input, [1, 3, h, w]) });
      const output = result[session.outputNames[0]];
      const dims = output.dims;
      const oh = dims[2], ow = dims[3], data = output.data;
      const outCanvas = document.createElement("canvas"); outCanvas.width = ow; outCanvas.height = oh;
      const outCtx = outCanvas.getContext("2d");
      const outImage = outCtx.createImageData(ow, oh);
      for (let i = 0; i < ow * oh; i++) {
        outImage.data[i * 4] = Math.max(0, Math.min(255, data[i] * 255));
        outImage.data[i * 4 + 1] = Math.max(0, Math.min(255, data[ow * oh + i] * 255));
        outImage.data[i * 4 + 2] = Math.max(0, Math.min(255, data[2 * ow * oh + i] * 255));
        outImage.data[i * 4 + 3] = 255;
      }
      outCtx.putImageData(outImage, 0, 0);
      const dataUrl = outCanvas.toDataURL("image/png");
      setPreview(dataUrl); setStatus(`AI output ready (${ow} × ${oh}).`);
      session.release();
    } catch (e) {
      setStatus(`Could not run this model: ${e.message}. Check that it accepts NCHW RGB float32 input and returns NCHW RGB output.`);
    } finally { setBusy(false); }
  }

  function downloadImage() {
    const canvas = canvasRef.current;
    if (!canvas.width) { setStatus("Click Apply basic enhancement first."); return; }
    canvas.toBlob(blob => blob && saveBlob(blob, "enhanced-image.png"), "image/png");
  }

  return <section className="tool-panel">
    <div className="panel-head"><div><span className="eyebrow">01 / IMAGE</span><h2>Image enhancer</h2><p>Improve clarity with classic image processing or an ONNX super-resolution model.</p></div></div>
    <div className="mode-switch">
      <button className={mode === "basic" ? "selected" : ""} onClick={() => setMode("basic")}>Basic enhancement</button>
      <button className={mode === "ai" ? "selected" : ""} onClick={() => setMode("ai")}>AI enhancement</button>
    </div>
    <div className="upload-area" onClick={() => inputRef.current?.click()}>
      <Upload size={24}/><strong>{file ? file.name : "Choose an image"}</strong><span>PNG, JPG or WEBP · processed in your browser</span>
      <input ref={inputRef} type="file" accept="image/*" hidden onChange={e => loadImage(e.target.files?.[0])}/>
    </div>
    {preview && <div className="preview-grid">
      <div className="preview-box"><span className="eyebrow">IMAGE PREVIEW</span><img src={preview} alt="Selected image preview"/></div>
      <div className="controls">
        {mode === "basic" ? <>
          <label>Sharpness <b>{sharpness.toFixed(1)}</b><input type="range" min="0" max="2.5" step=".1" value={sharpness} onChange={e => setSharpness(+e.target.value)}/></label>
          <label>Brightness <b>{brightness}%</b><input type="range" min="50" max="150" value={brightness} onChange={e => setBrightness(+e.target.value)}/></label>
          <label>Contrast <b>{contrast}%</b><input type="range" min="50" max="160" value={contrast} onChange={e => setContrast(+e.target.value)}/></label>
          <button className="primary" onClick={drawBasic}><WandSparkles size={17}/> Apply basic enhancement</button>
          <button className="secondary" onClick={downloadImage}><Download size={17}/> Download PNG</button>
        </> : <>
          <label className="file-label">ONNX model file<input type="file" accept=".onnx" onChange={e => setModel(e.target.files?.[0] || null)}/></label>
          <div className="notice"><Info size={16}/> This AI option requires a compatible ONNX model. See README for the expected model input/output format.</div>
          <button className="primary" disabled={busy || !model} onClick={runAI}><WandSparkles size={17}/> {busy ? "Processing…" : "Run AI enhancement"}</button>
          <p className="hint">The generic runner expects RGB NCHW float32 input and output. Some Real-ESRGAN exports require model-specific preprocessing or tiling and may not work without adaptation.</p>
        </>}
        {status && <p className="status">{status}</p>}
      </div>
    </div>}
    <canvas ref={canvasRef} className="hidden-canvas"/>
  </section>;
}

function OCRTool() {
  const [image, setImage] = useState("");
  const [text, setText] = useState("");
  const [lang, setLang] = useState("eng+vie");
  const [busy, setBusy] = useState(false);
  const [progress, setProgress] = useState("");
  const inputRef = useRef(null);
  async function recognize() {
    if (!image) return;
    setBusy(true); setProgress("Loading OCR engine…"); setText("");
    let worker;
    try {
      worker = await createWorker(lang, 1, { logger: m => {
        if (m.status) setProgress(`${m.status}${m.progress ? ` — ${Math.round(m.progress * 100)}%` : ""}`);
      }});
      const result = await worker.recognize(image);
      setText(result.data.text.trim());
      setProgress("Recognition complete.");
    } catch (e) { setProgress(`OCR error: ${e.message}`); }
    finally { if (worker) await worker.terminate(); setBusy(false); }
  }
  return <section className="tool-panel">
    <div className="panel-head"><div><span className="eyebrow">02 / OCR</span><h2>Image to text</h2><p>Extract editable text from a photo, screenshot or scanned page.</p></div></div>
    <div className="upload-area" onClick={() => inputRef.current?.click()}><Upload size={24}/><strong>{image ? "Image selected" : "Choose an image to scan"}</strong><span>Supports Vietnamese and English</span><input ref={inputRef} type="file" accept="image/*" hidden onChange={e => { const f=e.target.files?.[0]; if(f) setImage(URL.createObjectURL(f)); }}/></div>
    {image && <div className="ocr-layout"><img className="ocr-image" src={image} alt="Image for OCR"/><div className="controls"><label>Recognition language<select value={lang} onChange={e => setLang(e.target.value)}><option value="eng+vie">English + Vietnamese</option><option value="eng">English</option><option value="vie">Vietnamese</option></select></label><button className="primary" disabled={busy} onClick={recognize}><ScanText size={17}/>{busy ? "Recognizing…" : "Extract text"}</button><p className="status">{progress}</p></div></div>}
    {text && <div className="result-area"><div className="result-title"><strong>Extracted text</strong><button className="secondary small" onClick={() => navigator.clipboard.writeText(text)}><Copy size={15}/> Copy text</button></div><textarea value={text} onChange={e => setText(e.target.value)} rows={10}/><button className="secondary" onClick={() => saveBlob(new Blob([text], {type:"text/plain;charset=utf-8"}), "extracted-text.txt")}><Download size={16}/> Download TXT</button></div>}
  </section>;
}

function AudioTool() {
  const [file, setFile] = useState(null);
  const [url, setUrl] = useState("");
  const [duration, setDuration] = useState(0);
  const [start, setStart] = useState(0);
  const [end, setEnd] = useState(0);
  const [busy, setBusy] = useState(false);
  const inputRef = useRef(null);
  function load(f) {
    if (!f || !f.type.startsWith("audio/")) return;
    setFile(f); setUrl(URL.createObjectURL(f)); setStart(0); setEnd(0);
  }
  function onMeta(e) { const d=e.currentTarget.duration; setDuration(d); setEnd(d); }
  function fmt(n) { if (!Number.isFinite(n)) return "0:00"; return `${Math.floor(n/60)}:${String(Math.floor(n%60)).padStart(2,"0")}`; }
  async function exportWav() {
    if (!file || end <= start) return;
    setBusy(true);
    try {
      const buffer = await file.arrayBuffer();
      const AudioCtx = window.AudioContext || window.webkitAudioContext;
      const ctx = new AudioCtx();
      const decoded = await ctx.decodeAudioData(buffer);
      const rate = decoded.sampleRate, from = Math.floor(start * rate), to = Math.min(decoded.length, Math.floor(end * rate));
      const length = to - from, channels = decoded.numberOfChannels;
      const wavBuffer = new ArrayBuffer(44 + length * channels * 2);
      const view = new DataView(wavBuffer);
      const write = (offset, str) => [...str].forEach((ch,i)=>view.setUint8(offset+i,ch.charCodeAt(0)));
      write(0,"RIFF"); view.setUint32(4,36+length*channels*2,true); write(8,"WAVE"); write(12,"fmt ");
      view.setUint32(16,16,true); view.setUint16(20,1,true); view.setUint16(22,channels,true);
      view.setUint32(24,rate,true); view.setUint32(28,rate*channels*2,true); view.setUint16(32,channels*2,true);
      view.setUint16(34,16,true); write(36,"data"); view.setUint32(40,length*channels*2,true);
      let offset=44;
      for(let i=0;i<length;i++) for(let ch=0;ch<channels;ch++) {
        const sample=Math.max(-1,Math.min(1,decoded.getChannelData(ch)[from+i]));
        view.setInt16(offset, sample < 0 ? sample*0x8000 : sample*0x7fff, true); offset+=2;
      }
      saveBlob(new Blob([wavBuffer],{type:"audio/wav"}),"trimmed-audio.wav");
      await ctx.close();
    } catch(e) { alert(`Could not decode this audio file: ${e.message}`); }
    finally { setBusy(false); }
  }
  return <section className="tool-panel">
    <div className="panel-head"><div><span className="eyebrow">03 / AUDIO</span><h2>Audio cutter</h2><p>Choose a start and end time, then export the selected segment as WAV.</p></div></div>
    <div className="upload-area" onClick={() => inputRef.current?.click()}><Upload size={24}/><strong>{file ? file.name : "Choose an audio file"}</strong><span>MP3, WAV, M4A and browser-supported audio formats</span><input ref={inputRef} type="file" accept="audio/*" hidden onChange={e => load(e.target.files?.[0])}/></div>
    {url && <div className="audio-editor"><audio controls src={url} onLoadedMetadata={onMeta}/><div className="time-row"><label>Start time <strong>{fmt(start)}</strong><input type="range" min="0" max={duration || 1} step=".1" value={Math.min(start,duration)} onChange={e=>setStart(Math.min(+e.target.value,end))}/></label><label>End time <strong>{fmt(end)}</strong><input type="range" min="0" max={duration || 1} step=".1" value={Math.min(end,duration)} onChange={e=>setEnd(Math.max(+e.target.value,start))}/></label></div><p className="hint">Total duration: {fmt(duration)} · Selected: {fmt(Math.max(0,end-start))}</p><button className="primary" disabled={busy || end<=start} onClick={exportWav}><Scissors size={17}/>{busy ? "Exporting…" : "Cut and download WAV"}</button></div>}
  </section>;
}

export default function App() {
  const [active, setActive] = useState("image");
  const current = tools.find(t => t.id === active);
  return <main className="app-shell">
    <header className="topbar"><a className="brand" href="#"><span className="brand-mark">T</span><span>TypingJobImprove<small>MEDIA TOOLKIT</small></span></a><span className="privacy"><span/> Runs locally in your browser</span></header>
    <section className="hero"><span className="eyebrow">YOUR EVERYDAY MEDIA WORKSPACE</span><h1>Make your files<br/><em>work better.</em></h1><p>Enhance images, extract text and trim audio — with simple tools that run directly in your browser.</p></section>
    <nav className="tool-nav" aria-label="Tools">{tools.map(t => { const Icon=t.icon; return <button key={t.id} className={active===t.id?"active":""} onClick={()=>setActive(t.id)}><Icon size={19}/><span>{t.label}</span></button>; })}</nav>
    <div className="tool-description"><span>{current.desc}</span><span>NO ACCOUNT · NO UPLOADS</span></div>
    {active==="image" && <ImageTool/>}
    {active==="ocr" && <OCRTool/>}
    {active==="audio" && <AudioTool/>}
    <footer><span>TypingJobImprove © {new Date().getFullYear()}</span><span>Files stay on your device. Nothing is sent to a server.</span></footer>
  </main>;
}