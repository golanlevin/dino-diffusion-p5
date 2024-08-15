// Dino Diffusion: Bare-bones Diffusion in p5.js
// This project is /heavily/ based on "Dino Diffusion: Bare-bones Diffusion Models"
// by Ollin Boer Bohan: https://madebyoll.in/posts/dino_diffusion/
// Uses p5.js v.1.10.0 and ONNX Runtime Web v1.18.0: https://onnxruntime.ai/
// To train your own model, see https://github.com/madebyollin/dino-diffusion.
//
// To run the program: 
// Press RETURN to start the AI process.
// Press SPACE to clear the canvas and start over.
// Draw on the canvas with the mouse to provide input to the AI.
// The image of the canvas will be interpreted by Stable Diffusion.

let network, generator;
let patch, patchNoise, patchNoiseLevel, patchLowRes, patchCoords, patchGuidance, outputImage;
let inputGraphics; // This offscreen buffer will contain the user's input
let denoised64x64imagep5; // a p5.js version of the denoised image
let finalCompGraphics512x512; // the 512x512 p5 Graphics buffer that contains the final composite
const nSteps = 100; // How many steps to take in the diffusion process
let ratio = 8; // The ratio between the input and output image sizes
let iteration = 0; // The current iteration of the diffusion process

function setup() {
  createCanvas(512, 512);
  pixelDensity(1); 
  noSmooth();

  // The AI ingests the pixel data from inputGraphics, a 64x64 buffer.
  inputGraphics = createGraphics(64,64);
  inputGraphics.pixelDensity(1);
	inputGraphics.background(255);
  denoised64x64imagep5 = createImage(64,64);
  finalCompGraphics512x512 = createGraphics(512,512);
  finalCompGraphics512x512.pixelDensity(1);
  finalCompGraphics512x512.background(255);

  initializeModelVariables();
  loadNetwork();
  generateInputImage(); 
}

function initializeModelVariables() {
  const inC = 3; // number of channels
  const inW = 64; // width of input
  const inH = 64; // heigt of input
  const inNumVals = inC * inW * inH; 

  patch = new ort.Tensor("float32", new Float32Array(inNumVals), [1, inC, inH, inW]);
  patchNoise = new ort.Tensor("float32", new Float32Array(inNumVals), [1, inC, inH, inW]);
  patchNoiseLevel = new ort.Tensor("float32", new Float32Array(1), [1, 1, 1, 1]);
  patchLowRes = new ort.Tensor("float32", new Float32Array(inNumVals), [1, inC, inH, inW]);
  patchCoords = new ort.Tensor("float32", new Float32Array(inNumVals), [1, inC, inH, inW]);
  patchGuidance = new ort.Tensor("float32", new Float32Array(inNumVals), [1, inC, inH, inW]);
  outputImage = new ImageData(new Uint8ClampedArray(inH * inW * 4 * 8 * 8), inW*8, inH*8);
  outputImage.data.fill(255);

  outputGraphics = createGraphics(inW, inH);
  outputGraphics.background(255); 
}

async function loadNetwork() {
  try {
    network = await ort.InferenceSession.create('./network.onnx', { 
      executionProviders: ['webgl'] 
    });
    console.log('Network loaded.');
    generator = makeGenerator(
      network, patch, patchNoise, patchNoiseLevel, patchLowRes, 
      patchCoords, patchGuidance, outputImage);
    sendImageData();
  } catch (error) {
    console.error('Failed to load the network:', error);
  }
}

function generateInputImage(){
  // Make a simple input image (a random bezier) to provide to the AI.
  iteration = 0; 
  let W = 64; 
  let H = 64;
  let ax = W * random(0.10, 0.90); 
  let ay = H * random(0.05, 0.20); 
  let bx = W * random(0.30, 0.70); 
  let by = H * random(0.30, 0.40); 
  let cx = W * random(0.30, 0.70); 
  let cy = H * random(0.60, 0.70); 
  let dx = W * random(0.10, 0.90); 
  let dy = H * random(0.80, 0.95); 
  inputGraphics.strokeWeight(1); 
  inputGraphics.stroke(0);
  inputGraphics.bezier(ax,ay, bx,by, cx,cy, dx,dy);
}


function draw() {
  background(255);

  // Draw the final composite image
  tint(255); 
  image(finalCompGraphics512x512, 0,0); 

  // Draw the user's input into the offscreen buffer
  if (mouseIsPressed) {
    inputGraphics.stroke(0);
    inputGraphics.strokeWeight(1); 
    inputGraphics.line(
      pmouseX/ratio, pmouseY/ratio, 
      mouseX/ratio, mouseY/ratio);
  }

	// Draw the inputGraphics as a semi-transparent overlay.
  let inputGraphicsAlpha = map(iteration,0,638, 128,16); 
	tint(255,255,255, inputGraphicsAlpha); 
	image(inputGraphics,0,0,512,512);
}

function keyPressed() {
  if (key == ' ') {
    // If the user presses the space bar, we clear the inputGraphics buffer.
    background(255);
    inputGraphics.background(255);
    finalCompGraphics512x512.background(255);
    generateInputImage(); 
    sendImageData();

  } else if (keyCode == RETURN) {
    // If the user presses RETURN, transmit the inputGraphics buffer to the AI.
    sendImageData();
  }
}

function sendImageData() {
  // Here, we copy the data from the user's input into the patchGuidance tensor.
  finalCompGraphics512x512.background(255);
  inputGraphics.loadPixels(); 
  let imageData = inputGraphics.get();
  let nPixels = imageData.width * imageData.height;
  imageData.loadPixels();
  for (let c = 0; c < 3; c++) {
    for (let i = 0; i < nPixels; i++) {
      let srcVal = imageData.pixels[4 * i + c] / 255.0;
      let dstIndex = c * nPixels + i;
      patchGuidance.data[dstIndex] = srcVal;
    }
  }
  resample();
  regenerate();
}

// Taken from the original DinoDiffusion script.js file.
function resample() {
  for (let i = 0; i < patchNoise.data.length; i++) patchNoise.data[i] = Math.random();
}

// Adapted from the original DinoDiffusion script.js file.
function regenerate() {
  generator([nSteps, 
    Math.max(1, Math.floor(nSteps / 10)), 
    Math.max(1, Math.floor(nSteps / 20)), 
    Math.max(1, Math.floor(nSteps / 25))]);
}

// set up image generator
function makeGenerator(
  network, patch, patchNoise, patchNoiseLevel, patchLowRes, 
  patchCoords, patchGuidance, outputImage) {
  
  // single step of denoising
  async function generatePatch(tsk) {

    // useful stuff
    const [nlIn, nlOut] = [noiseLevelSchedule(1 - tsk.step / tsk.steps), noiseLevelSchedule(1 - (tsk.step + 1) / tsk.steps)];
    const [h, w] = [patch.dims[2], patch.dims[3]];
    
    // fill input information
    patchNoiseLevel.data[0] = nlIn;
    if (tsk.step == 0) {
      // fill working image
      for (let i = 0; i < patch.data.length; i++) {
        patch.data[i] = patchNoise.data[i];
      }
      // fill lowres image
      for (let i = 0; i < patchLowRes.data.length; i++) {
        patchLowRes.data[i] = (tsk.stage == 0) ? -1 : 
          outputImage.data[patchIndexToImageIndex(i, tsk, h, w, outputImage.height, outputImage.width)] / 255.0;
      }
      // fill coords
      for (let i = 0; i < patchCoords.data.length; i++) {
        const coords = patchIndexToImageCoords(i, tsk, h, w);
        patchCoords.data[i] = coords.c == 0 ? (coords.x / outputImage.width) : (coords.c == 1 ? (coords.y / outputImage.height) : 1);
      }
      // fill guidance
      if (tsk.stage > 0) {
        for (let i = 0; i < patchGuidance.data.length; i++) {
          patchGuidance.data[i] = 1;
        }
      }
    }

    // perform denoising step
    const denoised = (await network.run({
      "x": patch, 
      "noise_level": patchNoiseLevel, 
      "x_lowres": patchLowRes, 
      "x_coords": patchCoords, 
      "x_cond": patchGuidance})).denoised;

    // update working image
    const alpha = nlOut / nlIn;
    for (let i = 0; i < patch.data.length; i++) {
        patch.data[i] = alpha * patch.data[i] + (1 - alpha) * denoised.data[i];
    }

    // update rendering
    writePatchToImageWithFancyOverlapHandling(denoised, tsk, outputImage);
    renderResult(denoised);
    updateComposite(tsk); 
    iteration++;
  }

  function updateComposite(tsk){ 
    let hw = tsk.hwIn;
    let px = tsk.xIn;
    let py = tsk.yIn;
    finalCompGraphics512x512.image(denoised64x64imagep5, px,py,hw,hw);
    finalCompGraphics512x512.noFill(); 
  }

  function renderResult(denoised) {
    if (denoised && denoised.data && denoised.data.length > 0) {
      // Copy the denoised image into the denoised64x64imagep5 image.
      // Note: the color channels are separated in the denoised tensor.
      denoised64x64imagep5.loadPixels();
      let nDenoisedPixels = 64 * 64;
      let dstIndex = 0; 
      for (let y = 0; y < 64; y++) {
        for (let x = 0; x < 64; x++) {
          let loc =  y * 64 + x;
          let r = int(denoised.data[loc] * 255.0); loc+=nDenoisedPixels;
          let g = int(denoised.data[loc] * 255.0); loc+=nDenoisedPixels;
          let b = int(denoised.data[loc] * 255.0);
          denoised64x64imagep5.pixels[dstIndex++] = r; 
          denoised64x64imagep5.pixels[dstIndex++] = g; 
          denoised64x64imagep5.pixels[dstIndex++] = b; 
          denoised64x64imagep5.pixels[dstIndex++] = 255; 
        }
      }
      denoised64x64imagep5.updatePixels();
    }
  }

  let generationHandle = null;
  function generate(stepsPerResolution) {
      // plan out the work we'll need for this image generation
      let patchTaskQueue = [];
      for (let i = 0; i < stepsPerResolution.length; i++) {
          const steps = stepsPerResolution[i];
          // extra patch here (the + 1) so we get some patch overlap and no ugly edges
          const patchesPerSide = i == 0 ? 1 : ((1 << i) + 1);
          const patchSidePx = Math.round(patch.dims[2] / patchesPerSide) * Math.round(outputImage.width / patch.dims[2]);
          const tasksInStage = patchesPerSide * patchesPerSide * steps;
          for (let t = 0; t < tasksInStage; t++) {
              const [patchY, patchX, step] = [Math.floor(t / patchesPerSide / steps), Math.floor(t / steps) % patchesPerSide, t % steps];
              patchTaskQueue.push({
                  "stage": i, "step": step, "steps": steps,
                  "xIn": patchX * patchSidePx, "yIn": patchY * patchSidePx, "hwIn": Math.round(outputImage.width / (1 << i)),
                  "xOut": patchX * patchSidePx, "yOut": patchY * patchSidePx, "hwOut": patchSidePx,
                  "progress": (t + 1) / tasksInStage
              });
          }
      }
      // if we're already generating something, stop doing that
      if (generationHandle) window.clearTimeout(generationHandle);
      // start generating the new thing
      const minFrameTime_ms = 10;
      function generateNextPatchInQueue() {
          if (patchTaskQueue.length == 0) return renderResult({"done": true});
          generatePatch(patchTaskQueue.shift()).then(() => {
              generationHandle = window.setTimeout(generateNextPatchInQueue, minFrameTime_ms);
          });
      }
      generationHandle = window.setTimeout(generateNextPatchInQueue, minFrameTime_ms);
  }
  return generate;
}

/**
 * Taken from the original DinoDiffusion script.js file.
 * Splat a given patch into the given output image, based on patch task description.
 * Patch overlap is handled, so that overlapping patches are alpha-blended together.
 * (Dynamic input sizes make ORT very sad so it's easier to handle patches manually)
 * @param {Tensor} patch - RGB, CHW float32 patch to write
 * @param {Task} tsk - Task describing the location of the patch
 * @param {ImageData} outputImage - Image to write patch into.
 */
function writePatchToImageWithFancyOverlapHandling(patch, tsk, outputImage) {
  const [h, w] = [patch.dims[2], patch.dims[3]];
  const overlap = ((tsk.hwIn - tsk.hwOut) + 1);
  for (let y = tsk.yIn; y < tsk.yIn + tsk.hwIn && y < outputImage.height; y++) {
    for (let x = tsk.xIn; x < tsk.xIn + tsk.hwIn && x < outputImage.width; x++) {
      const py = constrain(Math.round((y - tsk.yIn + 0.5) / tsk.hwIn * h - 0.5), 0, h - 1);
      const px = constrain(Math.round((x - tsk.xIn + 0.5) / tsk.hwIn * w - 0.5), 0, w - 1);
      // alpha follows an overlap-length linear ramp on the top-left of each patch,
      // except for the patches on the top or left edges of the entire image.
      let alphaX = constrain((y - tsk.yIn) / overlap + (tsk.yIn == 0), 0, 1);
      let alphaY = constrain((x - tsk.xIn) / overlap + (tsk.xIn == 0), 0, 1)
      let alpha = Math.min(alphaX, alphaY);
      for (let c = 0; c < 3; c++) {
        let v = 255 * patch.data[c * (h * w) + py * w + px];
        v = alpha * v + (1 - alpha) * outputImage.data[(y * outputImage.width + x) * 4 + c];
        outputImage.data[(y * outputImage.width + x) * 4 + c] = v;
      }
    }
  }
}

/**
 * Taken from the original DinoDiffusion script.js file.
 * Convert a linear index in a CHW, RGB patch to {y, x, c} coordinates in output image
 * @param {int} i - index into patch
 * @param {Task} tsk - task used to create patch
 * @param {int} h - height of patch
 * @param {int} w - width of patch
 * @returns Floating-point, unclamped coordinates into output image
 */
function patchIndexToImageCoords(i, tsk, h, w) {
  const c = Math.floor(i / (h * w));
  const y = (Math.floor(i / w) % h + 0.5) / h * tsk.hwIn + tsk.yIn - 0.5;
  const x = (i % w + 0.5) / w * tsk.hwIn + tsk.xIn - 0.5;
  return {y, x, c};
}

/**
 * Taken from the original DinoDiffusion script.js file.
 * Convert a linear index in a CHW, RGB patch to nearest linear index in the larger HWC, RGBA output image.
 * @param {int} i - index into patch
 * @param {Task} tsk - task used to create patch
 * @param {int} h - height of patch
 * @param {int} w - width of patch
 * @param {int} oh - height of output image
 * @param {int} ow - width of output image
 * @returns Index into HWC RGBA output image
 */
function patchIndexToImageIndex(i, tsk, h, w, oh, ow) {
  const coords = patchIndexToImageCoords(i, tsk, h, w);
  const outY = constrain(Math.round(coords.y), 0, oh - 1);
  const outX = constrain(Math.round(coords.x), 0, ow - 1);
  return (outY * ow + outX) * 4 + coords.c;
}

/**
 * Taken from the original DinoDiffusion script.js file.
 * Implement the noise level schedule function.
 * Schedule that spends more time at high noise levels.
 * (Needed for reasonably vibrant results at small stepcounts)
 * @param {float} x - Noisiness under default linspace(1, 0) schedule.
 * @returns Adjusted noise level under the modified schedule.
 */
function noiseLevelSchedule(x) {
  const k = 0.2;
  return x * (1 + k) / (x + k);
}
