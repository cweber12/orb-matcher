// main.js
// Main script for ORB feature detection and matching tool

import { ORBModule } from './orb_module.js?v=20251104';
import { VideoFrameExtractor } from './video_frame_extractor.js?v=20251104';
import { setupCropBox } from './setup_crop_box.js?v=20251104';
import { loadImg, matFromImageEl, cropImage } from './image_utils.js?v=20251104';

// ------------------------------------------------
//                   ELEMENTS 
// ------------------------------------------------

// Helper to get element by ID
const el = (id) => document.getElementById(id);
// Main elements
const fileA = el('fileA'), imgA = el('imgA'), canvasA = el('canvasA');
const fileJSON = el('fileJSON'), fileB = el('fileB'), imgB = el('imgB');
const canvasMatches = el('canvasMatches');
// Action buttons
const btnDetect = el('btnDetect');
const btnDownload = el('btnDownload');
const btnMatch = el('btnMatch');
// Stats display elements
const statsA = el('statsA'), statsB = el('statsB');
// ORB parameters elements
const nfeatures = el('nfeatures');
const ratio = el('ratio');
const ransac = el('ransac'); 
const edgeThreshold = el('edgeThreshold');
const scaleFactor = el('scaleFactor');
const nlevels = el('nlevels');
const fastThreshold = el('fastThreshold');
const patchSize = el('patchSize');
// Video frame extractor elements
const fileVideo = el('fileVideo');
const frameNumber = el('frameNumber');
const btnExtractFrame = el('btnExtractFrame');
const videoPreview = el('videoPreview');
const canvasFrame = el('canvasFrame');
const frameImage = new Image();
// ORB detection section element
const detectOrb = el('detectOrb');

// Crop box elements
const cropBox = document.getElementById('cropBox');
const cropBoxB = document.getElementById('cropBoxB'); 

setupCropBox(imgA, cropBox);
setupCropBox(imgB, cropBoxB);

// ------------------------------------------------
//                     STATE 
// ------------------------------------------------

// ORB feature tool state
let mod;
let cvReady = false;
let imgAReady = false;
let imgBReady = false;
let detectResult = null;
let loadedJSON = null;

// Video frame extractor state
let videoExtractor;

const haveFeatures = () => Boolean(loadedJSON || detectResult);

// ------------------------------------------------
//                     HELPERS 
// ------------------------------------------------

// Get crop rectangle relative to an image element
// ______________________________________________________

function getCropRectGeneric(imgEl, cropBoxEl) {
    const imgRect = imgEl.getBoundingClientRect();
    const cropRect = cropBoxEl.getBoundingClientRect();
    const scaleX = imgEl.naturalWidth / imgRect.width;
    const scaleY = imgEl.naturalHeight / imgRect.height;
    return {
    x: Math.round((cropRect.left - imgRect.left) * scaleX),
    y: Math.round((cropRect.top - imgRect.top) * scaleY),
    width: Math.round(cropRect.width * scaleX),
    height: Math.round(cropRect.height * scaleY)
    };
}

// Compatible imshow function: uses cv.imshow if available, 
// otherwise converts Mat to ImageData and draws to canvas
// ______________________________________________________

function imshowCompat(canvas, mat) {
    if (window.cv.imshow) { // if the build supports imshow
    window.cv.imshow(canvas, mat); // use it directly
    return;
    }
    let rgba = mat; // placeholder for RGBA Mat
    
    /* 
    Convert the input Mat to RGBA format for display on a canvas. OpenCV Mats can 
    have different channel formats (RGB, RGBA). This ensures the Mat is always in 
    CV_8UC4 (RGBA) format for compatibility with ImageData and canvas.
    */

    // If the Mat is in 3-channel RGB format (CV_8UC3), convert it to 4-channel RGBA.
    if (mat.type() === window.cv.CV_8UC3) {
    rgba = new window.cv.Mat(); // Create an empty Mat for the result
    window.cv.cvtColor(mat, rgba, window.cv.COLOR_RGB2RGBA); // Convert RGB to RGBA
    // If the Mat is not already in 4-channel RGBA format (CV_8UC4), convert it.
    } else if (mat.type() !== window.cv.CV_8UC4) {
    const tmp = new window.cv.Mat(); // Temporary Mat for conversion
    window.cv.cvtColor(mat, tmp, window.cv.COLOR_RGBA2RGBA); // Convert to RGBA
    rgba = tmp; // Use the converted Mat   
    // If the Mat is already in RGBA format, clone it to avoid modifying the original.
    } else {
    rgba = mat.clone();
    }

    // Create ImageData from the RGBA Mat data
    const imageData = new ImageData(
    new Uint8ClampedArray(rgba.data), // pixel data
    rgba.cols, rgba.rows // width and height
    );

    // Resize the canvas and put the ImageData onto it
    canvas.width = rgba.cols; // set canvas width
    canvas.height = rgba.rows; // set canvas height
    canvas.getContext('2d').putImageData(imageData, 0, 0); // draw image data
    rgba.delete(); // clean up temporary Mat if created
}

// Refresh button enabled/disabled states 
// ______________________________________________________

function refreshButtons() {
    // Log current states for debugging
    console.log('refreshButtons', { cvReady, imgAReady, imgBReady, haveFeatures: haveFeatures(), detectResult });
    btnDetect.disabled = !(cvReady && imgAReady); // Detect enabled if cv and imgA ready
    btnDownload.disabled = !(detectResult && detectResult.descriptors); // Download enabled if detection result with descriptors
    btnMatch.disabled = !(cvReady && imgBReady && haveFeatures()); // Match enabled if cv, imgB ready and features available
}

// Get crop rectangle relative to image A
// ______________________________________________________

function getCropRect() {
    const imgRect = imgA.getBoundingClientRect(); // get image A bounding rect
    const cropRect = cropBox.getBoundingClientRect(); // get crop box bounding rect

    // Calculate scale factors
    const scaleX = imgA.naturalWidth / imgRect.width;
    const scaleY = imgA.naturalHeight / imgRect.height;

    return { // return crop rectangle in image coordinates
    x: Math.round((cropRect.left - imgRect.left) * scaleX),
    y: Math.round((cropRect.top - imgRect.top) * scaleY),
    width: Math.round(cropRect.width * scaleX),
    height: Math.round(cropRect.height * scaleY)
    };
}

// Initialize ORBModule when OpenCV.js is ready
// ______________________________________________________

function onCvReady() {
    // Create ORBModule instance
    try {
    mod = new ORBModule(window.cv); // create ORBModule instance
    cvReady = true; // set cvReady flag
    console.log('onCvReady → cvReady=true'); // log readiness
    console.log('cv.imread:', typeof window.cv.imread); // log imread availability
    // Catch any errors during initialization
    } catch (e) {
    console.error('cv init error', e);
    cvReady = false;
    }
    // Refresh button states
    refreshButtons();
}

// init: catch both the event *and* the already-ready case
if (window.cvIsReady || (window.cv && (window.cv.Mat || window.cv.getBuildInformation))) {
    onCvReady();
} else {
    document.addEventListener('cv-ready', onCvReady, { once: true });
}

// ------------------------------------------------
//                     EVENTS 
// ------------------------------------------------
    
// Video frame extraction
// ________________________________________________

fileVideo.addEventListener('change', () => {
    const f = fileVideo.files?.[0];
    if (!f) return;
    const url = URL.createObjectURL(f);
    videoPreview.src = url;
    videoPreview.load();
    videoPreview.hidden = false;
    frameNumber.hidden = false;
    btnExtractFrame.hidden = false;
    videoExtractor = new VideoFrameExtractor(videoPreview, canvasFrame);
});

// Extract frame button
// ________________________________________________

btnExtractFrame.addEventListener('click', async () => {
    const frameIdx = Number(frameNumber.value) || 0;
    const fps = 25; // Can make this user configurable later
    try {
        await videoExtractor.extractFrame(frameIdx, fps);

        // Convert the extracted frame (canvasFrame) to an image and set as imgA
        imgA.src = canvasFrame.toDataURL();
        imgA.onload = () => {
            imgA.hidden = false;
            // Get the rendered size of the image
            const imgRect = imgA.getBoundingClientRect();
            // Set parent container size to match image
            const parent = imgA.parentElement;
            parent.style.width = imgRect.width + 'px';
            parent.style.height = imgRect.height + 'px';

            // Show and initialize the crop box
            cropBox.style.display = 'block';
            cropBox.hidden = false;
            cropBox.style.left = '0px';
            cropBox.style.top = '0px';
            cropBox.style.width = imgRect.width + 'px';
            cropBox.style.height = imgRect.height + 'px';

            imgAReady = true;
            videoPreview.hidden = true;
            frameNumber.hidden = true;
            btnExtractFrame.hidden = true;
            detectResult = null;
            statsA.textContent = '';
            canvasA.hidden = true;
            canvasFrame.hidden = true;
            refreshButtons();
        };
    } catch (e) {
        alert('Frame extraction failed: ' + e);
        imgAReady = false;
        refreshButtons();
    }
});

// Image A load
// ________________________________________________

fileA.addEventListener('change', async () => {
    const f = fileA.files?.[0]; // get selected file
    if (!f) return; // if no file, exit
    try {
        // Load image into imgA and initialize crop box after image is rendered
        await loadImg(f, imgA, cropBox);

        // Ensure imgA and cropBox are visible before sizing
        imgA.hidden = false;
        cropBox.hidden = false;
        detectOrb.hidden = false; // show ORB detection section

        // Get the rendered size of the image
        const imgRect = imgA.getBoundingClientRect();
        const parent = imgA.parentElement;
        parent.style.width = imgRect.width + 'px';
        parent.style.height = imgRect.height + 'px';

        // Initialize crop box to cover the whole image
        cropBox.style.display = 'block';
        cropBox.style.left = '0px';
        cropBox.style.top = '0px';
        cropBox.style.width = imgRect.width + 'px';
        cropBox.style.height = imgRect.height + 'px';

        imgAReady = true; // set imgAReady flag
        detectResult = null; // reset previous detection result
        statsA.textContent = ''; // clear stats
        canvasA.hidden = true; // hide canvasA
    } catch (e) {
        console.error('Image A preview error', e);
        imgAReady = false;
        imgA.hidden = true;
    }
    refreshButtons();
});

// Image B load
// ________________________________________________

fileB.addEventListener('change', async () => {
    const f = fileB.files?.[0];
    if (!f) return;
    try {
    await loadImg(f, imgB, cropBoxB);
    imgBReady = true;
    imgB.hidden = false;
    cropBoxB.hidden = false;
    statsB.textContent = '';
    canvasMatches.hidden = true;
    } catch (e) {
    console.error('Image B preview error', e);
    imgBReady = false;
    imgB.hidden = true;
    }
    refreshButtons();
});

// JSON load
// ________________________________________________

fileJSON.addEventListener('change', async () => {
    const f = fileJSON.files?.[0];
    if (!f) return;
    try {
    loadedJSON = JSON.parse(await f.text());
    } catch (e) {
    console.error('JSON parse error', e);
    loadedJSON = null;
    }
    refreshButtons();
});

// Detect ORB
// ________________________________________________

btnDetect.addEventListener('click', () => {
    if (!cvReady || !imgAReady) return;
    const cv = window.cv;
    // const src = cv.imread(imgA);
    const cropRect = getCropRect();
    // Crop image A according to crop box
    const croppedCanvas = cropImage(imgA, cropRect);
    // Convert cropped image to Mat
    const src = matFromImageEl(croppedCanvas);
    // Set ORB options
    const opts = { 
    nfeatures: Number(nfeatures.value) || 1200,
    edgeThreshold: Number(edgeThreshold.value) || 31,
    scaleFactor: Number(scaleFactor.value) || 1.2,
    nlevels: Number(nlevels.value) || 8,
    fastThreshold: Number(fastThreshold.value) || 20,
    patchSize: Number(patchSize.value) || 31,
    };
    // Run detection
    try {
    // Perform ORB detection
    detectResult = mod.detectORB(src, opts);
    // Offset keypoints to match their position on the full image
    const offsetKeypoints = detectResult.keypoints.map(kp => ({
        ...kp,
        x: kp.x + cropRect.x,
        y: kp.y + cropRect.y
    }));
    // Update stats display
    statsA.textContent =
        `A: ${detectResult.width}x${detectResult.height}\n` +
        `keypoints: ${detectResult.keypoints.length}\n` +
        `descriptors: ${detectResult.descriptors?.rows ?? 0} x ${detectResult.descriptors?.cols ?? 0}`;
    // show canvasA (image with keypoints)
    canvasA.hidden = false; 
    imgA.hidden = true; // hide original imageA
    cropBox.style.display = 'none'; // hide crop box when showing keypoints
    // Draw keypoints on the full image
    const fullMat = matFromImageEl(imgA);
    mod.drawKeypoints(fullMat, offsetKeypoints, canvasA);
    fullMat.delete(); // clean up full image Mat
    
    } catch (e) { // catch errors
    console.error('Detect error', e);
    alert('Detect failed. See console.');
    detectResult = null;
    } finally { // cleanup
    src.delete(); // release Mat
    refreshButtons(); // refresh buttons
    }
});

// Download JSON
// ________________________________________________

btnDownload.addEventListener('click', () => {
    if (!detectResult) return;
    const json = mod.exportJSON(detectResult);
    const blob = new Blob([JSON.stringify(json, null, 2)], { type: 'application/json' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = 'features.json';
    a.click();
    URL.revokeObjectURL(a.href);
});

// Match features.json to Image B
// ________________________________________________

btnMatch.addEventListener('click', () => {
    if (!cvReady || !imgBReady) return; // If not ready, exit
    if (!loadedJSON && !detectResult) { // No features available
    alert('Load features.json or run Detect on Image A first.');
    return;
    }
    // Get OpenCV.js reference
    const cv = window.cv;

    // Crop and process image B
    const cropRectB = getCropRectGeneric(imgB, cropBoxB);
    const croppedCanvasB = cropImage(imgB, cropRectB);
    const target = matFromImageEl(croppedCanvasB);

    // Detect ORB features on cropped image B and offset keypoints
    // Options for ORB detection
    const opts = {
    nfeatures: Number(nfeatures.value) || 1200,
    edgeThreshold: Number(edgeThreshold.value) || 31,
    scaleFactor: Number(scaleFactor.value) || 1.2,
    nlevels: Number(nlevels.value) || 8,
    fastThreshold: Number(fastThreshold.value) || 20,
    patchSize: Number(patchSize.value) || 31
    };
    // Detect features on image B
    const detectResultB = mod.detectORB(target, opts);
    // Offset keypoints to match their position on the full image B
    const offsetKeypointsB = detectResultB.keypoints.map(kp => ({
        ...kp,
        x: kp.x + cropRectB.x,
        y: kp.y + cropRectB.y
    }));

    // Prepare source features from loaded JSON or detected result
    const source = loadedJSON || mod.exportJSON(detectResult);
    const cropRectA = getCropRectGeneric(imgA, cropBox);
    const offsetKeypointsA = source.keypoints.map(kp => ({
        ...kp,
        x: kp.x + cropRectA.x,
        y: kp.y + cropRectA.y
    }));

    try {
    // Match features
    const res = mod.matchToTarget(
        { ...source, keypoints: offsetKeypointsA },
        target,
        {
        useKnn: true,
        ratio: Number(ratio.value) || 0.75,
        ransacReprojThreshold: Number(ransac.value) || 3.0
        }
    );

    statsB.textContent =
        `B: ${target.cols}x${target.rows}\n` +
        `matches: ${res.matches.length}\n` +
        `inliers: ${res.numInliers ?? 0}\n` +
        (res.homography ? `H: [${res.homography.map(v => v.toFixed(3)).join(', ')}]` : 'H: (none)');

    if (!Array.isArray(offsetKeypointsA) || !Array.isArray(offsetKeypointsB) || !Array.isArray(res.matches)) {
        alert('No keypoints or matches found. Check your crop area and images.');
        return;
    }
    
    // Draw matches on full images using offset keypoints
    const A = matFromImageEl(imgA);
    const B = matFromImageEl(imgB);
    
    mod.drawMatches(A, B, offsetKeypointsA, offsetKeypointsB, res);
    imshowCompat(canvasMatches, mod._lastCanvasMat);

    fileB.hidden = true;
    canvasMatches.hidden = false;
    A.delete();
    B.delete();
    mod._releaseLastCanvasMat();
    } catch (e) {
    console.error('Match error', e);
    alert('Match failed. See console.');
    } finally {
    target.delete();
    refreshButtons();
    }
});
