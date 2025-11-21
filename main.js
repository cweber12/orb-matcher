// main.js
// Main script for ORB feature detection and matching tool

import { ORBModule } from './orb_module.js?v=20251104';
import { setupCropBox } from './setup_crop_box.js?v=20251104';
import { loadImg, matFromImageEl, cropImage } from './image_utils.js?v=20251104';

// ELEMENTS 
// ------------------------------------------------

// Helper to get element by ID
const el = (id) => document.getElementById(id);

// Main elements

// Image A elements
const fileA = el('fileA');                      // File input for Image A
const imgA  = el('imgA');                       // Image A element
const canvasA = el('canvasA');                  // Canvas for Image A display

// JSON file element
const fileJSON = el('fileJSON');

// Image B elements
const fileB = el('fileB');                      // File input for Image B
const imgB  = el('imgB');                       // Image B element

// Canvas for displaying matches
const canvasMatches = el('canvasMatches'); 

// Action buttons
const btnDetect = el('btnDetect');              // Detect features button
const btnDownload = el('btnDownload');          // Download features.json button
const btnMatch = el('btnMatch');                // Match features button

// ORB detection stats elements
const statsA = el('statsA');  
const statsB = el('statsB');

// ORB parameters elements
const nfeatures = el('nfeatures');              // Number of features to detect
const ratio = el('ratio');                      // Ratio for feature matching
const ransac = el('ransac');                    // RANSAC threshold
const edgeThreshold = el('edgeThreshold');      // Edge threshold for ORB
const scaleFactor = el('scaleFactor');          // Scale factor for ORB
const nlevels = el('nlevels');                  // Number of levels in the pyramid
const fastThreshold = el('fastThreshold');      // FAST threshold for ORB
const patchSize = el('patchSize');              // Patch size for ORB

// ORB detection section element
const detectOrb = el('detectOrb');

// Crop box for Image A
const cropBox = document.getElementById('cropBox');
// Crop box for Image B
const cropBoxB = document.getElementById('cropBoxB'); 

setupCropBox(imgA, cropBox);    // Initialize crop box for Image A
setupCropBox(imgB, cropBoxB);   // Initialize crop box for Image B

// STATE 
// ------------------------------------------------

// ORB feature tool state
let mod;                    // ORBModule instance
let cvReady = false;        // OpenCV.js readiness flag 
let imgAReady = false;      // Image A readiness flag
let imgBReady = false;      // Image B readiness flag
let detectResult = null;    // Detection result state
let loadedJSON = null;      // Loaded JSON state

// Check if features available (from detection or loaded JSON)
const haveFeatures = () => Boolean(loadedJSON || detectResult);

// HELPERS 
// ------------------------------------------------

// Generic function to get crop rectangle relative to an image element
//  - imgEl: HTMLImageElement
//  - cropBoxEl: crop box HTML element
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

// Draws a Mat on a canvas using cv.imshow if available, 
// else converts Mat to ImageData and draws manually.
//   - canvas: HTMLCanvasElement to draw on
//   - mat: cv.Mat to display
function imshowCompat(canvas, mat) {
    
    // if the build supports imshow
    if (window.cv.imshow) {             
        window.cv.imshow(canvas, mat);  // use it directly
        return;                         // done
    }
    
    // placeholder for RGBA Mat
    let rgba = mat; 

    // If the Mat is in 3-channel RGB format (CV_8UC3)
    //   - convert to 4-channel RGBA
    if (mat.type() === window.cv.CV_8UC3) {
        rgba = new window.cv.Mat();     // Temporary Mat for conversion
        window.cv.cvtColor(             // Convert to RGBA
            mat,                        //   - source Mat
            rgba,                       //   - destination Mat
            window.cv.COLOR_RGB2RGBA    //   - color conversion code
        );

    // If the Mat is not already in 4-channel RGBA format (CV_8UC4)
    //   - convert to RGBA
    } else if (mat.type() !== window.cv.CV_8UC4) {
        const tmp = new window.cv.Mat();    // Temporary Mat for conversion
        window.cv.cvtColor(                 // Convert to RGBA
            mat,                            //   - source Mat               
            tmp,                            //   - destination Mat
            window.cv.COLOR_RGBA2RGBA       //   - color conversion code
        ); 
        rgba = tmp;                         // Use converted Mat    
                                     
    // If the Mat is already in RGBA format
    //   - Clone to avoid modifying original
    } else {
        rgba = mat.clone(); 
    }

    // Create ImageData from the RGBA Mat data
    const imageData = new ImageData(
        new Uint8ClampedArray(rgba.data), // pixel data
        rgba.cols,                        // width 
        rgba.rows                         // height
    );

    // Resize the canvas and put the ImageData onto it
    canvas.width = rgba.cols;   // set canvas width
    canvas.height = rgba.rows;  // set canvas height

    // Draw ImageData to canvas
    canvas.getContext('2d').putImageData(imageData, 0, 0); 

    // Clean up temporary Mat if created
    rgba.delete(); 
}

// Refresh button enabled/disabled states 
function refreshButtons() {
    
    // Log current states for debugging
    console.log('refreshButtons', { cvReady, imgAReady, imgBReady, haveFeatures: haveFeatures(), detectResult });
    
    // Enable/disable buttons based on current states
    btnDetect.disabled = !(cvReady && imgAReady); 
    btnDownload.disabled = !(detectResult && detectResult.descriptors); 
    btnMatch.disabled = !(cvReady && imgBReady && haveFeatures()); 
}

// Get crop rectangle relative to image A
function getCropRect() {
    
    // Get bounding rectangles for image A and the crop box
    const imgRect = imgA.getBoundingClientRect(); 
    const cropRect = cropBox.getBoundingClientRect(); 

    // Calculate scale factors (natural image size vs rendered size)
    const scaleX = imgA.naturalWidth / imgRect.width;
    const scaleY = imgA.naturalHeight / imgRect.height;

    // Return crop rectangle in natural image coordinates
    return { 
        x: Math.round((cropRect.left - imgRect.left) * scaleX), // x coordinate (left)
        y: Math.round((cropRect.top - imgRect.top) * scaleY),   // y coordinate (top)
        width: Math.round(cropRect.width * scaleX),             // width
        height: Math.round(cropRect.height * scaleY)            // height
    };
}

// Initialize ORBModule when OpenCV.js is ready
function onCvReady() {
    
    // Create ORBModule instance
    try {
        mod = new ORBModule(window.cv); // create ORBModule instance
        cvReady = true;                 // set cvReady flag
        
        // DEBUG
        console.log('onCvReady → cvReady=true');    
        console.log('cv.imread:', typeof window.cv.imread);
    
    // Catch any errors during initialization
    } catch (e) {
        console.error('cv init error', e);  
        cvReady = false;                    
    }
    
    // Refresh button states
    refreshButtons();
}

// Check if OpenCV.js is already ready before setting up event listeners
//   - if ready, call onCvReady immediately
//   - else, set up event listener for 'cv-ready' event
if (window.cvIsReady || (window.cv && (window.cv.Mat || window.cv.getBuildInformation))) {
    onCvReady();
} else {
    document.addEventListener('cv-ready', onCvReady, { once: true });
}

// EVENTS 
// ------------------------------------------------    

// Image A load
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
    const keypointsA = source.keypoints; 
    try {
    // Match features
    const res = mod.matchToTarget(
        { ...source, keypoints: keypointsA },
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

    if (!Array.isArray(keypointsA) || !Array.isArray(offsetKeypointsB) || !Array.isArray(res.matches)) {
        alert('No keypoints or matches found. Check your crop area and images.');
        return;
    }
    
    
    // Draw matches on full images using offset keypoints
    const A = matFromImageEl(imgA);
    const B = matFromImageEl(imgB);
    
    mod.drawMatches(A, B, keypointsA, offsetKeypointsB, res, source.imageSize);
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
