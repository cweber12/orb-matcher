// main.js
// Main script for ORB feature detection and matching tool

import { ORBModule } from './orb_module.js?v=20251104';
import { setupCropBox } from './setup_crop_box.js?v=20251104';
import { loadImg, matFromImageEl, cropImage } from './image_utils.js?v=20251104';

// ELEMENTS 
// ---------------------------------------------------------------------------

// Helper to get element by ID
const el = (id) => document.getElementById(id);

// File elements
const fileA = el('fileA');       // File input for Image A
const imgA  = el('imgA');        // Image A element
const canvasA = el('canvasA');   // Canvas for Image A display
const fileJSON = el('fileJSON'); // File input for features.json
const fileB = el('fileB');       // File input for Image B
const imgB  = el('imgB');        // Image B element

// Canvas for displaying matches
const canvasMatches = el('canvasMatches'); 

// Action buttons
const btnDetect = el('btnDetect');      // Detect features button
const btnDownload = el('btnDownload');  // Download features.json button
const btnMatch = el('btnMatch');        // Match features button

// ORB detection stats elements
const statsA = el('statsA');  
const statsB = el('statsB');

// ORB parameters elements
const nfeatures = el('nfeatures');          // Number of features to detect
const ratio = el('ratio');                  // Ratio for feature matching
const ransac = el('ransac');                // RANSAC threshold
const edgeThreshold = el('edgeThreshold');  // Edge threshold for ORB
const scaleFactor = el('scaleFactor');      // Scale factor for ORB
const nlevels = el('nlevels');              // Number of levels in the pyramid
const fastThreshold = el('fastThreshold');  // FAST threshold for ORB
const patchSize = el('patchSize');          // Patch size for ORB

// Section elements for showing/hiding sections
const detectOrb = el('detectOrb');
const matchSection = el('matchSection');          

// Crop box for Image A
const cropBox = document.getElementById('cropBox');
// Crop box for Image B
const cropBoxB = document.getElementById('cropBoxB'); 

setupCropBox(imgA, cropBox);    // Initialize crop box for Image A
setupCropBox(imgB, cropBoxB);   // Initialize crop box for Image B

// STATE 
// ---------------------------------------------------------------------------

let mod;                    // ORBModule instance
let cvReady = false;        // OpenCV.js readiness flag 
let imgAReady = false;      // Image A readiness flag
let imgBReady = false;      // Image B readiness flag
let detectResult = null;    // Detection result state
let loadedJSON = null;      // Loaded JSON state
let detectJSON = null;      // Detected features JSON state

// Check if features available (from detection or loaded JSON)
const haveFeatures = () => Boolean(loadedJSON || detectResult);

// HELPERS 
// ---------------------------------------------------------------------------

/* ___________________________________________________________________________

Generic function to get crop rectangle relative to an image element
  - imgEl: HTMLImageElement
  - cropBoxEl: crop box HTML element
____________________________________________________________________________ */

function getCropRectGeneric(imgEl, cropBoxEl) {
    const imgRect = imgEl.getBoundingClientRect();      // rendered image rect
    const cropRect = cropBoxEl.getBoundingClientRect(); // crop box rect
    
    /* Calculate scale factors between natural image size and displayed size
       to account for resizing (with CSS) in the browser.
        - naturalWidth/Height: original image size
        - imgRect.width/height: displayed size when rendered in browser */
    const scaleX = imgEl.naturalWidth / imgRect.width;   // scale factor X
    const scaleY = imgEl.naturalHeight / imgRect.height; // scale factor Y   
    
    // calculate crop rectangle in natural image coordinates 
    // NOTE: See diagram below for coordinate reference
    const result = {    
        x: Math.round((cropRect.left - imgRect.left) * scaleX),
        y: Math.round((cropRect.top - imgRect.top) * scaleY),
        width: Math.round(cropRect.width * scaleX),
        height: Math.round(cropRect.height * scaleY)
    };

    /* 
    DIAGRAM OF IMG AND CROP BOX COORDINATES

    (imgRect.left, imgRect.top)
    |
    V    (cropRect.left, cropRect.top)
    -----|----------------------------- 
    |    |     IMAGE ELEMENT          |  
    |    |                            |    
    |    |                            |
    |    |                            |
    |    V                            |
    |    ----------------------       | 
    |    |                    |       |
    |    |     CROP BOX       |       |
    |    |                    |       |
    |    |                    |       |
    |    ---------------------- <-----|---- (cropRect.width, cropRect.height)
    |                                 |
    ----------------------------------- <-- (imgRect.width, imgRect.height) */

    // DEBUG
    console.log('getCropRectGeneric:', {
        imgRect, cropRect, scaleX, scaleY, result
    });
    return result;       
}

/*___________________________________________________________________________

Draw a Mat on a canvas using cv.imshow if available, else converts Mat 
to ImageData and draws manually.
   - canvas: HTMLCanvasElement to draw on
   - mat: cv.Mat to display 
____________________________________________________________________________*/

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
        rgba = new window.cv.Mat();   // Temporary Mat for conversion
        window.cv.cvtColor(           // Convert to RGBA
            mat,                      //   - source Mat
            rgba,                     //   - destination Mat
            window.cv.COLOR_RGB2RGBA  //   - color conversion code
        );
    // If the Mat is not already in 4-channel RGBA format (CV_8UC4)
    //   - convert to RGBA
    } else if (mat.type() !== window.cv.CV_8UC4) {
        const tmp = new window.cv.Mat(); // Temporary Mat for conversion
        window.cv.cvtColor(              // Convert to RGBA
            mat,                         //   - source Mat               
            tmp,                         //   - destination Mat
            window.cv.COLOR_RGBA2RGBA    //   - color conversion code
        ); 
        rgba = tmp; // Use converted Mat                                         
    // If the Mat is already in RGBA format
    } else {
        rgba = mat.clone(); // clone to avoid modifying original 
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

/*___________________________________________________________________________

Refresh button enabled/disabled states based on current app states
  - Ensures buttons are only enabled when the right conditions are met
____________________________________________________________________________*/

function refreshButtons() {    
    // Log current states for debugging
    console.log('refreshButtons', { cvReady, imgAReady, imgBReady, haveFeatures: haveFeatures(), detectResult });    
    // Enable/disable buttons based on current states
    btnDetect.disabled = !(cvReady && imgAReady); 
    btnDownload.disabled = !(detectResult && detectResult.descriptors); 
    btnMatch.disabled = !(cvReady && imgBReady && haveFeatures()); 
}
/*___________________________________________________________________________

Initialize ORBModule when OpenCV.js is ready
____________________________________________________________________________*/

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

/*___________________________________________________________________________

Check if OpenCV.js is already ready before setting up event listeners
  - if ready, call onCvReady immediately
  - else, set up event listener for 'cv-ready' event
____________________________________________________________________________*/

if (window.cvIsReady || (window.cv && (window.cv.Mat || window.cv.getBuildInformation))) {
    onCvReady();
} else {
    document.addEventListener('cv-ready', onCvReady, { once: true });
}

// EVENTS 
// ---------------------------------------------------------------------------    

/*___________________________________________________________________________

Image A load event handler
  - Loads image, initializes crop box, and updates state
____________________________________________________________________________*/  

fileA.addEventListener('change', async () => {
    const f = fileA.files?.[0]; // get selected file
    if (!f) return;             // if no file, exit
    try {                       // try to load image
        await loadImg(f, imgA, cropBox); // load image into imgA
        imgA.hidden = false;             // show imgA
        cropBox.hidden = false;          // show crop box
        detectOrb.hidden = false;        // show ORB detection section

        // Get the rendered size of the image
        const imgRect = imgA.getBoundingClientRect();
        const parent = imgA.parentElement;
        parent.style.width = imgRect.width + 'px';
        parent.style.height = imgRect.height + 'px';

        // Initialize crop box to cover the whole image
        cropBox.style.display = 'block';              // show crop box
        cropBox.style.left = '0px';                   // position left
        cropBox.style.top = '0px';                    // position top
        cropBox.style.width = imgRect.width + 'px';   // set width
        cropBox.style.height = imgRect.height + 'px'; // set height

        // Update flags and reset previous state
        imgAReady = true;         // set imgAReady flag
        detectResult = null;      // reset previous detection result
        statsA.textContent = '';  // clear stats
        canvasA.hidden = true;    // hide canvasA
    
    // Catch any errors during image loading
    } catch (e) {
        console.error('Image A preview error', e);
        imgAReady = false;
        imgA.hidden = true;
    }
    refreshButtons(); // refresh button states
});

/*___________________________________________________________________________

Image B load event handler
  - Loads image, initializes crop box, and updates state
____________________________________________________________________________*/

fileB.addEventListener('change', async () => {
    const f = fileB.files?.[0]; // get selected file
    if (!f) return;             // if no file, exit
    try {                       // try to load image
        await loadImg(f, imgB, cropBoxB); // load image into imgB    
        imgB.hidden = false;              // show imgB
        cropBoxB.hidden = false;          // show crop box B
        matchSection.hidden = false;      // show match section

        // Get the rendered size of the image
        const imgRect = imgB.getBoundingClientRect();
        const parent = imgB.parentElement;
        parent.style.width = imgRect.width + 'px';
        parent.style.height = imgRect.height + 'px';

        // Initialize crop box to cover the whole image
        cropBoxB.style.display = 'block';              // show crop box B
        cropBoxB.style.left = '0px';                   // position left
        cropBoxB.style.top = '0px';                    // position top
        cropBoxB.style.width = imgRect.width + 'px';   // set width
        cropBoxB.style.height = imgRect.height + 'px'; // set height
        
        imgBReady = true;            // set imgBReady flag
        statsB.textContent = '';     // clear prev stats B
        canvasMatches.hidden = true; // hide prev matches canvas
    
    // Catch any errors during image loading
    } catch (e) {
        console.error('Image B preview error', e);
        imgBReady = false;
        imgB.hidden = true;
    }
    refreshButtons(); // refresh button states
});

/*___________________________________________________________________________

features.json load event handler
  - Loads and parses features.json file
____________________________________________________________________________*/

fileJSON.addEventListener('change', async () => {
    const f = fileJSON.files?.[0]; // get selected file
    if (!f) return;                // if no file, exit
    try {                          // parse JSON
        loadedJSON = JSON.parse(await f.text()); 
    // catch parse errors 
    } catch (e) {                                    
        console.error('JSON parse error', e);
        loadedJSON = null;
    }
    refreshButtons(); // refresh button states
});

/*___________________________________________________________________________

Detect ORB features on Image A
  - Crops image according to crop box and runs ORB detection
  - Updates stats and displays keypoints on canvas
____________________________________________________________________________*/

btnDetect.addEventListener('click', () => {
    if (!cvReady || !imgAReady) return; // If not ready, exit
    const cv = window.cv;               // Get OpenCV.js reference 
    
    // Crop image A according to crop box and convert to Mat
    const cropRect = getCropRectGeneric(imgA, cropBox); // get crop rectangle
    const croppedCanvas = cropImage(imgA, cropRect);    // crop image to canvas
    const src = matFromImageEl(croppedCanvas);          // convert to Mat     

    // Set ORB options from input fields, use defaults if inputs empty
    const opts = { 
        nfeatures: Number(nfeatures.value) || 1200,
        edgeThreshold: Number(edgeThreshold.value) || 31,
        scaleFactor: Number(scaleFactor.value) || 1.2,
        nlevels: Number(nlevels.value) || 8,
        fastThreshold: Number(fastThreshold.value) || 20,
        patchSize: Number(patchSize.value) || 31,
    };
    
    // Run ORB detection
    try {
        // Detect ORB features on cropped image
        detectResult = mod.detectORB(src, opts);
        // Get full image dimensions
        const fullW = imgA.naturalWidth || imgA.width;
        const fullH = imgA.naturalHeight || imgA.height;
        // Offset keypoints to full image coordinates
        const keypointsFullPx = detectResult.keypoints.map(kp => ({
            ...kp,                  // copy keypoint
            x: kp.x + cropRect.x,   // offset x by crop rectangle
            y: kp.y + cropRect.y,   // offset y by crop rectangle
        }))
        // Start from the module's standard JSON (descriptors + normalized to CROPPED size)
        const baseJson = mod.exportJSON(detectResult);

        // Build JSON
        detectJSON = {
            ...baseJson,                                // copy base JSON
            imageSize: { width: fullW, height: fullH }, // full image size
            // Normalize keypoints to full image size [0 - 1]
            keypoints: keypointsFullPx.map(kp => ({     
                ...kp,            // copy keypoint 
                x: kp.x / fullW,  // normalize x to full image width
                y: kp.y / fullH,  // normalize y to full image height
            })),
        };
        // NOTE: descriptors stay exactly as baseJson.descriptors (with data_b64)
        
        // Update stats display
        statsA.textContent =
            `A: ${detectResult.width}x${detectResult.height}\n` +
            `keypoints: ${detectResult.keypoints.length}\n` +
            `descriptors: ${detectResult.descriptors?.rows ?? 0} x ${detectResult.descriptors?.cols ?? 0}`;
        
        canvasA.hidden = false;         // show canvasA (image with keypoints)
        imgA.hidden = true;             // hide original imageA
        cropBox.style.display = 'none'; // hide crop box when showing keypoints
        
        const fullMat = matFromImageEl(imgA); // Create Mat from full image A
        mod.drawKeypoints(                    // Draw keypoints on full image
            fullMat,          // full image Mat
            keypointsFullPx,  // keypoints with full image coordinates
            canvasA           // canvas to draw on
        );
        fullMat.delete(); 
    // Catch any errors during detection
    } catch (e) { 
        console.error('Detect error', e);
        alert('Detect failed. See console.');
        detectResult = null;
        detectJSON = null;
    // Cleanup
    } finally { 
        src.delete();     // release Mat
        refreshButtons(); // refresh buttons
    }
});

/*___________________________________________________________________________

Download detected features as features.json 
  - features.json contains keypoints and descriptors
  - used for matching to image B later
____________________________________________________________________________*/

btnDownload.addEventListener('click', () => {
    if (!detectResult) return; // If no detection result, exit
    // Inintialize JSON to download
    const json = detectJSON || mod.exportJSON(detectResult);
    const blob = new Blob( // Create Blob from JSON string
        [JSON.stringify(json, null, 2)], // pretty-print with 2-space indent
        { type: 'application/json' }     // MIME type
    );
    const a = document.createElement('a'); // create temporary anchor element
    a.href = URL.createObjectURL(blob);    // create object URL for Blob
    a.download = 'features.json';          // set download filename
    a.click();                             // trigger download
    URL.revokeObjectURL(a.href);           // revoke object URL
});

/*___________________________________________________________________________

Match features from Image A to Image B
    - Crops image B according to crop box and runs ORB detection
    - Matches features from loaded JSON or detected features on image A
    - Draws matches on full images
____________________________________________________________________________*/

btnMatch.addEventListener('click', () => {
    if (!cvReady || !imgBReady) return; // If not ready, exit
    if (!loadedJSON && !detectResult) { // No features available
        alert('Load features.json or run Detect on Image A first.');
        return;
    }
    
    const cv = window.cv; // Get OpenCV.js reference

    // Crop image B according to crop box and convert to Mat
    const cropRectB = getCropRectGeneric(imgB, cropBoxB);
    const croppedCanvasB = cropImage(imgB, cropRectB);
    const target = matFromImageEl(croppedCanvasB);

    // Options for ORB detection on image B 
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
        ...kp,                  // copy keypoint
        x: kp.x + cropRectB.x,  // offset x by crop rectangle
        y: kp.y + cropRectB.y   // offset y by crop rectangle
    }));

    // Prepare source features from loaded JSON or detected result
    const source = loadedJSON || detectJSON || mod.exportJSON(detectResult);
    const keypointsA = source.keypoints; // keypoints from source
    try {
        // Match features
        const res = mod.matchToTarget(
            { ...source, keypoints: keypointsA }, // source features
            target,                               // target Mat (image B)
            {                                     // matching options
                useKnn: true,
                ratio: Number(ratio.value) || 0.75,
                ransacReprojThreshold: Number(ransac.value) || 3.0
            }
        );

        // Update stats for image B
        statsB.textContent =
            `B: ${target.cols}x${target.rows}\n` +
            `matches: ${res.matches.length}\n` +
            `inliers: ${res.numInliers ?? 0}\n` +
            (res.homography ? `H: [${res.homography.map(v => v.toFixed(3)).join(', ')}]` : 'H: (none)');

        // Check if keypoints and matches are valid arrays
        if (!Array.isArray(keypointsA) || !Array.isArray(offsetKeypointsB) || !Array.isArray(res.matches)) {
            alert('No keypoints or matches found. Check your crop area and images.');
            return;
        }
        
        
        // Draw matches on full images using offset keypoints
        const A = matFromImageEl(imgA);
        const B = matFromImageEl(imgB);
        
        // Draw matches on full images using offset keypoints
        mod.drawMatches(
            A,                // full image A
            B,                // full image B
            keypointsA,       // use full image A keypoints
            offsetKeypointsB, // use offset keypoints for full image B
            res,              // match result
            source.imageSize  // original image A size for correct scaling
        );

        // Display matches on canvas
        imshowCompat(
            canvasMatches,     // canvas to draw on
            mod._lastCanvasMat // get last drawn matches Mat
        );

        imgB.hidden = true;              // hide original image B
        cropBoxB.style.display = 'none'; // hide crop box B
        canvasMatches.hidden = false;    // show matches canvas
        
        // Clean up
        A.delete();
        B.delete();
        mod._releaseLastCanvasMat();
    // Catch any errors during matching
    } catch (e) {
        console.error('Match error', e);
        alert('Match failed. See console.');
    // Cleanup
    } finally {
        target.delete();
        refreshButtons();
    }
});
