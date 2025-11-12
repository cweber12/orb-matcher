// image_utils.js
// Utilities for loading images and converting to cv.Mat

// Load an image file into an HTMLImageElement
export function loadImg(file, imgEl, cropBox) {
    // Return a promise that resolves when the image is loaded
    return new Promise((res, rej) => {
    const r = new FileReader(); // FileReader to read the file
    r.onload = () => {
        imgEl.onload = () => {
        imgEl.hidden = false; // show image element
        // Crop box initialization
        const imgRect = imgEl.getBoundingClientRect(); // get image A bounding rect 
        cropBox.style.display = 'block'; // show crop box
        cropBox.style.left = '0px'; // align to left
        cropBox.style.top = '0px'; // align to top
        cropBox.style.width = imgRect.width + 'px'; // image width
        cropBox.style.height = imgRect.height + 'px'; // image height
        res(); 
        };
        imgEl.onerror = rej;
        imgEl.src = r.result;
    };
    r.onerror = rej; // onerror handler
    r.readAsDataURL(file); // read file as data URL
    });
}

// Temporary offscreen canvas for image to Mat conversion
const __tmpCanvas = document.createElement('canvas');
// Context for the temporary canvas
const __tmpCtx = __tmpCanvas.getContext('2d', { willReadFrequently: true });

// Convert an HTMLImageElement to cv.Mat (CV_8UC4)
export function matFromImageEl(imgEl) {
    // Get image dimensions, use natural size if available
    const w = imgEl.naturalWidth || imgEl.width;
    const h = imgEl.naturalHeight || imgEl.height;
    // Draw the <img> to an offscreen canvas and grab RGBA bytes
    __tmpCanvas.width = w; // set canvas to image width
    __tmpCanvas.height = h; // set canvas to image height
    __tmpCtx.clearRect(0, 0, w, h); // clear canvas
    __tmpCtx.drawImage(imgEl, 0, 0, w, h); // draw image to canvas
    const imageData = __tmpCtx.getImageData(0, 0, w, h); // get RGBA pixel data

    // Allocate CV_8UC4 without using 'new cv.Mat(...)'
    const mat = window.cv.Mat.zeros(h, w, window.cv.CV_8UC4);
    mat.data.set(imageData.data);   // copy RGBA buffer in
    return mat; // return CV_8UC4 Mat
}

// cropRect: { x, y, width, height }
export function cropImage(imgEl, cropRect) {
    // Create a canvas to hold the cropped image
    const cropCanvas = document.createElement('canvas');
    // Set the canvas size to the crop rectangle size
    cropCanvas.width = cropRect.width; 
    cropCanvas.height = cropRect.height;
    // Get the 2D context and draw the cropped area
    const ctx = cropCanvas.getContext('2d');
    ctx.drawImage(
    imgEl, // Source image element
    cropRect.x, cropRect.y, cropRect.width, cropRect.height, // Source rectangle
    0, 0, cropRect.width, cropRect.height // Destination rectangle
    );
    // Return the cropped canvas
    return cropCanvas;
}
