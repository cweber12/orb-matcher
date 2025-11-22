# ORB Feature Tool

Small browser tool that detects ORB features in Image A (or a cropped frame) using OpenCV.js and matches them to Image B. Designed for quick experimentation and JSON export/import of features.

## Features
- Detect ORB keypoints and descriptors in the browser (OpenCV.js)
- Export features.json (keypoints normalized, descriptors base64)
- Import features.json and match to Image B with ratio test + RANSAC
- Side-by-side match visualization with inliers highlighted

## Prerequisites
- Modern browser with ES modules (Chrome/Edge/Firefox)
- Local static server (recommended) — files must be served over HTTP, not file://

Example quick server:
```bash
# from project root
python -m http.server 8000
# or
npx http-server -p 8000
```

Open http://localhost:8000 in your browser.

## Files of interest
- `index.html` — UI and OpenCV.js bootstrap
- `main.js` — app logic, UI handlers
- `orb_module.js` — ORB detection, export/import, matching, drawing
- `image_utils.js` — image loading, cropping, Mat conversion
- `setup_crop_box.js` — crop box drag/resize handlers
- `styles.css` — styles

## Quick usage
1. Start a local server and open the app.
2. Upload Image A.
3. Adjust crop if needed, click **Detect ORB**.
4. Inspect keypoints; click **Download features.json** to save normalized features.
5. Upload Image B (and/or load previously saved `features.json`).
6. Click **Match** to run matching and view matches in the canvas.

## Data format notes
- Exported JSON stores keypoints normalized to `[0..1]` relative to original image size:
  - To draw/use on a displayed image, multiply by the rendered image pixel dimensions.
- Descriptors are stored as base64 in `descriptors.data_b64`.


## Developer tips
- `exportJSON(detectResult)` normalizes keypoints. `drawMatches` must denormalize using the actual Mat dimensions used for drawing.
- Use browser DevTools console to inspect `imgA.naturalWidth`, `imgA.getBoundingClientRect()`, and the JSON `imageSize` when diagnosing coordinate issues.
- Free OpenCV.js memory (`Mat.delete()`) to avoid leaks.

