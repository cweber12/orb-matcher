// orb_module.js
// ORB feature detection and matching
export class ORBModule {
  constructor(cv) { this.cv = cv; this._lastCanvasMat = null; }

  /*_________________________________________________________________________

  Public methods:
    - detectORB: detect ORB features in an image
    - exportJSON: export detected features to JSON
    - importJSON: import features from JSON
    - matchToTarget: match features to a target image
    - drawKeypoints: draw keypoints on a canvas
    - drawMatches: draw matches between two images
  __________________________________________________________________________*/

  // Detect ORB features in a cv.Mat image
  //________________________________________________________________________
  
  detectORB(srcRGBA, opts = {}) {
    const cv = this.cv; 
    
    // Default parameters
    const {
      nfeatures = 1200,   // Number of features to detect
      scaleFactor = 1.2,  // Pyramid scale factor
      nlevels = 8,        // Number of pyramid levels
      edgeThreshold = 31, // Size of the border where features are not detected
      firstLevel = 0,     // Level of pyramid to put source image to
      WTA_K = 2,          // Number of points that produce each element of ORB descriptor
      scoreType = cv.ORB_HARRIS_SCORE, // Score type (HARRIS or FAST)
      patchSize = 31,     // Size of the patch used by the oriented BRIEF descriptor
      fastThreshold = 20  // FAST threshold
    } = opts; 

    // 1. Create new Mats and ORB detector
    const gray = new cv.Mat();
    cv.cvtColor(srcRGBA, gray, cv.COLOR_RGBA2GRAY);

    // 2. Create ORB detector with specified parameters
    const orb = new cv.ORB(
      nfeatures, 
      scaleFactor, 
      nlevels, 
      edgeThreshold, 
      firstLevel, 
      WTA_K, 
      scoreType, 
      patchSize, 
      fastThreshold
    );

    // 3. Prepare keypoint vector and descriptor Mat
    const kpVec = new cv.KeyPointVector();
    const des = new cv.Mat();

    // 4. Perform detection and computation
    try {
      // 4.1 Detect and compute
      orb.detectAndCompute(
        gray,         // input image (grayscale)
        new cv.Mat(), // mask (none)
        kpVec,        // output keypoints
        des           // output descriptors
      );

      // 4.2 Serialize keypoints and descriptors
      const keypoints = this._serializeKeypoints(kpVec);
      const descriptors = this._serializeDescriptors(des);

      // 4.3 Return the results
      return { 
        keypoints,            // array of keypoints
        descriptors,          // descriptors Mat
        width: srcRGBA.cols,  // image width
        height: srcRGBA.rows  // image height
      };
    
    // 5. Clean up
    } finally {
      orb.delete(); kpVec.delete(); des.delete(); gray.delete();
    }
  }

  // Export a detect result to JSON (descriptors are base64)
  //__________________________________________________________________________

  exportJSON(detectResult) {  
    // 1. Get results from detectORB
    const { keypoints, descriptors, width, height } = detectResult;
    
    // 2. Normalize keypoints to [0,1] range
    const normKeypoints = keypoints.map(kp => ({
      ...kp,            // spread existing properties
      x: kp.x / width,  // normalize x coordinate
      y: kp.y / height  // normalize y coordinate
    }));

    // 3. Return JSON object
    return {
      version: 1,                                 // version number
      type: "ORB",                                // feature type                     
      imageSize: { width, height },               // original image size
      keypoints: normKeypoints,                   // normalized keypoints
      descriptors: descriptors ? {                // descriptors (base64 encoded)
        rows: descriptors.rows,                   //   - number of rows
        cols: descriptors.cols,                   //   - number of columns
        data_b64: this._u8ToB64(descriptors.data) //   - base64 data
      } : null                                    // null if no descriptors
    };
  }

  // Import JSON (reverse of export)
  //__________________________________________________________________________

  importJSON(obj) {  
    // 1. Validate input JSON
    if (!obj || obj.type !== "ORB") throw new Error("Invalid features JSON");
    
    // 2. Extract properties
    const { imageSize, keypoints, descriptors } = obj;
    
    // 3. Return the reconstructed detect result
    return {
      width: imageSize.width,                     // original image width
      height: imageSize.height,                   // original image height
      keypoints,                                  // keypoints (normalized)
      descriptors: descriptors ? {                // descriptors
        rows: descriptors.rows,                   //   - number of rows                   
        cols: descriptors.cols,                   //   - number of columns             
        data: this._b64ToU8(descriptors.data_b64) //   - Uint8Array data
      } : null                                    // null if no descriptors
    };
  }

  // Match JSON features (Image A) against a target Mat (Image B)
  //__________________________________________________________________________
  
  matchToTarget(sourceJson, targetMat, opts = {}) {  
    // 1. Access OpenCV.js
    const cv = this.cv; 

    // 2. Set matching parameters (set in UI, else defaults)
    const ratio = opts.ratio ?? 0.75;
    const ransacThresh = opts.ransacReprojThreshold ?? 3.0;

    // 3. Validate source JSON descriptors
    if (!sourceJson?.descriptors?.rows || !sourceJson?.descriptors?.cols) {
      throw new Error('Invalid features JSON: missing descriptors');
    }

    // Reconstruct source descriptors Mat
    const srcU8 = sourceJson.descriptors.data 
      // from existing data (Uint8Array)          
      ? new Uint8Array(sourceJson.descriptors.data)
      // or decode from base64      
      : this._b64ToU8(sourceJson.descriptors.data_b64); 

    // 5. Create cv.Mat for source descriptors
    const srcDesc = cv.matFromArray(
      sourceJson.descriptors.rows, // number of rows
      sourceJson.descriptors.cols, // number of columns
      cv.CV_8U,                    // type (unsigned 8-bit)
      srcU8                        // data buffer (Uint8Array)
    );

    // 6. Convert target image to grayscale
    const gray = new cv.Mat(); // Grayscale Mat
    cv.cvtColor (              // Convert to grayscale
      targetMat,          // source Mat
      gray,               // destination Mat
      cv.COLOR_RGBA2GRAY  // color conversion code
    );

    // 7. Set up ORB detector
    const orb = new cv.ORB(this._nfeatures || 1200); // ORB detector
    const tgtKP = new cv.KeyPointVector();           // Target keypoints
    const tgtDesc = new cv.Mat();                    // Target descriptors
    const empty = new cv.Mat();                      // Empty mask
    
    // 8. Detect and compute on target image
    orb.detectAndCompute(gray, empty, tgtKP, tgtDesc, false);

    // 9. KNN match (k=2) + ratio test
    const bf = new cv.BFMatcher(  // Brute-Force matcher
      cv.NORM_HAMMING,            // Hamming distance
      false                       // crossCheck disabled        
    ); 
    const knn = new cv.DMatchVectorVector(); // Init KNN matches
    bf.knnMatch(srcDesc, tgtDesc, knn, 2);   // Match descriptors

    // 10. Initialize array for good matches
    const good = [];

    // 11. Iterate through KNN matches and apply ratio test
    for (let i = 0; i < knn.size(); i++) {
      const vec = knn.get(i); // DMatchVector (needs delete)
      // If there are at least 2 matches
      if (vec.size() >= 2) {  
        const m = vec.get(0); // get first (best) match  
        const n = vec.get(1); // get second match
        // If the the distance is withing the ratio threshold
        // push to good matches
        if (m.distance < ratio * n.distance) good.push(m);
      }
      vec.delete(); // cleanup DMatchVector
    }
    knn.delete(); // cleanup KNN matches

    // 12. Estimate homography using RANSAC if enough good matches
    let H = null;          // Homography matrix
    let inliers = 0;       // Number of inliers
    let inlierMask = null; // Inlier mask array

    // 13. If at least 4 good matches, compute homography
    if (good.length >= 4) {
      
      // Prepare array for source points
      const srcPts = new cv.Mat(
        good.length,  // number of points
        1,            // single column
        cv.CV_32FC2   // type (2-channel float32
      );

      // Prepare array for destination points
      const dstPts = new cv.Mat(
        good.length,  // number of points
        1,            // single column
        cv.CV_32FC2   // type (2-channel float32
      );

      // Fill point arrays based on good matches
      const srcW = sourceJson.imageSize?.width  ?? targetMat.cols;
      const srcH = sourceJson.imageSize?.height ?? targetMat.rows;

      for (let i = 0; i < good.length; i++) {
        const m = good[i]; // current match
        // normalized source keypoint from JSON (image A)
        const sN = sourceJson.keypoints[m.queryIdx];
        // target keypoint from detected KPs (image B)
        const t  = tgtKP.get(m.trainIdx).pt;

        // de-normalize to pixel coords on source image A
        const sx = sN.x * srcW; // de-normalized x
        const sy = sN.y * srcH; // de-normalized y
 
        // Set coordinates for ith point in source and destination Mats
        srcPts.data32F[i*2]   = sx;  // source x
        srcPts.data32F[i*2+1] = sy;  // source y
        dstPts.data32F[i*2]   = t.x; // destination x
        dstPts.data32F[i*2+1] = t.y; // destination y
      }
      
      // Prepare mask for inliers
      const mask = new cv.Mat();

      // Compute homography using RANSAC
      const Hmat = cv.findHomography(
        srcPts,       // source points
        dstPts,       // destination points
        cv.RANSAC,    // method (RANSAC)
        ransacThresh, // RANSAC reprojection threshold
        mask          // output mask
      );
      // If homography is found, extract data
      if (!Hmat.empty()) {
        // Convert homography Mat to JS array
        H = Array.from(Hmat.data64F ?? Hmat.data32F);
        // Count inliers from mask
        inliers = cv.countNonZero(mask);
        // Create inlier mask array (boolean)
        inlierMask = Array.from(mask.data).map(v => v !== 0);
      }
      // Cleanup
      srcPts.delete(); // source points Mat
      dstPts.delete(); // destination points Mat
      mask.delete();   // inlier mask Mat
      Hmat.delete();   // homography Mat
    }

    // 14. Cache target KPs for drawMatches
    const tgtKP_JS = this._serializeKeypoints(tgtKP);
    this._lastDetB = { keypoints: tgtKP_JS };

    // 15. Cleanup
    gray.delete();     // grayscale image
    empty.delete();    // empty mask
    tgtDesc.delete();  // target descriptors
    tgtKP.delete();    // target keypoints
    orb.delete();      // ORB detector
    bf.delete();       // brute force matcher
    srcDesc.delete();  // source descriptors

    // 16. Return match results
    return { 
      matches: good,       // array of good matches
      homography: H,       // homography array (or null)
      numInliers: inliers, // number of inliers
      inlierMask           // inlier mask array (or null)
    };
  }

  // Draw keypoints on canvas
  //__________________________________________________________________________

  drawKeypoints(imgRGBA, keypoints, outCanvas) {
    const cv = this.cv; // OpenCV.js
    
    // Set output canvas size
    outCanvas.width = imgRGBA.cols;   
    outCanvas.height = imgRGBA.rows;
    
    // Initialize output Mat
    const out = new cv.Mat( 
      imgRGBA.rows,   // height
      imgRGBA.cols,   // width
      cv.CV_8UC4  ,   // type
    );
    
    imgRGBA.copyTo(out); // copy source image

    // Loop through keypoints and draw circles
    for (const kp of keypoints) {
      // Draw green circle at keypoint location
      cv.circle(
        out,                        // target Mat
        new cv.Point(               // center point
          Math.round(kp.x),         //   - x coordinate
          Math.round(kp.y)          //   - y coordinate
        ), 
        3,                          // radius 
        new cv.Scalar(0,255,0,255), // color (green) 
        -1,                         // filled circle
        cv.LINE_AA                  // line type (antialiased)
      );
    }
    cv.imshow(outCanvas, out); // display on canvas
    out.delete();              // cleanup
  }

  // Draw matches side-by-side (A|B) with inliers in green, others red
  //__________________________________________________________________________

  drawMatches(imgA, imgB, keypointsA, keypointsB, matchRes, originalSizeA) {
    
    const cv = this.cv;                           // OpenCV.js
    const outH = Math.max(imgA.rows, imgB.rows);  // Output height
    const outW = imgA.cols + imgB.cols;           // Output width
    this._releaseLastCanvasMat();                 // Release previous if any
    
    // Create new output Mat
    this._lastCanvasMat = new cv.Mat(
      outH,                     // output height     
      outW,                     // output width
      cv.CV_8UC4,               // type
      new cv.Scalar(0,0,0,255)  // black background
    );

    // Region of Interest (ROI) for image A
    const roiA = this._lastCanvasMat.roi(
      new cv.Rect(
        0,          // init x (left)
        0,          // init y (top)
        imgA.cols,  // width
        imgA.rows   // height
      )
    );
    imgA.copyTo(roiA);  // copy image A into ROI
    roiA.delete();      // release ROI

    // ROI for image B
    const roiB = this._lastCanvasMat.roi(
      new cv.Rect(
        imgA.cols,  // init x (after image A)
        0,          // init y (top)
        imgB.cols,  // width
        imgB.rows   // height
      )
    );
    imgB.copyTo(roiB); // copy image B into ROI
    roiB.delete();     // release ROI
    
    // Determine inlier mask
    const inMask = matchRes.inlierMask;

    // Loop through matches and draw lines + circles
    for (let i = 0; i < matchRes.matches.length; i++) {
      
      const m = matchRes.matches[i];      // current match
      const p1 = keypointsA[m.queryIdx];  // point from image A
      const p2 = keypointsB[m.trainIdx];  // point from image B
      if (!p1 || !p2) continue;           // sanity check
      
      // Determine if inlier
      const inlier = inMask ? Boolean(inMask[i]) : true;
      // Determine color (green for inlier, red for outlier)
      const color = inlier ? new cv.Scalar(0,255,0,255) : new cv.Scalar(255,0,0,255);

      // Denormalize keypointsA (from normalized [0,1] to pixel coordinates)
      // Use originalSizeA for denormalization
      const a = new cv.Point(
        Math.round(p1.x * imgA.cols),
        Math.round(p1.y * imgA.rows)
      );

      // keypointsB are already in pixel coordinates
      const b = new cv.Point(
        Math.round(p2.x + imgA.cols), // x coordinate (offset by image A width)
        Math.round(p2.y)              // y coordinate
      );

      /* ---------------------------------------------------------------------
      
      DIAGRMA FOR OUTPUT IMAGE LAYOUT: 

            {   imgA.cols       }{         imgB.cols      }
      (0,0) -----------------------------------------------
            |                  >|                        >|<
            |                >  |                      >  |  <
            |    imgA.rows -->  |          imgB.rows -->  |  <-- outH
            |                  >|                      >  |  <
    empty   --------------------|                        >|<
    space ->////////////////////---------------------------
            {                   outW                      }

      ---------------------------------------------------------------------- */

      // Draw line between matched keypoints
      cv.line(this._lastCanvasMat, a, b, color, 1, cv.LINE_AA);
      // Draw circles at keypoints
      cv.circle(this._lastCanvasMat, a, 3, color, -1, cv.LINE_AA);
      cv.circle(this._lastCanvasMat, b, 3, color, -1, cv.LINE_AA);
    }
  }

  // Internal: serialize KeyPointVector to JS array
  _serializeKeypoints(kpVec) {
    const n = kpVec.size(); // number of keypoints
    const out = [];         // output array  
    // Loop through keypoints
    for (let i=0; i < n; i++) {       
      const k = kpVec.get(i); // get KeyPoint   
      out.push({              // push serialized object
        x:k.pt.x,                 // point coordinates  
        y:k.pt.y,                 // point coordinates
        size:k.size,              // diameter of the meaningful keypoint area
        angle:k.angle,            // orientation
        response:k.response,      // response strength
        octave:k.octave,          // octave level
        class_id:k.class_id ?? -1 // class_id may be undefined 
      });
    }
    return out; // return array of keypoints
  }

  /*_________________________________________________________________________
  
  Internal methods for: 
    - base64 <-> Uint8Array conversions
    - releasing cached Mats
  __________________________________________________________________________*/
  
  // Internal: serialize descriptor Mat to JS object
  _serializeDescriptors(des) {
    // Check for empty descriptors
    if (!des || des.rows===0 || des.cols===0) return null;
    // Return serialized descriptor object
    return { 
      rows: des.rows,                 // number of rows 
      cols: des.cols,                 // number of columns
      data: new Uint8Array(des.data)  // copy data to Uint8Array
    };
  }
  
  // Internal: base64 <-> Uint8Array conversions
  _u8ToB64(u8) {
    
    let binary = '';      // binary string
    const chunk = 0x8000; // chunk size for processing
    // Iterate through Uint8Array in chunks
    for (let i=0;i<u8.length;i+=chunk) {
      // Convert each chunk to binary string
      binary += String.fromCharCode.apply(null, u8.subarray(i,i+chunk));
    }
    // Encode binary string to base64
    return btoa(binary);
  }

  // Internal: base64 to Uint8Array
  _b64ToU8(b64) {
    const bin = atob(b64); // decode base64 to binary string
    const u8 = new Uint8Array(bin.length); // create Uint8Array
    // Iterate through binary string and fill Uint8Array
    for (let i=0;i<bin.length;i++) {
      // Convert each character to its char code
      u8[i]=bin.charCodeAt(i);
    } 
    // Return the Uint8Array
    return u8;
  }

  // Internal: release last cached canvas Mat
  _releaseLastCanvasMat(){
    // If there is a cached Mat
    if (this._lastCanvasMat) { 
      this._lastCanvasMat.delete(); // delete it
      this._lastCanvasMat=null;     // clear reference
    } 
  }
}
