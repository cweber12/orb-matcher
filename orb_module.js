// orb_module.js
// ORB feature detection and matching using OpenCV.js
export class ORBModule {
  constructor(cv) { this.cv = cv; this._lastCanvasMat = null; }

  // Detect ORB on an RGBA image Mat
  detectORB(srcRGBA, opts = {}) {
    const cv = this.cv; 
    // Default parameters
    const {
      nfeatures = 1200, // Number of features to detect
      scaleFactor = 1.2, // Pyramid scale factor
      nlevels = 8, // Number of pyramid levels
      edgeThreshold = 31, // Size of the border where features are not detected
      firstLevel = 0, // Level of pyramid to put source image to
      WTA_K = 2, // Number of points that produce each element of the oriented BRIEF descriptor
      scoreType = cv.ORB_HARRIS_SCORE, // Score type (HARRIS or FAST)
      patchSize = 31, // Size of the patch used by the oriented BRIEF descriptor
      fastThreshold = 20 // FAST threshold
    } = opts; 

    // Create new Mats and ORB detector
    const gray = new cv.Mat();
    cv.cvtColor(srcRGBA, gray, cv.COLOR_RGBA2GRAY);

    // Create ORB detector with specified parameters
    const orb = new cv.ORB(nfeatures, scaleFactor, nlevels, edgeThreshold, firstLevel, WTA_K, scoreType, patchSize, fastThreshold);
    // Detect keypoints and compute descriptors
    const kpVec = new cv.KeyPointVector();
    const des = new cv.Mat();

    // Perform detection and computation
    try {
      // Detect and compute
      orb.detectAndCompute(gray, new cv.Mat(), kpVec, des);
      // Serialize keypoints and descriptors
      const keypoints = this._serializeKeypoints(kpVec);
      // Serialize descriptors
      const descriptors = this._serializeDescriptors(des);
      return { keypoints, descriptors, width: srcRGBA.cols, height: srcRGBA.rows };
    } finally {
      orb.delete(); kpVec.delete(); des.delete(); gray.delete();
    }
  }

  // Export a detect result to JSON (descriptors are base64)
  exportJSON(detectResult) {
    const { keypoints, descriptors, width, height } = detectResult;
    return {
      version: 1,
      type: "ORB",
      imageSize: { width, height },
      keypoints,
      descriptors: descriptors ? {
        rows: descriptors.rows,
        cols: descriptors.cols,
        data_b64: this._u8ToB64(descriptors.data)
      } : null
    };
  }

  // Import JSON (reverse of export)
  importJSON(obj) {
    if (!obj || obj.type !== "ORB") throw new Error("Invalid features JSON");
    const { imageSize, keypoints, descriptors } = obj;
    return {
      width: imageSize.width,
      height: imageSize.height,
      keypoints,
      descriptors: descriptors ? {
        rows: descriptors.rows,
        cols: descriptors.cols,
        data: this._b64ToU8(descriptors.data_b64)
      } : null
    };
  }

  // Match JSON features (Image A) against a target Mat (Image B)
 // Match JSON features (Image A) against a target Mat (Image B)
matchToTarget(sourceJson, targetMat, opts = {}) {
  const cv = this.cv;
  const ratio = opts.ratio ?? 0.75;
  const ransacThresh = opts.ransacReprojThreshold ?? 3.0;

  // --- Rebuild descriptors from JSON (no cv.KeyPoint construction) ---
  if (!sourceJson?.descriptors?.rows || !sourceJson?.descriptors?.cols) {
    throw new Error('Invalid features JSON: missing descriptors');
  }
  const srcU8 = sourceJson.descriptors.data
    ? new Uint8Array(sourceJson.descriptors.data)
    : this._b64ToU8(sourceJson.descriptors.data_b64);

  const srcDesc = cv.matFromArray(
    sourceJson.descriptors.rows,
    sourceJson.descriptors.cols,
    cv.CV_8U,
    srcU8
  );

  // --- Detect on target (ORB) ---
  const gray = new cv.Mat();
  cv.cvtColor(targetMat, gray, cv.COLOR_RGBA2GRAY);
  const orb = new cv.ORB(this._nfeatures || 1200);
  const tgtKP = new cv.KeyPointVector();
  const tgtDesc = new cv.Mat();
  const empty = new cv.Mat();
  orb.detectAndCompute(gray, empty, tgtKP, tgtDesc, false);

  // --- KNN match (k=2) + ratio test ---
  const bf = new cv.BFMatcher(cv.NORM_HAMMING, false);
  const knn = new cv.DMatchVectorVector();
  bf.knnMatch(srcDesc, tgtDesc, knn, 2);

  const good = [];
  for (let i = 0; i < knn.size(); i++) {
    const vec = knn.get(i);              // DMatchVector (needs delete)
    if (vec.size() >= 2) {
      const m = vec.get(0);              // DMatch (plain)
      const n = vec.get(1);
      if (m.distance < ratio * n.distance) good.push(m);
    }
    vec.delete();
  }
  knn.delete();

  // --- Homography with RANSAC (>=4 matches) ---
  let H = null, inliers = 0, inlierMask = null;
  if (good.length >= 4) {
    const srcPts = new cv.Mat(good.length, 1, cv.CV_32FC2);
    const dstPts = new cv.Mat(good.length, 1, cv.CV_32FC2);
    for (let i = 0; i < good.length; i++) {
      const m = good[i];
      // source uses JSON keypoints:
      const s = sourceJson.keypoints[m.queryIdx];   // {x,y}
      // target uses detector KeyPointVector:
      const t = tgtKP.get(m.trainIdx).pt;           // Point2f
      srcPts.data32F[i*2]   = s.x; srcPts.data32F[i*2+1]   = s.y;
      dstPts.data32F[i*2]   = t.x; dstPts.data32F[i*2+1]   = t.y;
    }
    const mask = new cv.Mat();
    const Hmat = cv.findHomography(srcPts, dstPts, cv.RANSAC, ransacThresh, mask);
    if (!Hmat.empty()) {
      H = Array.from(Hmat.data64F ?? Hmat.data32F);
      inliers = cv.countNonZero(mask);
      inlierMask = Array.from(mask.data).map(v => v !== 0);
    }
    srcPts.delete(); dstPts.delete(); mask.delete(); Hmat.delete();
  }

  // Cache target KPs for drawMatches
  const tgtKP_JS = this._serializeKeypoints(tgtKP);
  this._lastDetB = { keypoints: tgtKP_JS };

  // Cleanup
  gray.delete(); empty.delete(); tgtDesc.delete(); tgtKP.delete(); orb.delete(); bf.delete(); srcDesc.delete();

  return { matches: good, homography: H, numInliers: inliers, inlierMask };
}



  // Draw keypoints on canvas
  drawKeypoints(imgRGBA, keypoints, outCanvas) {
    const cv = this.cv;
    outCanvas.width = imgRGBA.cols; outCanvas.height = imgRGBA.rows;
    const out = new cv.Mat(imgRGBA.rows, imgRGBA.cols, cv.CV_8UC4);
    imgRGBA.copyTo(out);
    for (const kp of keypoints) {
      cv.circle(out, new cv.Point(Math.round(kp.x), Math.round(kp.y)), 3, new cv.Scalar(0,255,0,255), -1, cv.LINE_AA);
    }
    cv.imshow(outCanvas, out);
    out.delete();
  }

  // Draw matches side-by-side (A|B) with inliers in green, others red
  drawMatches(imgA, imgB, keypointsA, matchRes) {
    const cv = this.cv;
    const outH = Math.max(imgA.rows, imgB.rows);
    const outW = imgA.cols + imgB.cols;
    this._releaseLastCanvasMat();
    this._lastCanvasMat = new cv.Mat(outH, outW, cv.CV_8UC4, new cv.Scalar(0,0,0,255));

    const roiA = this._lastCanvasMat.roi(new cv.Rect(0, 0, imgA.cols, imgA.rows)); imgA.copyTo(roiA); roiA.delete();
    const roiB = this._lastCanvasMat.roi(new cv.Rect(imgA.cols, 0, imgB.cols, imgB.rows)); imgB.copyTo(roiB); roiB.delete();

    const inMask = matchRes.inlierMask;
    for (let i = 0; i < matchRes.matches.length; i++) {
      const m = matchRes.matches[i];
      const p1 = keypointsA[m.queryIdx];
      const p2 = this._lastDetB.keypoints[m.trainIdx];
      if (!p1 || !p2) continue;
      const inlier = inMask ? Boolean(inMask[i]) : true;
      const color = inlier ? new cv.Scalar(0,255,0,255) : new cv.Scalar(255,0,0,255);
      const a = new cv.Point(Math.round(p1.x), Math.round(p1.y));
      const b = new cv.Point(Math.round(p2.x + imgA.cols), Math.round(p2.y));
      cv.line(this._lastCanvasMat, a, b, color, 1, cv.LINE_AA);
      cv.circle(this._lastCanvasMat, a, 3, color, -1, cv.LINE_AA);
      cv.circle(this._lastCanvasMat, b, 3, color, -1, cv.LINE_AA);
    }
  }

  _serializeKeypoints(kpVec) {
    const n = kpVec.size(), out = [];
    for (let i=0;i<n;i++){
      const k = kpVec.get(i);
      out.push({ x:k.pt.x, y:k.pt.y, size:k.size, angle:k.angle, response:k.response, octave:k.octave, class_id:k.class_id ?? -1 });
    }
    return out;
  }
  _serializeDescriptors(des) {
    if (!des || des.rows===0 || des.cols===0) return null;
    return { rows: des.rows, cols: des.cols, data: new Uint8Array(des.data) };
    // Note: for JSON export we convert to base64.
  }
  _u8ToB64(u8) {
    let binary = ''; const chunk = 0x8000;
    for (let i=0;i<u8.length;i+=chunk) binary += String.fromCharCode.apply(null, u8.subarray(i,i+chunk));
    return btoa(binary);
  }
  _b64ToU8(b64) {
    const bin = atob(b64); const u8 = new Uint8Array(bin.length);
    for (let i=0;i<bin.length;i++) u8[i]=bin.charCodeAt(i);
    return u8;
  }
  _releaseLastCanvasMat(){ if (this._lastCanvasMat){ this._lastCanvasMat.delete(); this._lastCanvasMat=null; } }
}
