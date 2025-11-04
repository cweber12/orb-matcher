// orb_module.js
export class ORBModule {
  constructor(cv) { this.cv = cv; this._lastCanvasMat = null; }

  // Detect ORB on an RGBA image Mat
  detectORB(srcRGBA, opts = {}) {
    const cv = this.cv;
    const {
      nfeatures = 1200,
      scaleFactor = 1.2,
      nlevels = 8,
      edgeThreshold = 31,
      firstLevel = 0,
      WTA_K = 2,
      scoreType = cv.ORB_HARRIS_SCORE,
      patchSize = 31,
      fastThreshold = 20
    } = opts;

    // Gray
    const gray = new cv.Mat();
    cv.cvtColor(srcRGBA, gray, cv.COLOR_RGBA2GRAY);

    const orb = new cv.ORB(nfeatures, scaleFactor, nlevels, edgeThreshold, firstLevel, WTA_K, scoreType, patchSize, fastThreshold);
    const kpVec = new cv.KeyPointVector();
    const des = new cv.Mat();

    try {
      orb.detectAndCompute(gray, new cv.Mat(), kpVec, des);
      const keypoints = this._serializeKeypoints(kpVec);
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
  matchToTarget(sourceJson, targetMat, opts = {}) {
    const cv = this.cv;
    const ratio = opts.ratio ?? 0.75;
    const ransacThresh = opts.ransacReprojThreshold ?? 3.0;

    // --- Rehydrate source keypoints/descriptors from JSON ---
    const srcKP = new cv.KeyPointVector();
    for (const kp of sourceJson.keypoints) {
      // x,y,size,angle,response,octave, class_id (use defaults for missing)
      srcKP.push_back(new cv.KeyPoint(kp.x, kp.y, kp.size ?? 31, kp.angle ?? -1,
                                      kp.response ?? 0, kp.octave ?? 0, kp.class_id ?? -1));
    }
    const srcDesc = cv.matFromArray(
      sourceJson.descriptors.rows,
      sourceJson.descriptors.cols,
      cv.CV_8U,
      new Uint8Array(sourceJson.descriptors.data)
    );

    // --- Detect on target image (ORB) ---
    const gray = new cv.Mat();
    cv.cvtColor(targetMat, gray, cv.COLOR_RGBA2GRAY);
    const orb = new cv.ORB(this._nfeatures || 1200);
    const tgtKP = new cv.KeyPointVector();
    const tgtDesc = new cv.Mat();
    const empty = new cv.Mat();
    orb.detectAndCompute(gray, empty, tgtKP, tgtDesc, false);

    // --- KNN match (k=2) with ratio test ---
    const bf = new cv.BFMatcher(cv.NORM_HAMMING, false);
    const knn = new cv.DMatchVectorVector();
    bf.knnMatch(srcDesc, tgtDesc, knn, 2);

    const good = [];
    for (let i = 0; i < knn.size(); i++) {
      const vec = knn.get(i);            // DMatchVector (HAS delete)
      if (vec.size() >= 2) {
        const m = vec.get(0);            // DMatch (plain JS, NO delete)
        const n = vec.get(1);            // DMatch (plain JS, NO delete)
        if (m.distance < ratio * n.distance) good.push(m);
      }
      vec.delete();                      // ✅ delete the DMatchVector
    }
    knn.delete();                        // ✅ delete the outer vector

    // --- Optional: RANSAC homography (needs >= 4 matches) ---
    let H = null, inliers = 0;
    if (good.length >= 4) {
      const srcPts = new cv.Mat(good.length, 1, cv.CV_32FC2);
      const dstPts = new cv.Mat(good.length, 1, cv.CV_32FC2);
      for (let i = 0; i < good.length; i++) {
        const m = good[i];
        // queryIdx maps to source, trainIdx maps to target
        const s = srcKP.get(m.queryIdx).pt;
        const t = tgtKP.get(m.trainIdx).pt;
        srcPts.data32F[i*2]   = s.x; srcPts.data32F[i*2+1]   = s.y;
        dstPts.data32F[i*2]   = t.x; dstPts.data32F[i*2+1]   = t.y;
      }
      const mask = new cv.Mat();
      const Hmat = cv.findHomography(srcPts, dstPts, cv.RANSAC, ransacThresh, mask);
      if (!Hmat.empty()) {
        H = Array.from(Hmat.data64F ?? Hmat.data32F); // flatten for JSON/readout
        inliers = cv.countNonZero(mask);
      }
      srcPts.delete(); dstPts.delete(); mask.delete();
      Hmat.delete();
    }

    // --- Cleanup ---
    gray.delete();
    empty.delete();
    tgtDesc.delete();
    tgtKP.delete();
    orb.delete();
    bf.delete();
    srcDesc.delete();
    srcKP.delete();

    return { matches: good, homography: H, numInliers: inliers };
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
