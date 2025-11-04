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
  matchToTarget(featuresJSON, targetRGBA, options = {}) {
    const cv = this.cv;
    const src = this.importJSON(featuresJSON);
    // Compute ORB on target
    const detB = this.detectORB(targetRGBA, { nfeatures: 1200 });
    // Rewrap descriptors
    if (!src.descriptors || !detB.descriptors) return { matches: [] };

    const A = cv.matFromArray(src.descriptors.rows, src.descriptors.cols, cv.CV_8U, Array.from(src.descriptors.data));
    const B = cv.matFromArray(detB.descriptors.rows, detB.descriptors.cols, cv.CV_8U, Array.from(detB.descriptors.data));

    const matcher = new cv.BFMatcher(cv.NORM_HAMMING, false);
    const {
      useKnn = true,
      k = 2,
      ratio = 0.75,
      ransacReprojThreshold = 3.0,
      refineWithHomography = true
    } = options;

    const result = { matches: [], homography: null, inlierMask: null, numInliers: 0 };

    try {
      if (useKnn) {
        const knn = new cv.DMatchVectorVector();
        matcher.knnMatch(A, B, knn, Math.max(1, k));
        const good = [];
        for (let i = 0; i < knn.size(); i++) {
          const v = knn.get(i);
          if (v.size() >= 2) {
            const m0 = v.get(0), m1 = v.get(1);
            if (m0.distance <= ratio * m1.distance) {
              good.push({ queryIdx: m0.queryIdx, trainIdx: m0.trainIdx, distance: m0.distance });
            }
            m0.delete(); m1.delete();
          } else if (v.size() === 1) {
            const m0 = v.get(0);
            good.push({ queryIdx: m0.queryIdx, trainIdx: m0.trainIdx, distance: m0.distance });
            m0.delete();
          }
          v.delete();
        }
        knn.delete();
        good.sort((a,b)=>a.distance-b.distance);
        result.matches = good;
      } else {
        const mv = new cv.DMatchVector();
        matcher.match(A,B,mv);
        const arr = [];
        for (let i=0;i<mv.size();i++){ const m=mv.get(i); arr.push({queryIdx:m.queryIdx,trainIdx:m.trainIdx,distance:m.distance}); m.delete(); }
        mv.delete();
        arr.sort((a,b)=>a.distance-b.distance);
        result.matches = arr;
      }

      // RANSAC Homography refinement
      if (refineWithHomography && result.matches.length >= 4) {
        const pts1 = [], pts2 = [];
        for (const m of result.matches) {
          const p1 = src.keypoints[m.queryIdx];
          const p2 = detB.keypoints[m.trainIdx];
          if (p1 && p2) { pts1.push(p1.x, p1.y); pts2.push(p2.x, p2.y); }
        }
        if (pts1.length >= 8) {
          const mat1 = cv.matFromArray(pts1.length/2, 1, cv.CV_32FC2, pts1);
          const mat2 = cv.matFromArray(pts2.length/2, 1, cv.CV_32FC2, pts2);
          const inlierMask = new cv.Mat();
          const H = cv.findHomography(mat1, mat2, cv.RANSAC, ransacReprojThreshold, inlierMask);
          if (!H.empty()) {
            const hArr = H.data64F?.length ? Array.from(H.data64F) : Array.from(H.data32F);
            result.homography = hArr.slice(0,9);
            const maskBytes = new Uint8Array(inlierMask.data);
            result.inlierMask = maskBytes;
            result.numInliers = maskBytes.reduce((a,b)=>a+(b?1:0),0);
          }
          H.delete(); inlierMask.delete(); mat1.delete(); mat2.delete();
        }
      }

      // Stash mats for drawing
      this._lastDetB = detB;
      return result;
    } finally {
      matcher.delete(); A.delete(); B.delete();
    }
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
