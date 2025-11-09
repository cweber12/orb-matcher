// video_frame_extractor.js
// Extract frames from a video element into a canvas
export class VideoFrameExtractor {
  constructor(videoEl, canvasEl) {
    this.videoEl = videoEl;
    this.canvasEl = canvasEl;
  }

  // Extract a specific frame (by frame number) at given fps
  async extractFrame(frameNumber, fps = 25) {
    // Return a promise that resolves when the frame is extracted
    return new Promise((resolve, reject) => {
      if (!this.videoEl.src) return reject('No video loaded');
      const time = frameNumber / fps; // Calculate time in seconds
      this.videoEl.currentTime = time; // Seek to the time
      this.videoEl.onseeked = () => { // When seeked, draw the frame to canvas
        this.canvasEl.width = this.videoEl.videoWidth; // Set canvas width to video width
        this.canvasEl.height = this.videoEl.videoHeight; // Set canvas height to video height
        const ctx = this.canvasEl.getContext('2d'); // Get canvas context

        // Draw the video frame onto the canvas
        ctx.drawImage(this.videoEl, 0, 0, this.canvasEl.width, this.canvasEl.height);
        this.canvasEl.hidden = false; // Unhide the canvas
        resolve(this.canvasEl); // Resolve with the canvas element
      };
      // Handle video load errors
      this.videoEl.onerror = reject;
    });
  }
}