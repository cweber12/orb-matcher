// setup_crop_box.js
// Setup drag and resize functionality for a crop box over an image

export function setupCropBox(imgEl, cropBoxEl) {
      let isDragging = false, isResizing = false, resizeCorner = null;
      let startX, startY, startLeft, startTop, startW, startH;

      // Mouse down on crop box to start dragging (only if not on a handle)
      cropBoxEl.addEventListener('mousedown', (e) => {
        if (e.target.classList.contains('resize-handle')) return;
        isDragging = true;
        startX = e.clientX;
        startY = e.clientY;
        startLeft = parseInt(cropBoxEl.style.left, 10);
        startTop = parseInt(cropBoxEl.style.top, 10);
        e.preventDefault();
      });

      // Mouse down on resize handles to start resizing
      cropBoxEl.querySelectorAll('.resize-handle').forEach(handle => {
        handle.addEventListener('mousedown', (e) => {
          isResizing = true;
          resizeCorner = handle.dataset.corner;
          startX = e.clientX;
          startY = e.clientY;
          startW = parseInt(cropBoxEl.style.width, 10);
          startH = parseInt(cropBoxEl.style.height, 10);
          startLeft = parseInt(cropBoxEl.style.left, 10);
          startTop = parseInt(cropBoxEl.style.top, 10);
          e.stopPropagation();
          e.preventDefault();
        });
      });

      // Mouse move to drag or resize crop box
      document.addEventListener('mousemove', (e) => {
        const imgRect = imgEl.getBoundingClientRect();
        if (isDragging) {
          let dx = e.clientX - startX;
          let dy = e.clientY - startY;
          let newLeft = startLeft + dx;
          let newTop = startTop + dy;
          newLeft = Math.max(0, Math.min(newLeft, imgRect.width - parseInt(cropBoxEl.style.width, 10)));
          newTop = Math.max(0, Math.min(newTop, imgRect.height - parseInt(cropBoxEl.style.height, 10)));
          cropBoxEl.style.left = `${newLeft}px`;
          cropBoxEl.style.top = `${newTop}px`;
        } else if (isResizing) {
          let dx = e.clientX - startX;
          let dy = e.clientY - startY;
          let left = startLeft, top = startTop, width = startW, height = startH;
          if (resizeCorner === 'nw') {
            left = startLeft + dx;
            top = startTop + dy;
            width = startW - dx;
            height = startH - dy;
          } else if (resizeCorner === 'ne') {
            top = startTop + dy;
            width = startW + dx;
            height = startH - dy;
          } else if (resizeCorner === 'sw') {
            left = startLeft + dx;
            width = startW - dx;
            height = startH + dy;
          } else if (resizeCorner === 'se') {
            width = startW + dx;
            height = startH + dy;
          }
          left = Math.max(0, Math.min(left, imgRect.width - 10));
          top = Math.max(0, Math.min(top, imgRect.height - 10));
          width = Math.max(10, Math.min(width, imgRect.width - left));
          height = Math.max(10, Math.min(height, imgRect.height - top));
          cropBoxEl.style.left = `${left}px`;
          cropBoxEl.style.top = `${top}px`;
          cropBoxEl.style.width = `${width}px`;
          cropBoxEl.style.height = `${height}px`;
        }
      });

      document.addEventListener('mouseup', () => {
        isDragging = false;
        isResizing = false;
        resizeCorner = null;
      });

      // Initialize crop box when image loads
      imgEl.onload = () => {
        imgEl.hidden = false;
        const imgRect = imgEl.getBoundingClientRect();
        const parent = imgEl.parentElement;
        parent.style.width = imgRect.width + 'px';
        parent.style.height = imgRect.height + 'px';
        cropBoxEl.style.display = 'block';
        cropBoxEl.style.left = '0px';
        cropBoxEl.style.top = '0px';
        cropBoxEl.style.width = imgRect.width + 'px';
        cropBoxEl.style.height = imgRect.height + 'px';
      };
    }