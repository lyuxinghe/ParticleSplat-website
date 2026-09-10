/* Only load the renderer and scene after an explicit request. */
(() => {
  const root = document.querySelector('.particle-demo');
  const button = document.querySelector('#demo-load');
  if (!root || !button) return;
  button.hidden = false;
  button.addEventListener('click', async () => {
    if (root.dataset.state === 'loading') return;
    root.dataset.state = 'loading';
    button.disabled = true;
    button.textContent = 'Loading scene…';
    const status = document.querySelector('#demo-status');
    status.textContent = 'Preparing the browser renderer…';
    try {
      const { mountDemo } = await import('./particle-demo.js?v=8');
      await mountDemo(root);
    } catch (error) {
      root.dataset.state = 'error';
      status.textContent = 'The 3D demo could not load. Check your connection and use a WebGL2 browser with hardware acceleration, then retry. The reconstruction preview remains available.';
      button.textContent = 'Try loading again';
      button.disabled = false;
      console.warn('Particle demo unavailable:', error.message);
    }
  });
})();
