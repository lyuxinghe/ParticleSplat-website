import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { TransformControls } from 'three/addons/controls/TransformControls.js';
import { SparkRenderer, SplatMesh } from '@sparkjsdev/spark';

const sceneURL = new URL('../scenes/close-jar/scene.json?v=8', import.meta.url);
const vec = (array) => new THREE.Vector3().fromArray(array);
const clamp = THREE.MathUtils.clamp;

class DemoSparkRenderer extends SparkRenderer {
  async driveSort() {
    try {
      return await super.driveSort();
    } catch (error) {
      // Three's asynchronous GPU fence rejects with no reason on WAIT_FAILED.
      // A lost context cancels an in-flight Spark sort; the poster fallback
      // handles that lifecycle event. Propagate every other rendering error.
      if (error !== undefined || !this.renderer.getContext().isContextLost()) throw error;
    }
  }
}

export async function mountDemo(root) {
  const $ = (selector) => root.querySelector(selector);
  const viewport = $('#particle-viewport');
  const panel = $('#demo-editor');
  const handleLayer = $('#demo-handles');
  const status = $('#demo-status');
  const listeners = new AbortController();
  const on = (element, event, callback) => element.addEventListener(event, callback, { signal: listeners.signal });
  const canvas = document.createElement('canvas');
  canvas.tabIndex = 0;
  canvas.setAttribute('aria-label', '3D reconstruction. Drag to orbit, right-drag to pan, and scroll to zoom. Tab to an editable region to open its controls.');
  const context = canvas.getContext('webgl2', { antialias: false, alpha: false });
  if (!context) throw new Error('WebGL2 is not available');
  const renderer = new THREE.WebGLRenderer({ canvas, context, antialias: false });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.5));
  renderer.setClearColor('#000000');
  const world = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(60, 1, 0.005, 200);
  camera.up.set(0, 0, 1);
  // Full covariance transforms preserve anisotropic X/Y/Z edits (A Σ Aᵀ).
  // Spark's default Gsplat transform averages the three scale components.
  const spark = new DemoSparkRenderer({ renderer, accumExtSplats: true, covSplats: true, sortRadial: false, preBlurAmount: 0.3, blurAmount: 0, maxStdDev: 3 });
  world.add(spark);
  const loadedMeshes = [];
  let orbit, gizmo, observer, resizeObserver;
  try {
    const response = await fetch(sceneURL, { cache: 'no-cache' });
    if (!response.ok) throw new Error(`Scene manifest: HTTP ${response.status}`);
    const data = await response.json();
    const nodeById = new Map();
    const groupById = new Map();
    const handles = new Map();
    let selected = null;
    let drag = null;
    let inView = true;
    let disposed = false;
    let showHandles = true;
    let lastInteraction = 0;
    let framing = { center: [...data.framing.center], height: data.framing.height };
    const raycaster = new THREE.Raycaster();
    const pointer = new THREE.Vector2();
    const invalidate = () => { lastInteraction = performance.now(); };
    const announce = (text) => { status.textContent = text; };
    let completed = 0;
    async function loadMesh(file) {
      const assetURL = new URL(file, sceneURL);
      const asset = data.assets.find(asset => asset.file === file);
      if (asset) assetURL.searchParams.set('v', asset.sha256.slice(0, 12));
      const response = await fetch(assetURL);
      if (!response.ok) throw new Error(`Gaussian asset: HTTP ${response.status}`);
      const mesh = new SplatMesh({ extSplats: true, covSplats: true, fileBytes: await response.arrayBuffer(), fileName: file, lod: false, enableLod: false, raycastable: false });
      loadedMeshes.push(mesh);
      await mesh.initialized;
      announce(`Loading scene… ${Math.round(++completed / (data.particles.length + 1) * 100)}%`);
      return mesh;
    }
    const fixed = await loadMesh(data.fixed);
    world.add(fixed);
    for (const [index, spec] of data.groups.entries()) {
      const group = new THREE.Group();
      const center = new THREE.Vector3();
      for (const id of spec.particles) center.add(vec(data.particles.find(p => p.id === id).center));
      center.multiplyScalar(1 / spec.particles.length);
      group.position.copy(center);
      group.userData = { spec, initial: center.clone() };
      groupById.set(spec.id, group);
      world.add(group);
      // A single camera-facing UI handle at the region's 3D center, not a
      // learned-particle marker. Native buttons support keyboard and touch.
      const button = document.createElement('button');
      button.type = 'button';
      button.className = 'demo-handle';
      button.dataset.region = spec.id;
      button.setAttribute('aria-label', `Edit region ${index + 1}`);
      button.setAttribute('aria-controls', 'demo-editor');
      button.setAttribute('aria-expanded', 'false');
      button.hidden = true;
      handleLayer.append(button);
      handles.set(spec.id, button);
      on(button, 'pointerdown', event => startDrag(event, spec.id));
      on(button, 'pointermove', moveDrag);
      for (const event of ['pointerup', 'pointercancel', 'lostpointercapture']) on(button, event, endDrag);
      on(button, 'click', event => {
        if (event.detail === 0) { select(spec.id); $('#demo-scale').focus(); }
      });
    }
    // Preserve particle ownership internally, exposing only aggregate edits.
    for (const particle of data.particles) {
      const spec = data.groups.find(g => g.particles.includes(particle.id));
      const group = groupById.get(spec.id);
      const node = new THREE.Group();
      node.position.copy(vec(particle.center).sub(group.position));
      node.userData = { id: particle.id, group: spec.id, initial: node.position.clone() };
      node.add(await loadMesh(particle.file));
      group.add(node);
      nodeById.set(particle.id, node);
    }
    viewport.prepend(canvas);
    orbit = new OrbitControls(camera, canvas);
    orbit.enableDamping = false;
    orbit.enablePan = true;
    orbit.minDistance = 0.02;
    orbit.maxDistance = 100;
    orbit.zoomSpeed = 0.7;
    // Keep the useful input-camera arc, with a small amount of extrapolation.
    // Match OrbitControls' Z-up -> Y-up spherical frame rather than assuming
    // its azimuth is the world's atan2(y, x). Limits stay fixed when focusing.
    const orbitFrame = new THREE.Quaternion().setFromUnitVectors(camera.up, new THREE.Vector3(0, 1, 0));
    const rigTarget = vec(data.camera.orbit_target);
    const poseAngles = (pose) => new THREE.Spherical().setFromVector3(
      new THREE.Vector3(pose[0][3], pose[1][3], pose[2][3]).sub(rigTarget).applyQuaternion(orbitFrame));
    const initialAngles = poseAngles(data.camera.c2w);
    const arcAngles = data.camera.arc_c2w.map(poseAngles);
    const azimuths = arcAngles.map(({ theta }) => initialAngles.theta + Math.atan2(
      Math.sin(theta - initialAngles.theta), Math.cos(theta - initialAngles.theta)));
    const margin = THREE.MathUtils.degToRad(5);
    orbit.minAzimuthAngle = Math.min(...azimuths) - margin;
    orbit.maxAzimuthAngle = Math.max(...azimuths) + margin;
    orbit.minPolarAngle = Math.max(0.01, Math.min(...arcAngles.map(({ phi }) => phi)) - margin);
    orbit.maxPolarAngle = Math.min(Math.PI - 0.01, Math.max(...arcAngles.map(({ phi }) => phi)) + margin);
    gizmo = new TransformControls(camera, canvas);
    gizmo.setMode('translate');
    gizmo.setSpace('world');
    gizmo.setSize(0.6);
    const helper = gizmo.getHelper();
    helper.visible = false;
    gizmo.enabled = false;
    world.add(helper);

    function frameCamera() {
      const aspect = viewport.clientWidth / viewport.clientHeight;
      const height = Math.min(framing.height, 0.96 / aspect);
      const width = height * aspect;
      const cx = clamp(framing.center[0], width / 2, 1 - width / 2);
      const cy = clamp(framing.center[1], height / 2, 1 - height / 2);
      camera.setViewOffset(128, 128, (cx - width / 2) * 128, (cy - height / 2) * 128, width * 128, height * 128);
      // Filter the learned 128px sampling footprint into the displayed crop.
      spark.preBlurAmount = 0.3 * (viewport.clientHeight * renderer.getPixelRatio() / (128 * height)) ** 2;
      camera.updateProjectionMatrix();
    }
    function initialCamera() {
      framing = { center: [...data.framing.center], height: data.framing.height };
      const matrix = new THREE.Matrix4().set(...data.camera.c2w.flat());
      // OpenCV +Z forward/+Y down -> Three -Z forward/+Y up, exactly once.
      matrix.multiply(new THREE.Matrix4().makeScale(1, -1, -1));
      camera.position.setFromMatrixPosition(matrix);
      camera.quaternion.setFromRotationMatrix(matrix);
      camera.fov = THREE.MathUtils.radToDeg(2 * Math.atan(0.5 / data.camera.intrinsics[1][1]));
      orbit.target.copy(vec(data.camera.orbit_target));
      orbit.update();
      frameCamera();
      invalidate();
    }
    function resize() {
      renderer.setSize(viewport.clientWidth, viewport.clientHeight, false);
      frameCamera();
      updateHandles();
      invalidate();
    }
    function projected(group) { return group.getWorldPosition(new THREE.Vector3()).project(camera); }
    function axisTips() {
      if (!selected || !helper.visible || !gizmo.enabled) return [];
      // Read the visible tips of the pinned Three.js gizmo, including its
      // camera-dependent size. Labels and hit-test diagnostics stay aligned.
      return ['X', 'Y', 'Z'].flatMap(axis => {
        const tip = helper.children[0].gizmo[gizmo.mode].children.find(part => {
          if (part.name !== axis || !part.isMesh || !part.visible) return false;
          part.geometry.computeBoundingBox();
          return part.geometry.boundingBox.getCenter(new THREE.Vector3())[axis.toLowerCase()] > 0.45;
        });
        if (!tip) return [];
        const p = tip.localToWorld(tip.geometry.boundingBox.getCenter(new THREE.Vector3())).project(camera);
        return [{ axis, x: (p.x + 1) * viewport.clientWidth / 2, y: (1 - p.y) * viewport.clientHeight / 2 }];
      });
    }
    function updateHandles() {
      world.updateMatrixWorld(true);
      camera.updateMatrixWorld(true);
      for (const [id, group] of groupById) {
        const button = handles.get(id);
        const p = projected(group);
        const visible = showHandles && p.z >= -1 && p.z <= 1 && Math.abs(p.x) < 1 && Math.abs(p.y) < 1;
        // Keep pointer capture alive if a handle is dragged to a viewport edge.
        button.hidden = !visible && drag?.button !== button;
        button.style.left = `${(p.x + 1) * viewport.clientWidth / 2}px`;
        button.style.top = `${(1 - p.y) * viewport.clientHeight / 2}px`;
        button.style.zIndex = group === selected ? 20 : Math.round((1 - p.z) * 5);
      }
      const tips = axisTips();
      $('#demo-axis-labels').hidden = tips.length === 0;
      for (const label of $('#demo-axis-labels').children) {
        const tip = tips.find(tip => tip.axis.toLowerCase() === label.dataset.axis);
        label.hidden = !tip;
        if (!tip) continue;
        const center = projected(selected);
        const dx = tip.x - (center.x + 1) * viewport.clientWidth / 2;
        const dy = tip.y - (1 - center.y) * viewport.clientHeight / 2;
        const length = Math.max(1, Math.hypot(dx, dy));
        label.style.left = `${clamp(tip.x + dx / length * 18, 11, viewport.clientWidth - 11)}px`;
        label.style.top = `${clamp(tip.y + dy / length * 18, 11, viewport.clientHeight - 11)}px`;
      }
    }
    function syncFields() {
      if (!selected) return;
      $('#demo-scale').value = selected.scale.x.toFixed(2);
      const uniform = Math.max(...selected.scale.toArray()) - Math.min(...selected.scale.toArray()) < 0.001;
      $('#demo-scale-value').value = uniform ? `${selected.scale.x.toFixed(2)}×` : 'Per-axis';
      $('#demo-scale').setAttribute('aria-valuetext', uniform ? `${selected.scale.x.toFixed(2)} times` : 'Per-axis scaling; adjusting this slider sets all axes equally');
      const delta = selected.position.clone().sub(selected.userData.initial);
      for (const axis of ['x', 'y', 'z']) $(`#demo-${axis}`).value = gizmo.mode === 'scale' ? selected.scale[axis].toFixed(2) : (delta[axis] * 100).toFixed(1);
      invalidate();
    }
    function syncArrows() {
      helper.visible = gizmo.enabled = !!selected && !drag;
      updateHandles();
      invalidate();
    }
    function setMode(mode) {
      gizmo.setMode(mode);
      gizmo.axis = null;
      const scaling = mode === 'scale';
      $('#demo-move-mode').setAttribute('aria-pressed', String(!scaling));
      $('#demo-scale-mode').setAttribute('aria-pressed', String(scaling));
      $('#demo-axis-heading').textContent = scaling ? 'Scale per axis' : 'Position offset';
      $('#demo-axis-unit').textContent = scaling ? '×' : 'cm';
      $('#demo-axis-hint').textContent = scaling ? 'Drag an axis tip to stretch X, Y or Z.' : 'Drag an arrow to move along X, Y or Z.';
      for (const axis of ['x', 'y', 'z']) {
        const input = $(`#demo-${axis}`);
        input.min = scaling ? '0.5' : '-25';
        input.max = scaling ? '2' : '25';
        input.step = scaling ? '0.01' : '0.1';
        input.setAttribute('aria-label', `${axis.toUpperCase()} ${scaling ? 'scale factor' : 'offset in centimeters'}`);
      }
      syncFields();
      syncArrows();
    }
    function select(id) {
      selected = groupById.get(id);
      world.updateMatrixWorld(true);
      const p = projected(selected);
      panel.dataset.side = p.x > 0 ? 'left' : 'right';
      panel.hidden = false;
      $('#demo-control-fields').disabled = false;
      for (const [key, button] of handles) button.setAttribute('aria-expanded', String(key === id));
      gizmo.attach(selected);
      syncArrows();
      syncFields();
      announce('Selection ready. Drag its handle or adjust the controls.');
    }
    function closeEditor(restoreFocus = true) {
      stopDrag();
      gizmo.pointerUp(null);
      const button = selected && handles.get(selected.userData.spec.id);
      selected = null;
      panel.hidden = true;
      $('#demo-control-fields').disabled = true;
      for (const button of handles.values()) button.setAttribute('aria-expanded', 'false');
      gizmo.detach();
      syncArrows();
      if (restoreFocus) (button && !button.hidden ? button : canvas).focus({ preventScroll: true });
      announce('Click a handle to edit.');
    }
    function enforceBounds() {
      if (!selected) return;
      for (const axis of ['x', 'y', 'z']) {
        const delta = selected.position[axis] - selected.userData.initial[axis];
        selected.position[axis] = selected.userData.initial[axis] + (Number.isFinite(delta) ? clamp(delta, -0.25, 0.25) : 0);
        selected.scale[axis] = Number.isFinite(selected.scale[axis]) ? clamp(selected.scale[axis], 0.5, 2) : 1;
      }
      syncFields();
    }
    function resetNode(node) { node.position.copy(node.userData.initial); node.scale.setScalar(1); }
    function setRay(event) {
      const rect = canvas.getBoundingClientRect();
      pointer.set((event.clientX - rect.left) / rect.width * 2 - 1, -(event.clientY - rect.top) / rect.height * 2 + 1);
      raycaster.setFromCamera(pointer, camera);
    }
    function startDrag(event, id) {
      if (event.button !== 0 || !event.isPrimary || drag) return;
      event.preventDefault();
      event.stopPropagation();
      select(id);
      setRay(event);
      const initial = selected.getWorldPosition(new THREE.Vector3());
      const plane = new THREE.Plane().setFromNormalAndCoplanarPoint(camera.getWorldDirection(new THREE.Vector3()), initial);
      const point = raycaster.ray.intersectPlane(plane, new THREE.Vector3());
      if (!point) return;
      const button = event.currentTarget;
      drag = { button, pointerId: event.pointerId, initial, plane, point, x: event.clientX, y: event.clientY, moved: false };
      orbit.enabled = false;
      syncArrows();
      button.setPointerCapture(event.pointerId);
      button.focus({ preventScroll: true });
      button.classList.add('is-dragging');
    }
    function moveDrag(event) {
      if (!drag || event.pointerId !== drag.pointerId) return;
      event.preventDefault();
      event.stopPropagation();
      if (!drag.moved && Math.hypot(event.clientX - drag.x, event.clientY - drag.y) < 4) return;
      drag.moved = true;
      setRay(event);
      const point = raycaster.ray.intersectPlane(drag.plane, new THREE.Vector3());
      if (!point) return;
      point.sub(drag.point).add(drag.initial);
      selected.position.copy(selected.parent.worldToLocal(point));
      enforceBounds();
      updateHandles();
    }
    function stopDrag() {
      if (!drag) return;
      const { button, pointerId, moved } = drag;
      drag = null;
      if (button.hasPointerCapture(pointerId)) button.releasePointerCapture(pointerId);
      button.classList.remove('is-dragging');
      orbit.enabled = true;
      syncArrows();
      if (moved) announce('Position updated.');
    }
    function endDrag(event) { if (drag?.pointerId === event.pointerId) stopDrag(); }

    gizmo.addEventListener('dragging-changed', event => { orbit.enabled = !event.value; invalidate(); });
    gizmo.addEventListener('change', invalidate);
    gizmo.addEventListener('objectChange', enforceBounds);
    on(canvas, 'pointercancel', () => gizmo.pointerUp(null));
    orbit.addEventListener('change', () => { updateHandles(); invalidate(); });
    on($('#demo-scale'), 'input', () => {
      if (!selected) return;
      selected.scale.setScalar(clamp(Number($('#demo-scale').value), 0.5, 2));
      syncFields();
    });
    for (const axis of ['x', 'y', 'z']) on($(`#demo-${axis}`), 'change', event => {
      if (!selected) return;
      const value = Number(event.target.value);
      if (Number.isFinite(value)) {
        if (gizmo.mode === 'scale') selected.scale[axis] = clamp(value, 0.5, 2);
        else selected.position[axis] = selected.userData.initial[axis] + clamp(value, -25, 25) / 100;
      }
      syncFields();
    });
    on($('#demo-move-mode'), 'click', () => setMode('translate'));
    on($('#demo-scale-mode'), 'click', () => setMode('scale'));
    on($('#demo-close'), 'click', () => closeEditor());
    on(root, 'keydown', event => { if (event.key === 'Escape' && selected) { event.preventDefault(); closeEditor(); } });
    on($('#demo-reset-selected'), 'click', () => {
      if (!selected) return;
      resetNode(selected);
      syncFields();
      announce('Selection restored.');
    });
    on($('#demo-reset-all'), 'click', () => {
      for (const group of groupById.values()) resetNode(group);
      syncFields();
      invalidate();
      announce('Scene edits reset.');
    });
    on($('#demo-camera'), 'click', initialCamera);
    on($('#demo-focus'), 'click', () => {
      if (!selected) return;
      const direction = camera.position.clone().sub(orbit.target).normalize();
      selected.getWorldPosition(orbit.target);
      camera.position.copy(orbit.target).addScaledVector(direction, 0.55);
      framing = { center: [0.5, 0.5], height: 0.9 };
      orbit.update();
      frameCamera();
      invalidate();
    });
    on($('#demo-show-handles'), 'click', () => {
      showHandles = !showHandles;
      $('#demo-show-handles').setAttribute('aria-pressed', String(showHandles));
      updateHandles();
      invalidate();
    });
    resizeObserver = new ResizeObserver(resize);
    resizeObserver.observe(viewport);
    observer = new IntersectionObserver(([entry]) => { inView = entry.isIntersecting; invalidate(); });
    observer.observe(viewport);
    resize();
    initialCamera();
    updateHandles();
    await spark.update({ scene: world, camera });
    renderer.render(world, camera);
    $('.demo-start').hidden = true;
    $('.demo-poster').hidden = true;
    $('.demo-canvas-caption').hidden = false;
    $('.demo-view-tools').hidden = false;
    root.dataset.state = 'ready';
    announce('Click a handle to edit.');
    canvas.focus({ preventScroll: true });
    renderer.setAnimationLoop(() => {
      if (disposed || document.hidden || !inView || performance.now() - lastInteraction > 1600) return;
      updateHandles();
      renderer.render(world, camera);
    });
    on(document, 'visibilitychange', invalidate);
    on(canvas, 'webglcontextlost', event => {
      event.preventDefault();
      closeEditor(false);
      renderer.setAnimationLoop(null);
      spark.autoUpdate = false;
      showHandles = false;
      updateHandles();
      canvas.hidden = true;
      $('.demo-poster').hidden = false;
      $('.demo-canvas-caption').hidden = true;
      $('.demo-view-tools').hidden = true;
      root.dataset.state = 'lost';
      announce('The graphics context was lost. Reload the page to restart the 3D demo.');
    });
    // Read-only diagnostics are opt-in and never offer user-facing particle edits.
    if (new URLSearchParams(location.search).has('demo-test')) {
      window.particleDemoTest = { snapshot: () => {
        updateHandles();
        return { selected: selected?.userData.spec.id ?? null,
          nodes: [...nodeById].map(([id, node]) => ({ id, position: node.position.toArray(), world: node.getWorldPosition(new THREE.Vector3()).toArray(), scale: node.scale.x })),
          groups: [...groupById].map(([id, node]) => ({ id, position: node.position.toArray(), scale: node.scale.x, scaleXYZ: node.scale.toArray() })),
          fixed: fixed.matrixWorld.toArray(), count: loadedMeshes.reduce((n, mesh) => n + mesh.numSplats, 0),
          camera: camera.position.toArray(), target: orbit.target.toArray(),
          angles: [orbit.getAzimuthalAngle(), orbit.getPolarAngle()],
          angleLimits: [[orbit.minAzimuthAngle, orbit.maxAzimuthAngle], [orbit.minPolarAngle, orbit.maxPolarAngle]],
          arrows: helper.visible && gizmo.enabled,
          mode: gizmo.mode, activeAxis: gizmo.axis, axisTips: axisTips(),
          covariance: loadedMeshes.every(mesh => mesh.covSplats) && spark.covSplats,
          renderedTransforms: [...nodeById].map(([id, node]) => ({ id, basis: node.children[0].context.covTransform.basis.value.toArray() })),
          extended: loadedMeshes.every(mesh => !!mesh.extSplats) && spark.accumExtSplats,
          handles: [...handles].map(([id, button]) => ({ id, visible: !button.hidden, x: parseFloat(button.style.left), y: parseFloat(button.style.top) })) };
      } };
    }
    on(window, 'pagehide', event => {
      if (event.persisted) return;
      disposed = true;
      renderer.setAnimationLoop(null);
      observer.disconnect();
      resizeObserver.disconnect();
      listeners.abort();
      orbit.dispose();
      gizmo.dispose();
      for (const mesh of loadedMeshes) mesh.dispose();
      spark.dispose();
      renderer.dispose();
    });
  } catch (error) {
    listeners.abort();
    observer?.disconnect();
    resizeObserver?.disconnect();
    orbit?.dispose();
    gizmo?.dispose();
    for (const mesh of loadedMeshes) mesh.dispose();
    spark.dispose();
    renderer.dispose();
    canvas.remove();
    handleLayer.replaceChildren();
    panel.hidden = true;
    throw error;
  }
}
