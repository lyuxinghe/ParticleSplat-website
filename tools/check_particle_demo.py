#!/usr/bin/env python3
"""Exercise actual WebGL, region handles and GitHub Pages subpath hosting."""
import argparse
import functools
import hashlib
import http.server
import json
from pathlib import Path
import threading

import numpy as np
from playwright.sync_api import sync_playwright

SOURCE_SITE = Path(__file__).resolve().parents[1]
SITE = SOURCE_SITE
PREFIX = "/ParticleSplat/"


class Handler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if not self.path.startswith(PREFIX):
            self.send_error(404)
            return
        self.path = "/" + self.path[len(PREFIX):]
        super().do_GET()

    def log_message(self, *args):
        pass


def main():
    global SITE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screenshots", type=Path, default=Path("/tmp/particlesplat-demo-check"))
    parser.add_argument("--site", type=Path, default=SOURCE_SITE, help="Source site or built static artifact")
    args = parser.parse_args()
    SITE = args.site.resolve()
    args.screenshots.mkdir(parents=True, exist_ok=True)
    scene_dir = SITE / "static/scenes/close-jar"
    manifest = json.loads((scene_dir / "scene.json").read_text())
    assert manifest["retained_gaussians"] == 42336
    cleanup = manifest['postprocessing']
    assert cleanup['source_gaussians'] == 43008 and cleanup['removed_gaussians'] == 672
    assert set(map(int, cleanup['removed_indices'])) == {8, 26, 10, 25, 13, 28}
    assert cleanup['rule']['scene'] == manifest['provenance']['scene']
    assert cleanup['rule']['method'] == 'yellow-tabletop-spill-v1'
    assert set(cleanup['rule']['particles']) == set(map(int, cleanup['removed_indices']))
    assert sum(len(indices) for indices in cleanup['removed_indices'].values()) == 672
    per_particle = manifest['provenance']['gaussians_per_particle']
    for particle in manifest['particles']:
        asset = next(a for a in manifest['assets'] if a['file'] == particle['file'])
        removed = cleanup['removed_indices'].get(str(particle['id']), [])
        assert asset['gaussians'] == per_particle - len(removed)
    assert manifest['provenance']['scene'] == 'close_jar_episode1_40'
    assert manifest['provenance']['context_views'] == [6, 14]
    assert len(manifest['groups']) == 8
    assert sum(len(g['particles']) for g in manifest['groups']) == 25
    expected_groups = json.loads((SOURCE_SITE / 'tools/close-jar-scene.json').read_text())['groups']
    assert manifest['groups'] == expected_groups
    assert next(g for g in manifest['groups'] if g['id'] == 'region-3')['particles'] == [13, 28]
    assert next(g for g in manifest['groups'] if g['id'] == 'region-4')['particles'] == [1, 6, 20, 24]
    pids = [pid for group in manifest['groups'] for pid in group['particles']]
    assert len(set(pids)) == len(pids)
    assert {p['id'] for p in manifest['particles']} == set(pids)
    assert {p.name for p in scene_dir.glob('*.ply')} == {asset['file'] for asset in manifest['assets']}
    assert manifest["camera"]["view"] not in manifest["provenance"]["context_views"]
    for asset in manifest["assets"]:
        assert hashlib.sha256((scene_dir / asset["file"]).read_bytes()).hexdigest() == asset["sha256"]
    for filename, asset in json.loads((SITE / "static/vendor/manifest.json").read_text()).items():
        assert hashlib.sha256((SITE / "static/vendor" / filename).read_bytes()).hexdigest() == asset["sha256"]
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(Handler, directory=str(SITE)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    origin = f"http://127.0.0.1:{server.server_port}"
    url = origin + PREFIX + "?demo-test=1#manipulation"
    errors, failed, requests = [], [], []

    def ready(page):
        page.locator('#demo-load').click()
        page.wait_for_function("document.querySelector('.particle-demo').dataset.state !== 'loading'", timeout=120000)
        assert page.locator('.particle-demo').get_attribute('data-state') == 'ready', page.locator('#demo-status').inner_text()
        page.wait_for_timeout(500)

    def state(page):
        return page.evaluate('window.particleDemoTest.snapshot()')

    def node(snapshot, pid):
        return next(n for n in snapshot['nodes'] if n['id'] == pid)

    def bounded(snapshot):
        for angle, (low, high) in zip(snapshot['angles'], snapshot['angleLimits']):
            assert low - 1e-6 <= angle <= high + 1e-6, (angle, low, high)

    def handle(page, group):
        return page.locator(f'.demo-handle[data-region="{group}"]')

    def close(page):
        if page.locator('#demo-editor').is_visible():
            page.locator('#demo-close').click()

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(args=['--enable-unsafe-swiftshader'])
            page = browser.new_page(viewport={'width':1440, 'height':1150})
            page.on('pageerror', lambda e: errors.append(str(e)))
            page.on('response', lambda r: failed.append(r.url) if r.status >= 400 else None)
            page.on('request', lambda r: requests.append(r.url))
            page.goto(url, wait_until='networkidle')
            assert not any('/vendor/' in u or '.ply' in u for u in requests), 'Heavy assets loaded before opt-in'
            assert not page.locator('#demo-editor').is_visible()
            page.locator('#manipulation').screenshot(path=args.screenshots / 'poster-desktop.png')
            ready(page)
            original = state(page)
            assert original['count'] == manifest['retained_gaussians'] and original['extended']
            assert original['covariance'], 'Per-axis scaling requires full covariance transforms'
            assert original['selected'] is None
            bounded(original)
            rig_target = np.array(manifest['camera']['orbit_target'])
            offsets = np.array(manifest['camera']['arc_c2w'])[:, :3, 3] - rig_target
            # Independent calculation of OrbitControls angles with camera.up=Z.
            azimuths = np.arctan2(offsets[:, 0], -offsets[:, 1])
            polars = np.arccos(offsets[:, 2] / np.linalg.norm(offsets, axis=1))
            margin = np.deg2rad(5)
            expected_limits = [[azimuths.min() - margin, azimuths.max() + margin],
                               [polars.min() - margin, polars.max() + margin]]
            assert np.allclose(original['angleLimits'], expected_limits)
            assert len(original['handles']) == 8 and all(h['visible'] for h in original['handles'])
            assert page.locator('.demo-handle').all_text_contents() == [''] * 8
            assert page.locator('#demo-particle, [data-mode], .demo-objects').count() == 0
            public_text = page.locator('.particle-demo').inner_text().lower()
            assert not any(text in public_text for text in ['violet', 'magenta', 'one particle', '2 particles', 'four learned'])
            page.locator('#manipulation').screenshot(path=args.screenshots / 'demo-desktop.png')
            assert all(u.startswith(origin + PREFIX) or u.startswith('blob:') for u in requests)
            splat_requests = [u for u in requests if '.ply' in u]
            assert len(splat_requests) == len(manifest['assets'])
            assert all('?v=' in u for u in splat_requests), 'Geometry URLs must invalidate stale cached fields'
            print('PASS: frame 40, eight unnamed handles, independent lid, hidden editor and local lazy loading', flush=True)
            for i, group in enumerate(manifest['groups']):
                close(page)
                handle(page, group['id']).click()
                assert state(page)['selected'] == group['id']
                assert state(page)['arrows'], 'Selecting a region should immediately show its axes'
                assert state(page)['nodes'] == original['nodes'], 'Click moved geometry'
                assert page.locator('#demo-editor').is_visible()
                page.locator('#demo-x').fill('5')
                page.locator('#demo-x').press('Tab')
                moved = state(page)
                for particle in manifest['particles']:
                    pid = particle['id']
                    delta = [.05, 0, 0] if pid in group['particles'] else [0, 0, 0]
                    assert np.allclose(np.array(node(moved, pid)['world']) - node(original, pid)['world'], delta)
                assert moved['fixed'] == original['fixed']
                page.locator('#demo-scale').fill('1.5')
                scaled = state(page)
                center = np.array(scaled['groups'][i]['position'])
                for pid in group['particles']:
                    assert np.allclose(node(scaled, pid)['world'], center + np.array(node(original, pid)['position']) * 1.5)
                for pid in set(pids) - set(group['particles']):
                    assert node(scaled, pid) == node(original, pid), 'Scaling changed an unselected field'
                assert scaled['fixed'] == original['fixed']
                if i in [0, 2, 3, 7]:
                    page.locator('#manipulation').screenshot(path=args.screenshots / f'edited-{group["id"]}.png')
                page.locator('#demo-reset-selected').click()
                assert state(page)['nodes'] == original['nodes']
                assert state(page)['groups'] == original['groups']
            print('PASS: every region independently translates, scales and resets', flush=True)
            close(page)
            handle(page, 'region-1').click()
            # Actual pointer drags on the rendered gizmo, not synthetic property edits.
            for mode in ['translate', 'scale']:
                page.locator('#demo-move-mode' if mode == 'translate' else '#demo-scale-mode').click()
                assert state(page)['mode'] == mode and state(page)['arrows']
                for index, axis in enumerate('XYZ'):
                    page.locator('#demo-reset-selected').click()
                    page.wait_for_timeout(150)
                    before = state(page)
                    tip = next(tip for tip in before['axisTips'] if tip['axis'] == axis)
                    center = next(h for h in before['handles'] if h['id'] == 'region-1')
                    box = page.locator('#particle-viewport canvas').bounding_box()
                    dx, dy = tip['x'] - center['x'], tip['y'] - center['y']
                    length = np.hypot(dx, dy)
                    x, y = box['x'] + tip['x'], box['y'] + tip['y']
                    page.mouse.move(x, y)
                    assert state(page)['activeAxis'] == axis, (mode, axis, state(page)['activeAxis'])
                    page.mouse.down()
                    page.mouse.move(x + dx / length * 20, y + dy / length * 20, steps=5)
                    page.mouse.up()
                    after = state(page)
                    a, b = before['groups'][0], after['groups'][0]
                    key = 'position' if mode == 'translate' else 'scaleXYZ'
                    delta = np.array(b[key]) - a[key]
                    assert abs(delta[index]) > .001, (mode, axis, delta)
                    assert np.allclose(np.delete(delta, index), 0), (mode, axis, delta)
                    assert after['groups'][1:] == original['groups'][1:]
                    assert after['fixed'] == original['fixed']
                    assert np.allclose(after['camera'], before['camera']), 'Axis drag also orbited camera'
                    print(f'PASS: {mode} {axis} gizmo drag', flush=True)
                page.locator('#manipulation').screenshot(path=args.screenshots / f'axes-{mode}.png')
            for axis, value in zip('xyz', ['1.2', '0.8', '1.6']):
                page.locator(f'#demo-{axis}').fill(value)
                page.locator(f'#demo-{axis}').press('Tab')
            scaled = state(page)
            assert np.allclose(scaled['groups'][0]['scaleXYZ'], [1.2, .8, 1.6])
            page.wait_for_function("""() => window.particleDemoTest.snapshot().renderedTransforms
                .filter(item => [8, 26].includes(item.id))
                .every(item => [1.2, 0, 0, 0, .8, 0, 0, 0, 1.6]
                  .every((value, index) => Math.abs(item.basis[index] - value) < 1e-6))""")
            assert page.locator('#demo-scale-value').inner_text() == 'Per-axis'
            center = np.array(scaled['groups'][0]['position'])
            for pid in manifest['groups'][0]['particles']:
                assert np.allclose(node(scaled, pid)['world'], center + np.array(node(original, pid)['position']) * [1.2, .8, 1.6])
            page.locator('#demo-x').fill('4')
            page.locator('#demo-x').press('Tab')
            assert state(page)['groups'][0]['scaleXYZ'][0] == 2
            page.locator('#demo-y').fill('-1')
            page.locator('#demo-y').press('Tab')
            assert state(page)['groups'][0]['scaleXYZ'][1] == .5
            page.locator('#demo-scale').fill('1.3')
            assert state(page)['groups'][0]['scaleXYZ'] == [1.3, 1.3, 1.3]
            page.locator('#demo-reset-selected').click()
            page.locator('#demo-move-mode').click()
            assert state(page)['groups'] == original['groups']
            print('PASS: automatic XYZ gizmos, real pointer drags on all six axes, independent scale fields and safe bounds', flush=True)
            close(page)
            first = manifest['groups'][0]
            control = handle(page, first['id'])
            control.focus()
            page.keyboard.press('Enter')
            assert page.locator('#demo-scale').evaluate('(el) => el === document.activeElement')
            page.keyboard.press('ArrowRight')
            assert page.locator('#demo-scale').input_value() == '1.01'
            assert state(page)['arrows']
            page.keyboard.press('Escape')
            assert not page.locator('#demo-editor').is_visible()
            assert control.evaluate('(el) => el === document.activeElement')
            assert not state(page)['arrows']
            control.click()
            assert page.locator('#demo-scale').input_value() == '1.01', 'Closing lost the edit'
            assert state(page)['arrows'], 'Axes did not return when reselecting'
            page.locator('#demo-reset-all').click()
            page.locator('#demo-focus').click()
            page.wait_for_timeout(250)
            bounded(state(page))
            assert state(page)['angleLimits'] == original['angleLimits'], 'Focus expanded camera limits'
            control.scroll_into_view_if_needed()
            box = control.bounding_box()
            x, y = box['x'] + box['width']/2, box['y'] + box['height']/2
            page.mouse.move(x, y)
            page.mouse.down()
            page.mouse.move(x + 55, y - 20, steps=8)
            page.mouse.up()
            dragged = state(page)
            delta = np.array(node(dragged, first['particles'][0])['world']) - node(original, first['particles'][0])['world']
            assert np.linalg.norm(delta) > .005
            for pid in first['particles']:
                assert np.allclose(np.array(node(dragged, pid)['world']) - node(original, pid)['world'], delta)
            page.locator('#demo-reset-all').click()
            close(page)
            page.locator('#demo-camera').click()
            assert np.allclose(state(page)['camera'], original['camera'])
            page.locator('#demo-show-handles').click()
            assert not any(h['visible'] for h in state(page)['handles'])
            canvas = page.locator('#particle-viewport canvas')
            box = canvas.bounding_box()
            # Large/repeated drags clamp at both ends, but do not lock rotation.
            def orbit_drag(dx, dy):
                x, y = box['x'] + box['width']/2, box['y'] + box['height']/2
                page.mouse.move(x, y)
                page.mouse.down()
                page.mouse.move(x + dx, y + dy, steps=5)
                page.mouse.up()
                snapshot = state(page)
                bounded(snapshot)
                assert snapshot['angleLimits'] == original['angleLimits']
                return snapshot

            for sign, end, label in [(1, 0, 'left'), (-1, 1, 'right')]:
                orbited = orbit_drag(sign * 300, sign * 150)
                assert np.allclose(orbited['angles'], np.array(expected_limits)[:, end])
                repeated = orbit_drag(sign * 300, sign * 150)
                assert np.allclose(repeated['camera'], orbited['camera']), 'Dragged past camera boundary'
                page.locator('#manipulation').screenshot(path=args.screenshots / f'bounded-orbit-{label}.png')
            inside = orbit_drag(10, 3)
            assert all(low < angle < high for angle, (low, high) in zip(inside['angles'], expected_limits))
            before_pan = state(page)
            page.mouse.down(button='right')
            page.mouse.move(box['x'] + 380, box['y'] + 245, steps=5)
            page.mouse.up(button='right')
            assert not np.allclose(state(page)['target'], before_pan['target'])
            bounded(state(page))
            before_zoom = state(page)
            page.mouse.wheel(0, -400)
            page.wait_for_timeout(200)
            assert not np.allclose(state(page)['camera'], before_zoom['camera'])
            bounded(state(page))
            page.locator('#demo-camera').click()
            page.locator('#demo-show-handles').click()
            assert np.allclose(state(page)['camera'], original['camera'])
            print('PASS: click, drag, keyboard panel, focus, arrows, bounded orbit/tilt with 5-degree margins, pan and zoom', flush=True)
            for width in [320, 390, 768, 1024]:
                page.set_viewport_size({'width':width, 'height':1100})
                page.wait_for_timeout(150)
                control.click()
                assert page.evaluate('document.documentElement.scrollWidth <= innerWidth'), f'Overflow at {width}'
                close(page)
            page.set_viewport_size({'width':390, 'height':1100})
            control.click()
            page.locator('#manipulation').screenshot(path=args.screenshots / 'demo-mobile.png')
            assert not errors and not failed, (errors, failed)
            page.close()

            mobile = browser.new_page(viewport={'width':390, 'height':844}, is_mobile=True, has_touch=True)
            mobile.on('pageerror', lambda e: errors.append(str(e)))
            mobile.goto(url)
            ready(mobile)
            control = handle(mobile, first['id'])
            control.tap()
            assert mobile.locator('#demo-editor').is_visible()
            assert state(mobile)['arrows']
            mobile.locator('#demo-scale-mode').tap()
            mobile.locator('#demo-z').fill('1.4')
            mobile.locator('#demo-z').press('Tab')
            assert state(mobile)['groups'][0]['scaleXYZ'] == [1, 1, 1.4]
            mobile.locator('#demo-reset-selected').tap()
            mobile.locator('#demo-move-mode').tap()
            before = state(mobile)
            control.scroll_into_view_if_needed()
            box = control.bounding_box()
            x, y = box['x'] + box['width']/2, box['y'] + box['height']/2
            cdp = mobile.context.new_cdp_session(mobile)
            cdp.send('Input.dispatchTouchEvent', {'type':'touchStart', 'touchPoints':[{'x':x, 'y':y}]})
            for step in range(1, 7):
                cdp.send('Input.dispatchTouchEvent', {'type':'touchMove', 'touchPoints':[{'x':x + step*5, 'y':y - step*2}]})
            cdp.send('Input.dispatchTouchEvent', {'type':'touchEnd', 'touchPoints':[]})
            assert not np.allclose(node(state(mobile), first['particles'][0])['world'], node(before, first['particles'][0])['world'])
            close(mobile)
            mobile.locator('#demo-show-handles').click()
            canvas = mobile.locator('#particle-viewport canvas')
            canvas.scroll_into_view_if_needed()
            box = canvas.bounding_box()
            for direction in [1, -1]:
                x, y = box['x'] + box['width']/2, box['y'] + box['height']/2
                cdp.send('Input.dispatchTouchEvent', {'type':'touchStart', 'touchPoints':[{'x':x, 'y':y}]})
                for step in range(1, 7):
                    cdp.send('Input.dispatchTouchEvent', {'type':'touchMove', 'touchPoints':[{'x':x + direction*step*20, 'y':y + direction*step*10}]})
                cdp.send('Input.dispatchTouchEvent', {'type':'touchEnd', 'touchPoints':[]})
                bounded(state(mobile))
                end = 0 if direction == 1 else 1
                assert np.allclose(state(mobile)['angles'], np.array(expected_limits)[:, end])
            mobile.locator('#demo-camera').click()
            assert np.allclose(state(mobile)['camera'], original['camera'])
            mobile.locator('#demo-show-handles').click()
            control.tap()
            mobile.evaluate("document.querySelector('#particle-viewport canvas').getContext('webgl2').getExtension('WEBGL_lose_context').loseContext()")
            mobile.wait_for_function("document.querySelector('.particle-demo').dataset.state === 'lost'")
            assert mobile.locator('.demo-poster').is_visible()
            assert not mobile.locator('#demo-editor').is_visible()
            assert mobile.locator('.demo-handle:visible').count() == 0
            # A context can be lost while Spark's async GPU sort is in flight.
            mobile.wait_for_timeout(1500)
            assert not errors, errors
            mobile.close()
            nojs = browser.new_page(java_script_enabled=False)
            nojs.goto(url)
            assert nojs.locator('.demo-poster').is_visible()
            assert not nojs.locator('#demo-load').is_visible()
            assert not nojs.locator('#demo-editor').is_visible()
            assert nojs.locator('#manipulation noscript').is_visible()
            nojs.close()
            fallback = browser.new_page()
            fallback.add_init_script("const original = HTMLCanvasElement.prototype.getContext; HTMLCanvasElement.prototype.getContext = function(type, ...args) { return type === 'webgl2' ? null : original.call(this, type, ...args); };")
            fallback.goto(url)
            fallback.locator('#demo-load').click()
            fallback.wait_for_function("document.querySelector('.particle-demo').dataset.state === 'error'")
            assert fallback.locator('.demo-poster').is_visible()
            assert fallback.locator('#demo-load').is_enabled()
            assert not fallback.locator('#demo-editor').is_visible()
            fallback.close()
            assert not errors, errors
            browser.close()
    finally:
        server.shutdown()
        server.server_close()
    print('PASS: touch editing, 320–1440px layouts, no-JS/WebGL fallbacks and GitHub Pages subpath hosting.')
    print(f'Screenshots: {args.screenshots}')


if __name__ == '__main__':
    main()
