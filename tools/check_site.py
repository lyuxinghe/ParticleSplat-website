"""Check the static project page and decode every video in local Chromium.

Requires Playwright and its Chromium browser, only for this development check.
Optionally pass --screenshots /tmp/some-directory to save layout screenshots.
"""

import argparse
import csv
import functools
import hashlib
import http.server
import json
from pathlib import Path
import threading

from playwright.sync_api import sync_playwright


SITE = Path(__file__).resolve().parents[1]


class QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


def main():
    global SITE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screenshots", type=Path)
    parser.add_argument("--site", type=Path, default=SITE, help="Source site or built static artifact")
    args = parser.parse_args()
    SITE = args.site.resolve()
    if args.screenshots:
        args.screenshots.mkdir(parents=True, exist_ok=True)
    entries = json.loads((SITE / "static/videos/manifest.json").read_text())["videos"]
    assert len(entries) == 36
    assert sum(item["group"] == "policy" for item in entries) == 20
    assert all("failure" not in item["path"] and "/f1.mp4" not in item["source_member"] for item in entries)
    assert len(list((SITE / "static/videos").rglob("*.mp4"))) == 36
    assert len(list((SITE / "static/videos").rglob("*.jpg"))) == 36
    assert not (SITE / "static/papers").exists(), "Private paper assets must not be included in the website"
    assert "supplementary_pdf" not in json.loads((SITE / "static/videos/manifest.json").read_text())
    # Keep the imported archive intact, but display only the first success per task.
    entries = [entry for entry in entries if entry["group"] != "policy" or entry["trial"] == "success-1"]
    assert len(entries) == 26
    pairs = json.loads((SITE / "static/images/reconstruction/manifest.json").read_text())["pairs"]
    metrics = json.loads((SITE / "static/reconstruction-results.json").read_text())["results"]
    assert len(pairs) == 8 and len(metrics) == 8
    for pair in pairs:
        for kind in ["gt", "reconstruction"]:
            assert hashlib.sha256((SITE / pair[kind]["path"]).read_bytes()).hexdigest() == pair[kind]["sha256"]
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(QuietHandler, directory=str(SITE)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    origin = f"http://127.0.0.1:{server.server_port}"
    errors, failed, media_requests = [], [], []

    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch()
            page = browser.new_page(viewport={"width": 1440, "height": 1050})
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.on("response", lambda response: failed.append(response.url) if response.status >= 400 else None)
            page.on("request", lambda request: media_requests.append(request.url) if ".mp4" in request.url else None)
            page.goto(origin, wait_until="networkidle")
            assert page.locator("h1").count() == 1
            assert page.locator("main > section h2").all_text_contents() == [
                "Abstract", "Architecture", "Scene decomposition", "A scene you can edit", "Results", "Related work"]
            assert page.locator("main > #decomposition").count() == 1
            assert page.locator("#results #decomposition").count() == 0
            assert page.locator("#decomposition video:visible").count() == 16
            assert page.locator("#results .result-domain > header h3").all_text_contents() == ["3D reconstruction", "Policy learning"]
            for domain in page.locator(".result-domain").all():
                assert domain.locator(".result-block > h4").all_text_contents() == ["Quantitative results", "Qualitative results"]
            assert page.locator("#policy-learning, .training-stages").count() == 0
            assert page.locator("#architecture img").evaluate("el => el.complete && el.naturalWidth === 4946")
            assert page.locator(".method-notes, .architecture-connectors").count() == 0
            assert page.locator(".scene-example").count() == 2
            assert page.locator(".reconstruction-grid video:visible").count() == 4
            assert page.locator(".component-grid video:visible").count() == 12
            assert page.locator("details video").count() == 0
            assert page.locator(".rollout-task").count() == 10
            assert page.locator(".rollout-task video:visible").count() == 10
            assert not media_requests, f"Videos should not download on page load: {media_requests}"
            for section in page.locator(".scene-example").all():
                assert section.locator("video").count() == 8
                assert section.locator(".panel-group-label").all_text_contents() == [
                    "Novel-view reconstruction", "Object decomposition", "Particle geometry"]
            for section in page.locator(".rollout-task").all():
                assert section.locator("video").count() == 1
            assert page.locator("video").evaluate_all("items => items.every(v => v.controls && v.muted && v.playsInline && !v.autoplay && v.preload === 'none' && v.poster && v.getAttribute('aria-label'))")
            assert set(page.locator("video source").evaluate_all("items => items.map(el => el.getAttribute('src'))")) == {entry["path"] for entry in entries}
            ids = page.locator("[id]").evaluate_all("items => items.map(el => el.id)")
            assert len(ids) == len(set(ids)), "Duplicate IDs"
            for link in page.locator('a[href^="#"]').all():
                assert page.locator(link.get_attribute("href")).count() == 1
            for selector, filename in [("#reconstruction-quantitative", "reconstruction-metrics-desktop.png"), (".comparison-dataset", "comparisons-desktop.png"), ("#decomposition", "decomposition-desktop.png"), ("#policy-rollouts", "rollouts-desktop.png"), (".publication-header", "header-desktop.png")]:
                if args.screenshots:
                    page.locator(selector).first.screenshot(path=str(args.screenshots / filename))

            # Each table value must match the manuscript-derived data, not a rounded summary.
            for metric in metrics:
                row = page.locator(f'.reconstruction-table tbody[data-dataset="{metric["dataset"]}"][data-setting="{metric["input"]}"] tr[data-method="{metric["method"]}"]')
                assert row.locator("td").all_text_contents() == [metric["psnr"], metric["ssim"], metric["lpips"]]

            # Sliders must reveal matched camera-view images and support native keyboard input.
            assert page.locator(".image-comparison.is-interactive").count() == 8
            assert page.locator(".comparison-range:visible").count() == 8
            for slider in page.locator(".comparison-range").all():
                slider.focus()
                page.keyboard.press("ArrowRight")
                assert slider.input_value() == "51"
                page.keyboard.press("Home")
                assert slider.input_value() == "0"
                assert slider.evaluate("el => getComputedStyle(el.closest('.image-comparison').querySelector('.comparison-gt')).clipPath") == "inset(0px 100% 0px 0px)"
                page.keyboard.press("End")
                assert slider.input_value() == "100"
                assert slider.evaluate("el => getComputedStyle(el.closest('.image-comparison').querySelector('.comparison-gt')).clipPath") == "inset(0px 0% 0px 0px)"
                slider.fill("50")
            for pair in pairs:
                for kind in ["gt", "reconstruction"]:
                    image = page.locator(f'img[src="{pair[kind]["path"]}"]')
                    image.scroll_into_view_if_needed()
                    page.wait_for_function("el => el.complete && el.naturalWidth > 0", arg=image.element_handle())
                    assert image.evaluate("el => [el.naturalWidth, el.naturalHeight]") == [pair["width"], pair["height"]]
            slider = page.locator(".comparison-range").first
            slider.scroll_into_view_if_needed()
            box = slider.bounding_box()
            page.mouse.move(box["x"] + box["width"] * .5, box["y"] + box["height"] * .5)
            page.mouse.down()
            page.mouse.move(box["x"] + box["width"] * .8, box["y"] + box["height"] * .5, steps=8)
            page.mouse.up()
            assert int(slider.input_value()) > 75, "Pointer drag did not move the image divider"
            slider.fill("50")

            # The existing independent benchmark subsections and CSV remain intact.
            assert page.locator(".benchmark h5").all_text_contents() == ["RLBench · single-view", "RLBench · multi-view", "MimicGen"]
            assert page.locator(".success-chart img").evaluate_all("imgs => imgs.every(img => img.complete && img.naturalWidth > 0)")
            assert page.locator("table:visible").count() == 1
            assert page.locator(".task-table:visible").count() == 0
            for details, count in zip(page.locator(".result-details").all(), [10, 10, 12]):
                details.locator("summary").focus()
                page.keyboard.press("Enter")
                assert details.get_attribute("open") is not None
                assert details.locator("tbody tr").count() == count
            # Figures and tables remain displayed, without separate file/download links.
            assert page.locator('a[href^="static/images/"], a[href^="static/charts/"], a[href$=".csv"], a[href$=".json"]').count() == 0
            assert page.locator('a[download]:not([href$=".mp4"])').count() == 0
            assert page.locator('a[href$="supplementary.pdf"]').count() == 0
            assert page.locator('.publication-links a').count() == 1
            assert page.locator('.video-card figcaption a[download]').count() == 26
            csv_response = page.request.get(origin + "/static/policy-results.csv")
            assert csv_response.ok
            assert len(list(csv.DictReader(csv_response.text().splitlines()))) == 140
            assert page.request.get(origin + "/static/papers/supplementary.pdf").status == 404
            resources = page.locator("a[href], img[src], video[poster], source[src], script[src], link[href]").evaluate_all("items => [...new Set(items.map(el => el.getAttribute('poster') || el.getAttribute('src') || el.getAttribute('href')))]")
            for path in resources:
                if not path.startswith(("http", "#")):
                    assert page.request.get(origin + "/" + path).ok, path

            # Decode an actual frame of every clip, not just its file extension.
            for entry in entries:
                video = page.locator(f'video:has(source[src="{entry["path"]}"])')
                video.scroll_into_view_if_needed()
                metadata = video.evaluate("""async (video) => {
                    video.load();
                    await new Promise((resolve, reject) => {
                        const timer = setTimeout(() => reject(new Error('Video metadata timeout: ' + video.currentSrc)), 12000);
                        video.addEventListener('loadedmetadata', () => { clearTimeout(timer); resolve(); }, {once: true});
                        video.addEventListener('error', () => { clearTimeout(timer); reject(new Error('Video decode failed: ' + video.currentSrc)); }, {once: true});
                    });
                    await video.play();
                    return {width: video.videoWidth, height: video.videoHeight, duration: video.duration};
                }""")
                assert metadata["width"] == entry["width"] and metadata["height"] == entry["height"], entry["path"]
                assert abs(metadata["duration"] - entry["duration"]) < 0.1, entry["path"]
                page.wait_for_function("el => el.currentTime > 0 && el.readyState >= 2", arg=video.element_handle())
                video.evaluate("el => el.pause()")

            for width in [320, 390, 768, 1024, 1440]:
                page.set_viewport_size({"width": width, "height": 900})
                assert page.evaluate("document.documentElement.scrollWidth <= innerWidth"), f"Overflow at {width}px"
                assert page.locator("video:visible").count() == 26, f"Hidden videos at {width}px"
                assert page.locator(".comparison-range:visible").count() == 8
                # Preserve each source's aspect ratio; no crop, stretch, or letterboxing.
                assert page.locator("video").evaluate_all("""videos => videos.every(video => {
                    const box = video.getBoundingClientRect();
                    const expected = Number(video.getAttribute('width')) / Number(video.getAttribute('height'));
                    return Math.abs((box.width - 2) / (box.height - 2) - expected) < 0.07;
                })"""), f"Video aspect ratio changed at {width}px"
                first, second = page.locator(".rollout-task").all()[:2]
                first_box, second_box = first.bounding_box(), second.bounding_box()
                if width > 720:
                    assert abs(first_box["y"] - second_box["y"]) < 1, "Desktop task rows not aligned"
                else:
                    assert second_box["y"] > first_box["y"] + first_box["height"], "Mobile tasks not stacked"
            page.set_viewport_size({"width": 390, "height": 844})
            scrollable = page.locator(".result-details .table-scroll").first
            assert scrollable.evaluate("el => { el.scrollLeft = 100; return el.scrollLeft > 0; }")
            if args.screenshots:
                page.locator("#decomposition").screenshot(path=str(args.screenshots / "decomposition-mobile.png"))
                page.locator("#policy-rollouts").screenshot(path=str(args.screenshots / "rollouts-mobile.png"))
                page.locator(".comparison-dataset").first.screenshot(path=str(args.screenshots / "comparisons-mobile.png"))
                page.locator("#reconstruction-quantitative").screenshot(path=str(args.screenshots / "reconstruction-metrics-mobile.png"))
            # Real touch events, not just synthetic input changes.
            mobile_context = browser.new_context(viewport={"width": 390, "height": 844}, is_mobile=True, has_touch=True)
            mobile = mobile_context.new_page()
            mobile.goto(origin, wait_until="networkidle")
            touch_slider = mobile.locator(".comparison-range").first
            touch_slider.scroll_into_view_if_needed()
            bounds = touch_slider.bounding_box()
            client = mobile_context.new_cdp_session(mobile)
            y = bounds["y"] + bounds["height"] * .5
            client.send("Input.dispatchTouchEvent", {"type": "touchStart", "touchPoints": [{"x": bounds["x"] + bounds["width"] * .5, "y": y}]})
            for position in [.45, .4, .35, .3, .25]:
                client.send("Input.dispatchTouchEvent", {"type": "touchMove", "touchPoints": [{"x": bounds["x"] + bounds["width"] * position, "y": y}]})
            client.send("Input.dispatchTouchEvent", {"type": "touchEnd", "touchPoints": []})
            assert int(touch_slider.input_value()) < 35, "Touch drag did not move the image divider"
            mobile_context.close()
            # Every clip is visible and usable even with JavaScript disabled.
            no_js = browser.new_context(java_script_enabled=False)
            fallback = no_js.new_page()
            fallback.goto(origin)
            assert fallback.locator("video:visible").count() == 26
            assert fallback.locator("details video").count() == 0
            assert fallback.locator(".comparison-layer:visible").count() == 16
            assert fallback.locator(".comparison-range:visible").count() == 0
            assert fallback.locator(".is-interactive").count() == 0
            no_js.close()
            assert not errors, errors
            assert not failed, failed
            browser.close()
            print(json.dumps({"videos_decoded": len(entries), "policy_success_clips": 10, "decomposition_clips": 16,
                              "comparison_pairs": 8, "slider_keyboard_pointer_touch": "passed", "reconstruction_metric_rows": 8,
                              "initial_video_requests": 0, "viewports_checked": [320, 390, 768, 1024, 1440],
                              "all_videos_visible": "passed", "native_no_js_fallback": "passed",
                              "console_errors": errors, "failed_requests": failed}, indent=2))
    finally:
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
