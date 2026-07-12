import argparse
import re
import shutil
import subprocess
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from tqdm import tqdm
from playwright.sync_api import sync_playwright

try:
    import imageio_ffmpeg
except ImportError:
    imageio_ffmpeg = None


VIDEO_PATTERNS = (
    ".mp4",
    ".m3u8",
    ".webm",
    ".mov",
)

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/126.0.0.0 Safari/537.36"
)


def safe_filename(text: str) -> str:
    text = text.strip()
    text = re.sub(r"[^\w\s.-]", "", text, flags=re.UNICODE)
    text = re.sub(r"\s+", "_", text)
    return text[:120] or "match_video"


def get_match_id(url: str) -> str:
    parts = urlparse(url).path.rstrip("/").split("/")
    match_id = parts[-1] if parts else "match"
    return safe_filename(match_id)


def get_output_filename(page_url: str) -> str:
    match_id = get_match_id(page_url)
    return f"Game_{match_id}.mp4"


def is_blocked_video_host(url: str) -> bool:
    host = urlparse(url).netloc.lower()

    blocked_hosts = [
        "pubads.g.doubleclick.net",
        "doubleclick.net",
        "googlevideo.com",
        "youtube.com",
        "googlesyndication.com",
        "imasdk.googleapis.com",
    ]

    return any(b in host for b in blocked_hosts)


def looks_like_video_url(url: str, content_type: str = "") -> bool:
    if not url:
        return False

    if is_blocked_video_host(url):
        return False

    parsed = urlparse(url)
    path = parsed.path.lower()
    ct = content_type.lower()

    if path.endswith(VIDEO_PATTERNS):
        return True

    if "video/" in ct:
        return True

    if "mpegurl" in ct:
        return True

    if "application/vnd.apple.mpegurl" in ct:
        return True

    return False


def normalize_request_headers(headers: dict, page_url: str, force_range: bool = True) -> dict:
    cleaned = {}

    skip_keys = {
        "host",
        "content-length",
        "connection",
        "accept-encoding",
    }

    for key, value in headers.items():
        lk = key.lower()

        if lk in skip_keys:
            continue

        cleaned[key] = value

    cleaned["User-Agent"] = cleaned.get("User-Agent", USER_AGENT)
    cleaned["Referer"] = cleaned.get("Referer", page_url)
    cleaned["Accept"] = cleaned.get("Accept", "video/mp4,video/*,*/*;q=0.9")
    cleaned["Accept-Language"] = cleaned.get(
        "Accept-Language",
        "tr-TR,tr;q=0.9,en-US;q=0.8,en;q=0.7",
    )

    if force_range:
        cleaned["Range"] = "bytes=0-"

    return cleaned


def extract_angle_from_video_url(video_url: str):
    """Dosya adında kamera/açı bilgisi varsa döndür."""
    filename = Path(urlparse(video_url).path).name.lower()
    patterns = [
        r"\.(\d+)-\d+\.(?:mp4|m3u8|webm|mov)$",
        r"angle[_-]?(\d+)",
        r"camera[_-]?(\d+)",
        r"cam[_-]?(\d+)",
        r"aci[_-]?(\d+)",
        r"açı[_-]?(\d+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, filename, flags=re.IGNORECASE)
        if match:
            try:
                return int(match.group(1))
            except (TypeError, ValueError):
                pass
    return None


def click_camera_angle(page, angle: int) -> bool:
    """Sayfadaki kamera/açı düğmesini görünür metin ve data alanlarıyla bul."""
    script = """
    (angleText) => {
        const norm = (value) => String(value || "")
            .toLocaleLowerCase("tr-TR")
            .replace(/ı/g, "i").replace(/ş/g, "s").replace(/ğ/g, "g")
            .replace(/ü/g, "u").replace(/ö/g, "o").replace(/ç/g, "c")
            .replace(/\\s+/g, " ").trim();
        const visible = (el) => {
            const r = el.getBoundingClientRect();
            const s = window.getComputedStyle(el);
            return r.width > 0 && r.height > 0 && s.display !== "none"
                && s.visibility !== "hidden" && s.opacity !== "0";
        };
        const target = norm(angleText);
        const nodes = Array.from(document.querySelectorAll(
            "button, a, li, label, [role='button'], [data-camera], " +
            "[data-angle], [data-id], [data-index]"
        ));
        const scored = nodes.map((el) => {
            if (!visible(el)) return {el, score: -1};
            const shown = norm(el.innerText || el.textContent || "");
            const data = norm([
                el.getAttribute("aria-label"), el.getAttribute("title"),
                el.getAttribute("data-camera"), el.getAttribute("data-angle"),
                el.getAttribute("data-id"), el.getAttribute("data-index"),
                el.id, el.className
            ].join(" "));
            const all = `${shown} ${data}`;
            let score = 0;
            if (shown === target) score += 90;
            if (shown === `kamera ${target}` || shown === `camera ${target}`
                    || shown === `aci ${target}`) score += 120;
            if (all.includes(`kamera ${target}`) || all.includes(`camera ${target}`)
                    || all.includes(`aci ${target}`)) score += 70;
            if (norm(el.getAttribute("data-camera")) === target
                    || norm(el.getAttribute("data-angle")) === target) score += 150;
            if (all.includes("reklam") || all.includes("google")) score -= 250;
            return {el, score, shown};
        }).filter((item) => item.score > 0).sort((a, b) => b.score - a.score);
        if (!scored.length) return {clicked: false};
        const best = scored[0];
        best.el.scrollIntoView({block: "center", inline: "center"});
        best.el.click();
        return {clicked: true, score: best.score, text: best.shown};
    }
    """
    try:
        result = page.evaluate(script, str(angle))
        if result and result.get("clicked"):
            print(
                f"[*] {angle}. kamera/açı seçildi "
                f"(eşleşme={result.get('score')}, metin='{result.get('text', '')}')."
            )
            return True
    except Exception as exc:
        print(f"[!] Kamera/açı seçimi sırasında hata: {exc}")
    print(f"[!] {angle}. kamera/açı düğmesi otomatik bulunamadı.")
    return False


def extract_video_urls(
    page_url: str,
    headed: bool,
    timeout_ms: int,
    target_angle: int = 1,
    wait_seconds: int = 20,
):
    candidates = []
    seen = set()
    candidate_headers = {}

    def clear_candidates():
        candidates.clear()
        seen.clear()
        candidate_headers.clear()

    def add_candidate(url, reason="", status=None, content_type="", request_headers=None):
        if not url:
            return

        absolute = urljoin(page_url, url)

        if absolute in seen:
            return

        if is_blocked_video_host(absolute):
            return

        seen.add(absolute)
        candidates.append(absolute)

        if request_headers:
            candidate_headers[absolute] = normalize_request_headers(
                request_headers,
                page_url,
                force_range=True,
            )

        extra = ""

        if status is not None:
            extra += f" status={status}"

        if content_type:
            extra += f" content-type={content_type}"

        detected_angle = extract_angle_from_video_url(absolute)
        if detected_angle is not None:
            extra += f" detected_angle={detected_angle}"

        print(f"[+] Video adayı yakalandı: {absolute}")

        if reason:
            print(f"    Sebep: {reason}{extra}")

    profile_dir = Path.home() / ".sosyalhalisaha_downloader_profile"

    with sync_playwright() as p:
        context = p.chromium.launch_persistent_context(
            user_data_dir=str(profile_dir),
            headless=not headed,
            user_agent=USER_AGENT,
            viewport={"width": 1366, "height": 768},
        )

        page = context.new_page()

        def on_response(response):
            try:
                url = response.url
                headers = response.headers
                content_type = headers.get("content-type", "")
                status = response.status
                request_headers = response.request.headers

                if status in (200, 206) and looks_like_video_url(url, content_type):
                    add_candidate(
                        url,
                        reason="network response",
                        status=status,
                        content_type=content_type,
                        request_headers=request_headers,
                    )

            except Exception:
                pass

        page.on("response", on_response)

        print("[*] Sayfa açılıyor...")
        page.goto(page_url, wait_until="domcontentloaded", timeout=timeout_ms)

        try:
            page.wait_for_load_state("networkidle", timeout=15000)
        except Exception:
            pass

        page.wait_for_timeout(2000)
        if target_angle != 1:
            print(f"[*] {target_angle}. kamera/açı seçilmeye çalışılıyor...")
            if click_camera_angle(page, target_angle):
                clear_candidates()
                page.wait_for_timeout(3000)
            elif headed:
                print(
                    "[!] Açık tarayıcıdan hedef açıyı elle seçip videoyu "
                    "başlatabilirsiniz."
                )

        print("[*] Sayfadaki video elementleri ve HTML taranıyor...")

        try:
            dom_urls = page.evaluate(
                """
                () => {
                    const urls = [];

                    document.querySelectorAll("video, video source, source").forEach(el => {
                        if (el.src) urls.push(el.src);
                        if (el.currentSrc) urls.push(el.currentSrc);
                    });

                    return urls;
                }
                """
            )

            for u in dom_urls:
                if looks_like_video_url(u):
                    add_candidate(u, "DOM video/source elementi")

        except Exception:
            pass

        try:
            html = page.content()

            regex_urls = re.findall(
                r"""https?://[^"'<>\\\s]+?(?:\.mp4|\.m3u8|\.webm|\.mov)(?:\?[^"'<>\\\s]*)?""",
                html,
                flags=re.IGNORECASE,
            )

            for u in regex_urls:
                if looks_like_video_url(u):
                    add_candidate(u, "HTML içinde regex")

        except Exception:
            pass

        print("[*] Video oynatmayı tetiklemeye çalışıyorum...")

        try:
            page.evaluate(
                """
                () => {
                    document.querySelectorAll("video").forEach(v => {
                        try {
                            v.muted = true;
                            v.play();
                        } catch(e) {}
                    });
                }
                """
            )
        except Exception:
            pass

        try:
            page.mouse.click(683, 384)
        except Exception:
            pass

        print(
            f"[*] {wait_seconds} saniye video/network bekleniyor. "
            "Gerekirse tarayıcıda reklamı geçip videoyu başlat."
        )
        page.wait_for_timeout(max(1, wait_seconds) * 1000)

        cookies = context.cookies()
        context.close()

    return candidates, cookies, candidate_headers


def cookies_to_requests_jar(cookies):
    jar = requests.cookies.RequestsCookieJar()

    for c in cookies:
        jar.set(
            c.get("name"),
            c.get("value"),
            domain=c.get("domain"),
            path=c.get("path", "/"),
        )

    return jar


def cookies_to_header(cookies):
    pairs = []

    for c in cookies:
        name = c.get("name")
        value = c.get("value")

        if name and value:
            pairs.append(f"{name}={value}")

    return "; ".join(pairs)


def browser_like_headers(page_url: str, range_header: str | None = "bytes=0-"):
    headers = {
        "User-Agent": USER_AGENT,
        "Referer": page_url,
        "Accept": "video/mp4,video/*,*/*;q=0.9",
        "Accept-Language": "tr-TR,tr;q=0.9,en-US;q=0.8,en;q=0.7",
        "Sec-Fetch-Dest": "video",
        "Sec-Fetch-Mode": "no-cors",
        "Sec-Fetch-Site": "cross-site",
    }

    if range_header:
        headers["Range"] = range_header

    return headers


def get_headers_for_url(video_url: str, page_url: str, candidate_headers: dict):
    headers = candidate_headers.get(video_url)

    if headers:
        return headers

    return browser_like_headers(page_url, range_header="bytes=0-")


def validate_video_url(video_url: str, page_url: str, cookies, candidate_headers: dict) -> bool:
    if is_blocked_video_host(video_url):
        return False

    session = requests.Session()
    session.cookies = cookies_to_requests_jar(cookies)

    headers = get_headers_for_url(video_url, page_url, candidate_headers)

    try:
        r = session.get(video_url, headers=headers, stream=True, timeout=25)
        content_type = r.headers.get("content-type", "").lower()

        print(f"[*] Test: {r.status_code} | {content_type} | {video_url}")

        if r.status_code not in (200, 206):
            return False

        if "video" in content_type:
            return True

        if "mpegurl" in content_type:
            return True

        if video_url.lower().split("?")[0].endswith(VIDEO_PATTERNS):
            return True

        return False

    except Exception as e:
        print(f"[!] Test başarısız: {video_url}")
        print(f"    {e}")
        return False


def choose_best_video_url(
    candidates,
    page_url,
    cookies,
    candidate_headers,
    target_angle: int = 1,
):
    if not candidates:
        return None

    clean = []

    for u in candidates:
        if is_blocked_video_host(u):
            continue

        if u not in clean:
            clean.append(u)

    angle_matches = [
        u for u in clean if extract_angle_from_video_url(u) == target_angle
    ]
    mp4s = [u for u in clean if ".mp4" in u.lower()]
    m3u8s = [u for u in clean if ".m3u8" in u.lower()]
    others = [u for u in clean if u not in mp4s and u not in m3u8s]

    if angle_matches:
        ordered = angle_matches + [
            u for u in mp4s + m3u8s + others if u not in angle_matches
        ]
        print(f"[*] Dosya adından {target_angle}. açıyla eşleşen video bulundu.")
    else:
        ordered = mp4s + m3u8s + others
        if target_angle != 1:
            print(
                f"[!] URL adından {target_angle}. açı doğrulanamadı; "
                "açı seçildikten sonra yakalanan ilk uygun video kullanılacak."
            )

    for u in ordered:
        if validate_video_url(u, page_url, cookies, candidate_headers):
            return u

    print("\n[!] Video adayları bulundu ama hiçbiri requests ile doğrulanamadı.")
    print("    Ama tarayıcıda 206 göründüyse video var olabilir.")
    print("    Şimdi yine de ilk gerçek maç videosunu indirmeyi deneyeceğiz.")

    if ordered:
        return ordered[0]

    return None


def download_direct_video(video_url: str, out_path: Path, page_url: str, cookies, candidate_headers: dict):
    session = requests.Session()
    session.cookies = cookies_to_requests_jar(cookies)

    headers = get_headers_for_url(video_url, page_url, candidate_headers)
    headers["Range"] = "bytes=0-"

    with session.get(video_url, headers=headers, stream=True, timeout=60) as r:
        print(f"[*] İndirme cevabı: {r.status_code} | {r.headers.get('content-type', '')}")

        r.raise_for_status()

        total = int(r.headers.get("content-length", 0))

        with open(out_path, "wb") as f, tqdm(
            total=total,
            unit="B",
            unit_scale=True,
            desc=out_path.name,
        ) as bar:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))


def download_hls_with_ffmpeg(video_url: str, out_path: Path, page_url: str, cookies, candidate_headers: dict):
    ffmpeg_exe = shutil.which("ffmpeg")
    if ffmpeg_exe is None and imageio_ffmpeg is not None:
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    if ffmpeg_exe is None:
        raise RuntimeError(
            "Bu video m3u8/HLS gibi görünüyor ama ffmpeg bulunamadı. "
            "Araçlar sekmesinden eksik Python paketlerini kurun."
        )

    cookie_header = cookies_to_header(cookies)

    headers = get_headers_for_url(video_url, page_url, candidate_headers)

    ffmpeg_headers = ""

    for key, value in headers.items():
        if key.lower() == "range":
            continue

        ffmpeg_headers += f"{key}: {value}\r\n"

    if cookie_header:
        ffmpeg_headers += f"Cookie: {cookie_header}\r\n"

    cmd = [
        ffmpeg_exe,
        "-y",
        "-headers",
        ffmpeg_headers,
        "-i",
        video_url,
        "-c",
        "copy",
        str(out_path),
    ]

    print("[*] ffmpeg ile indiriliyor...")
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("url", nargs="?", help="Sosyal Halı Saha maç detay linki")
    parser.add_argument("-o", "--output", default="downloads", help="Kayıt klasörü")
    parser.add_argument("--headed", action="store_true", help="Tarayıcıyı görünür açar")
    parser.add_argument("--timeout", type=int, default=60000, help="Sayfa açma timeout ms")
    parser.add_argument("--angle", type=int, default=1, help="Kamera/açı numarası")
    parser.add_argument("--wait", type=int, default=20, help="Network bekleme süresi (sn)")
    args = parser.parse_args()

    page_url = args.url or input("Maç linkini gir: ").strip()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    candidates, cookies, candidate_headers = extract_video_urls(
        page_url=page_url,
        headed=args.headed,
        timeout_ms=args.timeout,
        target_angle=args.angle,
        wait_seconds=args.wait,
    )

    video_url = choose_best_video_url(
        candidates=candidates,
        page_url=page_url,
        cookies=cookies,
        candidate_headers=candidate_headers,
        target_angle=args.angle,
    )

    if not video_url:
        print("\n[!] Video URL bulunamadı.")
        print("    Şunu dene:")
        print(
            f"    python download_sosyalhalisaha.py LINK --headed "
            f"--angle {args.angle} --wait 35"
        )
        print(
            "    Açılan tarayıcıda gerekirse giriş yap, reklamı geç, "
            "doğru açıyı seç ve videoyu başlat."
        )
        return

    print(f"\n[*] Seçilen video URL:")
    print(video_url)
    detected_angle = extract_angle_from_video_url(video_url)
    if detected_angle is not None:
        print(f"[*] Dosya adından algılanan kamera açısı: {detected_angle}")
    if detected_angle is not None and detected_angle != args.angle:
        print(
            f"[!] İstenen açı {args.angle}, ancak seçilen dosya "
            f"{detected_angle}. açı gibi görünüyor."
        )

    output_filename = get_output_filename(page_url)
    if args.angle != 1:
        output_path = Path(output_filename)
        output_filename = f"{output_path.stem}_angle_{args.angle}{output_path.suffix}"
    out_path = out_dir / output_filename

    print(f"[*] Kayıt adı: {output_filename}")

    if ".m3u8" in video_url.lower():
        download_hls_with_ffmpeg(video_url, out_path, page_url, cookies, candidate_headers)
    else:
        download_direct_video(video_url, out_path, page_url, cookies, candidate_headers)

    print(f"\n[✓] İndirildi: {out_path.resolve()}")


if __name__ == "__main__":
    main()
