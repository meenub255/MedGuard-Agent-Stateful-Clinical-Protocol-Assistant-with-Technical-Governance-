"""Capture UI screenshots and add annotated arrows. Usage: python docs/make_screenshots.py"""
import subprocess, sys, time, os, urllib.request
from playwright.sync_api import sync_playwright
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "docs", "images"); os.makedirs(OUT, exist_ok=True)
PORT = 8599
env = {k: v for k, v in os.environ.items() if k not in ("OPENAI_API_KEY", "TAVILY_API_KEY")}
srv = subprocess.Popen([sys.executable, "-m", "streamlit", "run", "app.py", "--server.port", str(PORT),
                        "--server.headless", "true", "--browser.gatherUsageStats", "false"], cwd=ROOT, env=env,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
for _ in range(60):
    try: urllib.request.urlopen(f"http://localhost:{PORT}"); break
    except Exception: time.sleep(1)

try: FONT = ImageFont.truetype("arialbd.ttf", 22)
except Exception: FONT = ImageFont.load_default()

def box(page, locator):
    b = locator.first.bounding_box(); return b

def annotate(path, notes):
    """notes: list of (n, label, target_xy, label_xy)"""
    im = Image.open(path).convert("RGB"); d = ImageDraw.Draw(im)
    red = (220, 38, 38)
    for n, label, (tx, ty), (lx, ly) in notes:
        d.line([(lx, ly), (tx, ty)], fill=red, width=4)
        # arrow head
        import math
        a = math.atan2(ty - ly, tx - lx)
        for s in (0.45, -0.45):
            d.line([(tx, ty), (tx - 22 * math.cos(a - s), ty - 22 * math.sin(a - s))], fill=red, width=4)
        text = f" {n}. {label} "
        w = d.textlength(text, font=FONT)
        d.rounded_rectangle([lx - w / 2, ly - 18, lx + w / 2, ly + 18], 8, fill=red)
        d.text((lx - w / 2, ly - 14), text, fill="white", font=FONT)
    im.save(path)

def ctr(b, dx=0.5, dy=0.5): return (b["x"] + b["width"] * dx, b["y"] + b["height"] * dy)

with sync_playwright() as p:
    br = p.chromium.launch(channel="chrome", headless=True)
    pg = br.new_page(viewport={"width": 1500, "height": 1000})
    pg.goto(f"http://localhost:{PORT}"); pg.wait_for_selector("text=MedGuard", timeout=60000); time.sleep(2)

    # 1. Home
    f = f"{OUT}/01_home.png"; pg.screenshot(path=f)
    annotate(f, [(1, "Optional API keys (session only)", (262, 178), (640, 60)),
                 (2, "Governance rules enforced", (240, 520), (700, 580)),
                 (3, "One-click example questions", (770, 266), (1130, 90)),
                 (4, "Ask any clinical question", (1000, 350), (1100, 480))])

    def ask(q):
        pg.get_by_placeholder("e.g. Which herbs interact with anticoagulants?").fill(q)
        pg.keyboard.press("Enter"); time.sleep(1)
        pg.get_by_role("button", name="Ask", exact=True).click()
        pg.wait_for_selector("text=Answer", timeout=60000); time.sleep(3)

    # 2. Pass
    ask("How often should type 2 diabetes patients get eye screening?")
    f = f"{OUT}/02_grounded_answer.png"; pg.screenshot(path=f)
    annotate(f, [(1, "Governance passed", (990, 447), (1180, 385)),
                 (2, "Live LangGraph trace", (1230, 580), (1200, 700)),
                 (3, "Cited answer + disclaimer tag", (985, 630), (1130, 790)),
                 (4, "Retrieved source passages", (995, 795), (1150, 890))])

    # 3. Refusal
    ask("What is the best treatment for a broken leg?")
    f = f"{OUT}/03_safe_refusal.png"; pg.screenshot(path=f)
    annotate(f, [(1, "No guideline match: nothing fabricated", (990, 447), (1130, 385)),
                 (2, "Trace: fail, web fallback, refusal", (1250, 690), (1190, 780)),
                 (3, "Disclaimer still attached", (960, 650), (1140, 860))])
    br.close()
srv.terminate()
print("done")
