"""Render the existing Markdown report as a paginated Chinese PDF."""
from pathlib import Path
import html
import re
from markdown_it import MarkdownIt
from bs4 import BeautifulSoup
from weasyprint import HTML

import argparse
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("experiment", type=Path)
ROOT = parser.parse_args().experiment.resolve()
SOURCE = ROOT / "评测报告.md"
PDF = ROOT / "joint8-step10000_三专家采样与GB10-4090评测报告.pdf"
PRINT_HTML = ROOT / "provenance/评测报告_打印版.html"
md = MarkdownIt("commonmark", {"html": False}).enable("table")
soup = BeautifulSoup(md.render(SOURCE.read_text()), "html.parser")
# Preserve all text and tables, but make technical paths and hashes wrap.
for code in soup.find_all("code"):
    value = code.get_text()
    code.clear()
    fragments = re.split(r"(?<=[/_.-])", value)
    for frag in fragments:
        for pos in range(0, len(frag), 18):
            code.append(frag[pos:pos + 18])
            code.append(soup.new_tag("wbr"))

page_starts = {
    "权重、数据与数值协议", "本机 GB10 时延实测", "与录制 GT 的离线误差",
    "视频质量与离线演示", "GB10 与 RTX 4090 实测对照",
}
for index, heading in enumerate(soup.find_all(["h2", "h3"])):
    heading["id"] = f"section-{index + 1}"
    if heading.get_text() in page_starts:
        heading["class"] = "new-page"

# Use the spare cover space for a compact, clickable contents list.
contents = soup.new_tag("nav", attrs={"class": "contents"})
contents_title = soup.new_tag("h3")
contents_title.string = "阅读导航"
contents.append(contents_title)
for heading in soup.find_all("h2")[1:]:
    entry = soup.new_tag("a", href="#" + heading["id"])
    entry.string = heading.get_text()
    contents.append(entry)
soup.find("h2", string="权重、数据与数值协议").insert_before(contents)

# Keep the bootstrap table together and put its explanatory sentence on the same page.
for paragraph in soup.find_all("p"):
    if paragraph.get_text().startswith("事件窗口按 episode 配对 bootstrap"):
        paragraph["class"] = "new-page keep-next"

for image in soup.find_all("img"):
    original = image["src"]
    assert (ROOT / original).is_file(), original
    image["src"] = (ROOT / original).as_uri()
    paragraph = image.parent
    if paragraph.name == "p" and len(paragraph.find_all("img")) == 1:
        paragraph["class"] = "figure"
        caption = soup.new_tag("span", attrs={"class": "caption"})
        caption.string = image.get("alt", "")
        paragraph.append(caption)
# The two quality plots are on a separate page, preserving readable labels.
quality = soup.find("img", src=lambda src: bool(src and src.endswith("speed-quality.png")))
if quality:
    quality.parent["class"] = "figure new-page quality-figures"
    quality_heading = soup.new_tag("h2", attrs={"class": "new-page"})
    quality_heading.string = "采样配置的速度与质量分布"
    quality.parent.insert_before(quality_heading)
    quality.parent["class"] = "figure quality-figures"

for table in soup.find_all("table"):
    if len(table.find_all("th")) >= 7:
        table["class"] = "wide"
    for row in table.find_all("tr"):
        cells = row.find_all("td")
        if cells and cells[0].get_text(strip=True) == "4/4/16":
            row["class"] = "recommended"
    table_head = table.find("thead")
    if table_head and "GB10/4090" in table_head.get_text():
        headings = ["V/F/A", "核心 GB10<br>P50 / P95", "核心 4090<br>P50 / P95",
                    "RGB GB10<br>P50 / P95", "RGB 4090<br>P50 / P95",
                    "GB10/4090<br>P50 比", "相对 16 步提速<br>GB10 / 4090"]
        for th, text in zip(table.find_all("th"), headings):
            th.clear()
            th.append(BeautifulSoup(text, "html.parser"))
        note = soup.new_tag("p", attrs={"class": "table-note"})
        note.string = "RGB 指 RGB→动作/力输出。延时单位：ms；粗体着色行为推荐候选。"
        table.insert_after(note)

# Add an unobtrusive print-edition note; videos remain external assets.
footer_note = soup.new_tag("p", attrs={"class": "print-note"})
footer_note.string = "PDF 排版版 · 来源：评测报告.md · 图表已嵌入；演示视频及原始数据为外部文件，需与原结果目录配套打开。"
soup.append(footer_note)

CSS = r"""
@page {
  size: A4;
  margin: 17mm 16mm 17mm;
  @top-left { content: "JOINT8 / STEP 10000"; font: 8pt "Noto Sans CJK SC"; color: #718096; }
  @top-right { content: "三专家采样与推理机评测"; font: 8pt "Noto Sans CJK SC"; color: #718096; }
  @bottom-left { content: "2026-10-21 · 离线评测"; font: 8pt "Noto Sans CJK SC"; color: #718096; }
  @bottom-right { content: counter(page) " / " counter(pages); font: 8pt "Noto Sans CJK SC"; color: #718096; }
}
@page:first { @top-left { content: none; } @top-right { content: none; } }
html { color: #203040; font-family: "Noto Sans CJK SC", sans-serif; font-size: 10pt; line-height: 1.63; }
body { margin: 0; }
h1 { font-size: 23pt; line-height: 1.35; color: #12364a; margin: 4mm 0 7mm; padding-bottom: 5mm; border-bottom: 2pt solid #1a7d8b; letter-spacing: -.2pt; }
h2 { font-size: 15pt; line-height: 1.35; margin: 6mm 0 3mm; color: #12364a; border-bottom: .6pt solid #c6d9df; padding-bottom: 2mm; }
h3 { font-size: 12.5pt; margin: 6mm 0 3mm; color: #145d6b; }
h1,h2,h3 { break-after: avoid; }
p { margin: 0 0 3mm; orphans: 3; widows: 3; }
strong { font-weight: 700; color: #123c51; }
a { color: #116f83; text-decoration: none; }
code { font-family: "DejaVu Sans Mono", "Noto Sans CJK SC", monospace; font-size: 8.1pt; color: #334960; overflow-wrap: anywhere; }
ul { padding-left: 5mm; margin-top: 2mm; }
li { margin-bottom: 2.8mm; padding-left: 1mm; orphans: 2; widows: 2; }
table { width: 100%; border-collapse: collapse; margin: 3mm 0 4mm; font-size: 8.8pt; line-height: 1.45; table-layout: fixed; }
th { color: #fff; background: #194a60; font-weight: 500; text-align: center !important; padding: 2.2mm 1.3mm; vertical-align: middle; }
td { padding: 2mm 1.3mm; border-bottom: .5pt solid #d8e2e8; text-align: center !important; vertical-align: middle; }
tbody tr:nth-child(even) { background: #f3f7f9; }
tr.recommended { background: #e4f2ef !important; }
tr.recommended td { font-weight: 700; color: #145d55; }
tr { break-inside: avoid; }
thead { display: table-header-group; }
table.wide { font-size: 8pt; }
table.wide td { padding: 1.7mm .8mm; }
table.wide th { padding: 2mm .8mm; }
.table-note { color: #677787; font-size: 8pt; margin-top: -2mm; }
img { max-width: 100%; max-height: 112mm; object-fit: contain; display: block; margin: 0 auto; }
.figure { break-inside: avoid; margin: 4mm 0; }
.caption { display: block; color: #677787; font-size: 8.5pt; text-align: center; margin-top: 1mm; }
.quality-figures img { max-height: 114mm; }
.new-page { break-before: page; }
h2.new-page,h3.new-page { margin-top: 1mm; }
.print-note { font-size: 8pt; color: #677787; padding-top: 4mm; border-top: .5pt solid #d8e2e8; margin-top: 7mm; }
.keep-next { break-after: avoid; }
.contents { margin-top: 8mm; font-size: 9pt; }
.contents h3 { font-size: 11pt; margin-bottom: 2mm; }
.contents a { display: block; padding: 1mm 0; border-bottom: .4pt solid #e4ebef; }
.contents a::after { content: leader(dotted) target-counter(attr(href), page); color: #677787; }
"""
page = '<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><title>joint8-step10000 三专家采样与 GB10 / RTX 4090 评测报告</title><style>' + CSS + '</style></head><body>' + str(soup) + '</body></html>'
PRINT_HTML.write_text(page)
HTML(string=page, base_url=str(ROOT)).write_pdf(PDF)
print(PDF)
print(f"Bytes: {PDF.stat().st_size}")
