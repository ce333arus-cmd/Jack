// Renders every SVG in ./svg onto black into ./preview/*.png (needs playwright).
const { chromium } = require('playwright');
const fs = require('fs');
const path = require('path');

(async () => {
  const dir = path.join(__dirname, 'svg');
  const out = path.join(__dirname, 'preview');
  fs.mkdirSync(out, { recursive: true });
  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: 1600, height: 1000 } });
  for (const f of fs.readdirSync(dir).filter(f => f.endsWith('.svg'))) {
    const svg = fs.readFileSync(path.join(dir, f), 'utf8')
      .replace(/width="[^"]+mm" height="[^"]+mm"/, 'width="100%" height="100%"');
    await page.setContent(`<html><body style="margin:0;background:#120d14;width:1600px;height:1000px;display:flex">
      <div style="margin:40px;flex:1">${svg}</div></body></html>`);
    await page.screenshot({ path: path.join(out, f.replace('.svg', '.png')) });
  }
  await browser.close();
})();
