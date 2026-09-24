const fs = require('fs');
const path = require('path');
const { chromium } = require('playwright-core');

function clean(value) {
  return String(value || '').replace(/\s+/g, ' ').trim();
}

function safeFilename(value) {
  return value.replace(/[<>:"/\\|?*\u0000-\u001f]/g, '_').replace(/[. ]+$/g, '').slice(0, 180);
}

function filenameFromDisposition(disposition, fallbackTitle) {
  const encoded = disposition.match(/filename\*=utf-8''([^;]+)/i)?.[1];
  if (encoded) {
    try { return safeFilename(decodeURIComponent(encoded.replace(/^"|"$/g, ''))); } catch {}
  }
  const plain = disposition.match(/filename="?([^";]+)"?/i)?.[1];
  if (plain) {
    try { return safeFilename(decodeURIComponent(plain)); } catch { return safeFilename(plain); }
  }
  return `${safeFilename(fallbackTitle)}.pdf`;
}

async function findDetail(page, title) {
  await page.goto('https://www.cnki.net/', { waitUntil: 'domcontentloaded', timeout: 60000 });
  const input = page.locator('#txt_SearchText');
  await input.waitFor({ state: 'visible', timeout: 30000 });
  await input.fill(title);
  await input.press('Enter');
  await page.waitForLoadState('domcontentloaded', { timeout: 60000 }).catch(() => null);
  await page.waitForTimeout(3000);

  const links = await page.evaluate(() => [...document.querySelectorAll('a.fz14, a[href*="/article/abstract"]')]
    .map((anchor) => ({ title: String(anchor.innerText || anchor.title || '').replace(/\s+/g, ' ').trim(), url: anchor.href }))
    .filter((item, index, all) => item.title && item.url && all.findIndex((other) => other.url === item.url) === index));
  return links.find((item) => item.title === title) || links.find((item) => item.title.includes(title) || title.includes(item.title));
}

async function main() {
  const outputDir = process.argv[2];
  const titles = process.argv.slice(3).map(clean).filter(Boolean);
  if (!outputDir || !titles.length) throw new Error('Usage: node download-cnki-pdfs.js <output-dir> <title> [title ...]');
  fs.mkdirSync(outputDir, { recursive: true });

  const browser = await chromium.connectOverCDP('http://127.0.0.1:9222');
  const context = browser.contexts()[0];
  let page = context.pages().find((candidate) => /cnki\.net/i.test(candidate.url())) || await context.newPage();
  const results = [];

  for (let index = 0; index < titles.length; index += 1) {
    const title = titles[index];
    process.stderr.write(`[${index + 1}/${titles.length}] Searching: ${title}\n`);
    try {
      const detail = await findDetail(page, title);
      if (!detail) throw new Error('No exact CNKI result found.');
      await page.goto(detail.url, { waitUntil: 'domcontentloaded', timeout: 60000 });
      await page.waitForTimeout(2200);
      const pdfUrl = await page.getByText('PDF下载', { exact: true }).first().getAttribute('href');
      if (!pdfUrl) throw new Error('PDF download link is unavailable.');

      const response = await context.request.get(new URL(pdfUrl, page.url()).href, {
        headers: { referer: page.url() },
        timeout: 180000,
      });
      const body = await response.body();
      const contentType = response.headers()['content-type'] || '';
      if (!response.ok() || !/^application\/pdf/i.test(contentType) || body.subarray(0, 5).toString() !== '%PDF-') {
        throw new Error(`Unexpected response: HTTP ${response.status()}, ${contentType}, ${body.length} bytes`);
      }

      let filename = filenameFromDisposition(response.headers()['content-disposition'] || '', detail.title);
      if (!/\.pdf$/i.test(filename)) filename += '.pdf';
      let destination = path.join(outputDir, filename);
      if (fs.existsSync(destination)) {
        const existing = fs.statSync(destination);
        if (existing.size === body.length) {
          results.push({ title: detail.title, status: 'already_present', path: destination, size: body.length });
          process.stderr.write(`  Already present: ${destination}\n`);
          continue;
        }
        destination = path.join(outputDir, `${path.basename(filename, '.pdf')}_CNKI.pdf`);
      }
      fs.writeFileSync(destination, body);
      results.push({ title: detail.title, status: 'downloaded', path: destination, size: body.length });
      process.stderr.write(`  Saved: ${destination} (${body.length} bytes)\n`);
    } catch (error) {
      results.push({ title, status: 'failed', error: error.message || String(error) });
      process.stderr.write(`  Failed: ${error.message || error}\n`);
    }
    await page.waitForTimeout(1800);
  }

  process.stdout.write(`${JSON.stringify(results, null, 2)}\n`);
  await browser.close();
}

main().catch((error) => {
  console.error(error.stack || error.message || error);
  process.exit(1);
});
